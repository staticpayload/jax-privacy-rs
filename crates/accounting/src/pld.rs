//! Privacy Loss Distribution (PLD) accounting.
//!
//! This module implements a discretized PLD accountant inspired by the
//! reference Python implementation. The implementation focuses on Gaussian
//! mechanisms and Poisson subsampling, which cover DP-SGD and BandMF usage.

use std::collections::HashMap;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use statrs::distribution::{Binomial, ContinuousCDF, DiscreteCDF, Normal};

use jax_privacy_core::clipping::NeighboringRelation;

const DEFAULT_VALUE_DISCRETIZATION_INTERVAL: f64 = 1e-4;
const DEFAULT_LOG_MASS_TRUNCATION_BOUND: f64 = -50.0;
const DEFAULT_TAIL_MASS_TRUNCATION: f64 = 1e-15;

#[derive(Clone, Copy, Debug)]
enum AdjacencyType {
    Add,
    Remove,
    Replace,
}

#[derive(Clone, Debug)]
struct TailPrivacyLossDistribution {
    lower_x_truncation: f64,
    upper_x_truncation: f64,
    tail_mass: Vec<(f64, f64)>,
}

trait MonotonePrivacyLoss {
    fn privacy_loss(&self, x: f64) -> f64;
    fn inverse_privacy_loss(&self, loss: f64) -> f64;
    fn mu_upper_cdf(&self, x: f64) -> f64;
    fn privacy_loss_tail(&self) -> TailPrivacyLossDistribution;
}

fn binary_search_increasing<F>(f: F, target: f64, mut lower: f64, mut upper: f64, tol: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    for _ in 0..120 {
        let mid = 0.5 * (lower + upper);
        let val = f(mid);
        if (upper - lower).abs() <= tol {
            return mid;
        }
        if val < target {
            lower = mid;
        } else {
            upper = mid;
        }
    }
    0.5 * (lower + upper)
}

#[derive(Clone, Debug)]
struct GaussianPrivacyLoss {
    standard_deviation: f64,
    sensitivity: f64,
    sampling_prob: f64,
    adjacency_type: AdjacencyType,
    pessimistic_estimate: bool,
    log_mass_truncation_bound: f64,
    normal: Normal,
    variance: f64,
}

impl GaussianPrivacyLoss {
    fn new(
        standard_deviation: f64,
        sensitivity: f64,
        sampling_prob: f64,
        adjacency_type: AdjacencyType,
        pessimistic_estimate: bool,
        log_mass_truncation_bound: f64,
    ) -> Self {
        let normal = Normal::new(0.0, standard_deviation).expect("normal");
        let variance = standard_deviation * standard_deviation;
        Self {
            standard_deviation,
            sensitivity,
            sampling_prob,
            adjacency_type,
            pessimistic_estimate,
            log_mass_truncation_bound,
            normal,
            variance,
        }
    }

    fn noise_cdf(&self, x: f64) -> f64 {
        self.normal.cdf(x)
    }

    fn noise_log_cdf(&self, x: f64) -> f64 {
        self.normal.ln_cdf(x)
    }

    fn privacy_loss_without_subsampling_remove(&self, x: f64) -> f64 {
        let s = self.sensitivity;
        s * (-0.5 * s - x) / self.variance
    }

    fn privacy_loss_remove(&self, x: f64) -> f64 {
        let loss = self.privacy_loss_without_subsampling_remove(x);
        if (self.sampling_prob - 1.0).abs() < 1e-12 {
            return loss;
        }
        let q = self.sampling_prob;
        (q * loss.exp() + (1.0 - q)).ln()
    }

    fn privacy_loss_add(&self, x: f64) -> f64 {
        -self.privacy_loss_remove(-x)
    }

    fn inverse_privacy_loss_without_subsampling_remove(&self, loss: f64) -> f64 {
        let s = self.sensitivity;
        let add_inv = 0.5 * s - loss * self.variance / s;
        add_inv - s
    }

    fn inverse_privacy_loss_remove(&self, loss: f64) -> f64 {
        let q = self.sampling_prob;
        if (q - 1.0).abs() < 1e-12 {
            return self.inverse_privacy_loss_without_subsampling_remove(loss);
        }
        let log_1_minus_q = (1.0 - q).ln();
        if loss <= log_1_minus_q {
            return f64::NEG_INFINITY;
        }
        let loss_wo = (1.0 + (loss.exp() - 1.0) / q).ln();
        self.inverse_privacy_loss_without_subsampling_remove(loss_wo)
    }

    fn inverse_privacy_loss_add(&self, loss: f64) -> f64 {
        -self.inverse_privacy_loss_remove(-loss)
    }

    fn inverse_privacy_loss_replace(&self, loss: f64) -> f64 {
        let tail = self.privacy_loss_tail();
        let lower = tail.lower_x_truncation;
        let upper = tail.upper_x_truncation;
        binary_search_increasing(|x| self.privacy_loss(x), loss, lower, upper, 1e-6)
    }
}

impl MonotonePrivacyLoss for GaussianPrivacyLoss {
    fn privacy_loss(&self, x: f64) -> f64 {
        match self.adjacency_type {
            AdjacencyType::Add => self.privacy_loss_add(x),
            AdjacencyType::Remove => self.privacy_loss_remove(x),
            AdjacencyType::Replace => self.privacy_loss_remove(x) + self.privacy_loss_add(x),
        }
    }

    fn inverse_privacy_loss(&self, loss: f64) -> f64 {
        match self.adjacency_type {
            AdjacencyType::Add => self.inverse_privacy_loss_add(loss),
            AdjacencyType::Remove => self.inverse_privacy_loss_remove(loss),
            AdjacencyType::Replace => self.inverse_privacy_loss_replace(loss),
        }
    }

    fn mu_upper_cdf(&self, x: f64) -> f64 {
        match self.adjacency_type {
            AdjacencyType::Add => self.noise_cdf(x),
            AdjacencyType::Remove | AdjacencyType::Replace => {
                if (self.sampling_prob - 1.0).abs() < 1e-12 {
                    return self.noise_cdf(x + self.sensitivity);
                }
                (1.0 - self.sampling_prob) * self.noise_cdf(x)
                    + self.sampling_prob * self.noise_cdf(x + self.sensitivity)
            }
        }
    }

    fn privacy_loss_tail(&self) -> TailPrivacyLossDistribution {
        let tail_mass = 0.5 * self.log_mass_truncation_bound.exp();
        let z = self.normal.inverse_cdf(tail_mass);
        let upper_x_truncation = -z;

        let lower_x_truncation = match self.adjacency_type {
            AdjacencyType::Add => z,
            AdjacencyType::Remove | AdjacencyType::Replace => {
                let lo = z - self.sensitivity;
                let hi = z;
                binary_search_increasing(|x| self.mu_upper_cdf(x), tail_mass, lo, hi, 1e-6)
            }
        };

        let mut tail_mass_map = Vec::new();
        if self.pessimistic_estimate {
            tail_mass_map.push((f64::INFINITY, self.mu_upper_cdf(lower_x_truncation)));
            tail_mass_map.push((
                self.privacy_loss(upper_x_truncation),
                1.0 - self.mu_upper_cdf(upper_x_truncation),
            ));
        } else {
            tail_mass_map.push((
                self.privacy_loss(lower_x_truncation),
                self.mu_upper_cdf(lower_x_truncation),
            ));
        }

        TailPrivacyLossDistribution {
            lower_x_truncation,
            upper_x_truncation,
            tail_mass: tail_mass_map,
        }
    }
}

#[derive(Clone, Debug)]
struct PldPmf {
    discretization: f64,
    lower_loss: i64,
    probs: Vec<f64>,
    infinity_mass: f64,
    pessimistic_estimate: bool,
}

impl PldPmf {
    fn identity(discretization: f64, pessimistic_estimate: bool) -> Self {
        Self {
            discretization,
            lower_loss: 0,
            probs: vec![1.0],
            infinity_mass: 0.0,
            pessimistic_estimate,
        }
    }

    fn from_map(
        loss_probs: &HashMap<i64, f64>,
        discretization: f64,
        infinity_mass: f64,
        pessimistic_estimate: bool,
    ) -> Self {
        let min = *loss_probs.keys().min().unwrap_or(&0);
        let max = *loss_probs.keys().max().unwrap_or(&0);
        let size = (max - min + 1) as usize;
        let mut probs = vec![0.0_f64; size];
        for (k, v) in loss_probs {
            let idx = (k - min) as usize;
            probs[idx] += *v;
        }
        Self {
            discretization,
            lower_loss: min,
            probs,
            infinity_mass,
            pessimistic_estimate,
        }
    }

    fn size(&self) -> usize {
        self.probs.len()
    }

    fn loss_at(&self, idx: usize) -> f64 {
        (self.lower_loss + idx as i64) as f64 * self.discretization
    }

    fn validate_composable(&self, other: &Self) {
        assert!(
            (self.discretization - other.discretization).abs() < 1e-12,
            "discretization intervals must match"
        );
        assert!(
            self.pessimistic_estimate == other.pessimistic_estimate,
            "estimate types must match"
        );
    }

    fn truncate_tails(&self, probs: Vec<f64>, tail_mass_truncation: f64) -> (usize, Vec<f64>, f64) {
        if tail_mass_truncation == 0.0 {
            return (0, probs, 0.0);
        }
        let mut left_idx = 0usize;
        let mut left_mass = 0.0;
        while left_idx < probs.len() {
            left_mass += probs[left_idx];
            if left_mass > tail_mass_truncation / 2.0 {
                break;
            }
            left_idx += 1;
        }

        let mut right_idx = probs.len();
        let mut right_mass = 0.0;
        while right_idx > 0 {
            right_idx -= 1;
            right_mass += probs[right_idx];
            if right_mass > tail_mass_truncation / 2.0 {
                right_idx += 1;
                break;
            }
        }

        if right_idx <= left_idx {
            right_idx = left_idx + 1;
        }

        let mut truncated = probs[left_idx..right_idx].to_vec();
        if self.pessimistic_estimate {
            if let Some(first) = truncated.first_mut() {
                *first += left_mass;
            }
            (left_idx, truncated, right_mass)
        } else {
            if let Some(last) = truncated.last_mut() {
                *last += right_mass;
            }
            (left_idx, truncated, 0.0)
        }
    }

    fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = a.len() + b.len() - 1;
        let mut size = 1usize;
        while size < n {
            size <<= 1;
        }

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(size);
        let ifft = planner.plan_fft_inverse(size);

        let mut fa = vec![Complex::new(0.0, 0.0); size];
        let mut fb = vec![Complex::new(0.0, 0.0); size];
        for (i, &val) in a.iter().enumerate() {
            fa[i].re = val;
        }
        for (i, &val) in b.iter().enumerate() {
            fb[i].re = val;
        }
        fft.process(&mut fa);
        fft.process(&mut fb);
        for (a_i, b_i) in fa.iter_mut().zip(fb.iter()) {
            *a_i *= *b_i;
        }
        ifft.process(&mut fa);

        let scale = 1.0 / size as f64;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(fa[i].re * scale);
        }
        out
    }

    fn compose(&self, other: &Self, tail_mass_truncation: f64) -> Self {
        self.validate_composable(other);
        let mut probs = Self::convolve(&self.probs, &other.probs);
        let mut lower_loss = self.lower_loss + other.lower_loss;
        let infinity_mass =
            self.infinity_mass + other.infinity_mass - self.infinity_mass * other.infinity_mass;
        let (offset, truncated, right_tail) = self.truncate_tails(probs, tail_mass_truncation);
        lower_loss += offset as i64;
        Self {
            discretization: self.discretization,
            lower_loss,
            probs: truncated,
            infinity_mass: infinity_mass + right_tail,
            pessimistic_estimate: self.pessimistic_estimate,
        }
    }

    fn self_compose(&self, num_times: usize, tail_mass_truncation: f64) -> Self {
        assert!(num_times >= 1, "num_times must be >= 1");
        if num_times == 1 {
            return self.clone();
        }
        let mut result = PldPmf::identity(self.discretization, self.pessimistic_estimate);
        let mut base = self.clone();
        let mut n = num_times;
        while n > 0 {
            if n % 2 == 1 {
                result = result.compose(&base, tail_mass_truncation);
            }
            n /= 2;
            if n > 0 {
                base = base.compose(&base, tail_mass_truncation);
            }
        }
        result
    }

    fn get_delta_for_epsilon(&self, epsilon: f64) -> f64 {
        let mut delta = self.infinity_mass;
        for (i, prob) in self.probs.iter().enumerate() {
            let loss = self.loss_at(i);
            if loss > epsilon {
                delta += (-((epsilon - loss).exp_m1())) * prob;
            }
        }
        delta.clamp(0.0, 1.0)
    }

    fn get_epsilon_for_delta(&self, delta: f64) -> f64 {
        if self.infinity_mass > delta {
            return f64::INFINITY;
        }
        let mut mass_upper = self.infinity_mass;
        let mut mass_lower = 0.0;

        let mut losses: Vec<f64> = (0..self.size()).map(|i| self.loss_at(i)).collect();
        losses.reverse();
        let mut probs = self.probs.clone();
        probs.reverse();

        for (loss, prob) in losses.into_iter().zip(probs.into_iter()) {
            if mass_upper > delta && mass_lower > 0.0 {
                let eps = ((mass_upper - delta) / mass_lower).ln();
                if eps >= loss {
                    break;
                }
            }
            mass_upper += prob;
            mass_lower += prob * (-loss).exp();
            if mass_upper >= delta && mass_lower == 0.0 {
                return loss.max(0.0);
            }
        }

        if mass_upper <= mass_lower + delta {
            return 0.0;
        }
        ((mass_upper - delta) / mass_lower).ln()
    }

    fn get_delta_for_epsilon_for_composed_pld(&self, other: &Self, epsilon: f64) -> f64 {
        self.validate_composable(other);
        let mut delta =
            self.infinity_mass + other.infinity_mass - self.infinity_mass * other.infinity_mass;
        if self.loss_at(self.size() - 1) + other.loss_at(other.size() - 1) <= epsilon {
            return delta;
        }

        let mut upper_mass = 0.0;
        let mut lower_mass = 0.0;
        let mut j = other.size() as i64 - 1;
        let mut i = 0i64;

        while i < self.size() as i64
            && self.loss_at(i as usize) + other.loss_at(j as usize) < epsilon
        {
            i += 1;
        }

        while j >= 1 && self.loss_at(i as usize) + other.loss_at((j - 1) as usize) >= epsilon {
            let prob = other.probs[j as usize];
            let loss = other.loss_at(j as usize);
            upper_mass += prob;
            lower_mass += prob * (-loss).exp();
            j -= 1;
        }

        for i_idx in i..self.size() as i64 {
            if j >= 0 {
                let prob = other.probs[j as usize];
                let loss = other.loss_at(j as usize);
                upper_mass += prob;
                lower_mass += prob * (-loss).exp();
                j -= 1;
            }
            let self_prob = self.probs[i_idx as usize];
            let self_loss = self.loss_at(i_idx as usize);
            delta += self_prob * (upper_mass - (epsilon - self_loss).exp() * lower_mass);
        }

        delta
    }

    fn compute_mixture(&self, other: &Self, self_weight: f64) -> Self {
        self.validate_composable(other);
        let min_loss = self.lower_loss.min(other.lower_loss);
        let max_loss = (self.lower_loss + self.size() as i64 - 1)
            .max(other.lower_loss + other.size() as i64 - 1);
        let size = (max_loss - min_loss + 1) as usize;
        let mut probs = vec![0.0_f64; size];

        let start_self = (self.lower_loss - min_loss) as usize;
        for (i, prob) in self.probs.iter().enumerate() {
            probs[start_self + i] += self_weight * prob;
        }
        let start_other = (other.lower_loss - min_loss) as usize;
        for (i, prob) in other.probs.iter().enumerate() {
            probs[start_other + i] += (1.0 - self_weight) * prob;
        }

        let infinity_mass =
            self_weight * self.infinity_mass + (1.0 - self_weight) * other.infinity_mass;
        Self {
            discretization: self.discretization,
            lower_loss: min_loss,
            probs,
            infinity_mass,
            pessimistic_estimate: self.pessimistic_estimate,
        }
    }
}

#[derive(Clone, Debug)]
struct PrivacyLossDistribution {
    pmf_remove: PldPmf,
    pmf_add: Option<PldPmf>,
}

impl PrivacyLossDistribution {
    fn symmetric(pmf: PldPmf) -> Self {
        Self {
            pmf_remove: pmf,
            pmf_add: None,
        }
    }

    fn from_pmfs(pmf_remove: PldPmf, pmf_add: PldPmf) -> Self {
        Self {
            pmf_remove,
            pmf_add: Some(pmf_add),
        }
    }

    fn identity(value_discretization_interval: f64, pessimistic: bool) -> Self {
        Self::symmetric(PldPmf::identity(value_discretization_interval, pessimistic))
    }

    fn get_delta_for_epsilon(&self, epsilon: f64) -> f64 {
        let delta_remove = self.pmf_remove.get_delta_for_epsilon(epsilon);
        if let Some(add) = &self.pmf_add {
            let delta_add = add.get_delta_for_epsilon(epsilon);
            delta_remove.max(delta_add)
        } else {
            delta_remove
        }
    }

    fn get_epsilon_for_delta(&self, delta: f64) -> f64 {
        let eps_remove = self.pmf_remove.get_epsilon_for_delta(delta);
        if let Some(add) = &self.pmf_add {
            let eps_add = add.get_epsilon_for_delta(delta);
            eps_remove.max(eps_add)
        } else {
            eps_remove
        }
    }

    fn compose(&self, other: &Self, tail_mass_truncation: f64) -> Self {
        let pmf_remove = self
            .pmf_remove
            .compose(&other.pmf_remove, tail_mass_truncation);
        match (&self.pmf_add, &other.pmf_add) {
            (None, None) => Self::symmetric(pmf_remove),
            (Some(a1), Some(a2)) => {
                let pmf_add = a1.compose(a2, tail_mass_truncation);
                Self::from_pmfs(pmf_remove, pmf_add)
            }
            _ => Self::symmetric(pmf_remove),
        }
    }

    fn self_compose(&self, n: usize, tail_mass_truncation: f64) -> Self {
        let pmf_remove = self.pmf_remove.self_compose(n, tail_mass_truncation);
        if let Some(add) = &self.pmf_add {
            let pmf_add = add.self_compose(n, tail_mass_truncation);
            Self::from_pmfs(pmf_remove, pmf_add)
        } else {
            Self::symmetric(pmf_remove)
        }
    }

    fn compute_mixture(&self, other: &Self, weight: f64) -> Self {
        let pmf_remove = self.pmf_remove.compute_mixture(&other.pmf_remove, weight);
        match (&self.pmf_add, &other.pmf_add) {
            (None, None) => Self::symmetric(pmf_remove),
            (Some(a1), Some(a2)) => {
                let pmf_add = a1.compute_mixture(a2, weight);
                Self::from_pmfs(pmf_remove, pmf_add)
            }
            _ => Self::symmetric(pmf_remove),
        }
    }

    fn from_gaussian_mechanism(
        standard_deviation: f64,
        sensitivity: f64,
        pessimistic_estimate: bool,
        value_discretization_interval: f64,
        log_mass_truncation_bound: f64,
        sampling_prob: f64,
        neighboring_relation: NeighboringRelation,
    ) -> Self {
        let make = |adj: AdjacencyType| {
            let mpl = GaussianPrivacyLoss::new(
                standard_deviation,
                sensitivity,
                sampling_prob,
                adj,
                pessimistic_estimate,
                log_mass_truncation_bound,
            );
            create_pld_pmf_from_monotone_privacy_loss(
                &mpl,
                pessimistic_estimate,
                value_discretization_interval,
            )
        };

        match neighboring_relation {
            NeighboringRelation::AddOrRemoveOne | NeighboringRelation::ReplaceSpecial => {
                let pmf_remove = make(AdjacencyType::Remove);
                if (sampling_prob - 1.0).abs() < 1e-12 {
                    Self::symmetric(pmf_remove)
                } else {
                    let pmf_add = make(AdjacencyType::Add);
                    Self::from_pmfs(pmf_remove, pmf_add)
                }
            }
            NeighboringRelation::ReplaceOne => {
                let pmf_replace = make(AdjacencyType::Replace);
                Self::symmetric(pmf_replace)
            }
        }
    }

    fn from_truncated_subsampled_gaussian_mechanism(
        dataset_size: usize,
        sampling_probability: f64,
        truncated_batch_size: usize,
        noise_multiplier: f64,
        value_discretization_interval: f64,
        log_mass_truncation_bound: f64,
        neighboring_relation: NeighboringRelation,
    ) -> Self {
        let sens_1 = Self::from_gaussian_mechanism(
            noise_multiplier,
            1.0,
            true,
            value_discretization_interval,
            log_mass_truncation_bound,
            sampling_probability,
            neighboring_relation,
        );

        if sampling_probability <= 0.0 || truncated_batch_size == 0 || dataset_size == 0 {
            return sens_1;
        }

        let binom =
            Binomial::new(sampling_probability, (dataset_size - 1) as u64).expect("binomial");
        let trunc_prob = 1.0 - binom.cdf((truncated_batch_size - 1) as u64);
        if trunc_prob == 0.0 {
            return sens_1;
        }

        let binom_full = Binomial::new(sampling_probability, dataset_size as u64).expect("binom");
        let cond_prob = (1.0 - binom_full.cdf(truncated_batch_size as u64))
            * (truncated_batch_size as f64)
            / trunc_prob
            / (dataset_size as f64);

        let sens_2 = Self::from_gaussian_mechanism(
            noise_multiplier / 2.0,
            1.0,
            true,
            value_discretization_interval,
            log_mass_truncation_bound,
            cond_prob,
            NeighboringRelation::ReplaceOne,
        );

        sens_2.compute_mixture(&sens_1, trunc_prob)
    }
}

fn create_pld_pmf_from_monotone_privacy_loss<T: MonotonePrivacyLoss>(
    mpl: &T,
    pessimistic_estimate: bool,
    value_discretization_interval: f64,
) -> PldPmf {
    let tail = mpl.privacy_loss_tail();
    let mut infinity_mass = 0.0;
    let mut loss_probs: HashMap<i64, f64> = HashMap::new();

    for (loss, mass) in tail.tail_mass.iter().copied() {
        if mass <= 0.0 {
            continue;
        }
        if loss.is_infinite() {
            infinity_mass += mass;
            continue;
        }
        let idx = round_loss(loss, value_discretization_interval, pessimistic_estimate);
        *loss_probs.entry(idx).or_insert(0.0) += mass;
    }

    let lower_pl = mpl.privacy_loss(tail.upper_x_truncation);
    let upper_pl = mpl.privacy_loss(tail.lower_x_truncation);
    let k_min = if pessimistic_estimate {
        (lower_pl / value_discretization_interval).ceil() as i64
    } else {
        (lower_pl / value_discretization_interval).floor() as i64
    };
    let k_max = if pessimistic_estimate {
        (upper_pl / value_discretization_interval).ceil() as i64
    } else {
        (upper_pl / value_discretization_interval).floor() as i64
    };

    for k in k_min..=k_max {
        let (lower_loss, upper_loss) = if pessimistic_estimate {
            (
                (k as f64 - 1.0) * value_discretization_interval,
                k as f64 * value_discretization_interval,
            )
        } else {
            (
                k as f64 * value_discretization_interval,
                (k as f64 + 1.0) * value_discretization_interval,
            )
        };

        let x_low = mpl
            .inverse_privacy_loss(upper_loss)
            .clamp(tail.lower_x_truncation, tail.upper_x_truncation);
        let x_high = mpl
            .inverse_privacy_loss(lower_loss)
            .clamp(tail.lower_x_truncation, tail.upper_x_truncation);
        if x_high > x_low {
            let mass = mpl.mu_upper_cdf(x_high) - mpl.mu_upper_cdf(x_low);
            if mass > 0.0 {
                *loss_probs.entry(k).or_insert(0.0) += mass;
            }
        }
    }

    PldPmf::from_map(
        &loss_probs,
        value_discretization_interval,
        infinity_mass,
        pessimistic_estimate,
    )
}

fn round_loss(loss: f64, discretization: f64, pessimistic: bool) -> i64 {
    if pessimistic {
        (loss / discretization).ceil() as i64
    } else {
        (loss / discretization).floor() as i64
    }
}

/// A PLD-style accountant based on discretized privacy loss distributions.
#[derive(Clone, Debug)]
pub struct PldAccountant {
    neighboring_relation: NeighboringRelation,
    value_discretization_interval: f64,
    tail_mass_truncation: f64,
    log_mass_truncation_bound: f64,
    pld: PrivacyLossDistribution,
    contains_non_dp_event: bool,
}

impl Default for PldAccountant {
    fn default() -> Self {
        Self::new()
    }
}

impl PldAccountant {
    /// Create a new accountant with default settings.
    pub fn new() -> Self {
        let interval = DEFAULT_VALUE_DISCRETIZATION_INTERVAL;
        Self::with_params(NeighboringRelation::AddOrRemoveOne, interval)
    }

    /// Create a new accountant with explicit parameters.
    pub fn with_params(
        neighboring_relation: NeighboringRelation,
        value_discretization_interval: f64,
    ) -> Self {
        let pld = PrivacyLossDistribution::identity(value_discretization_interval, true);
        Self {
            neighboring_relation,
            value_discretization_interval,
            tail_mass_truncation: DEFAULT_TAIL_MASS_TRUNCATION,
            log_mass_truncation_bound: DEFAULT_LOG_MASS_TRUNCATION_BOUND,
            pld,
            contains_non_dp_event: false,
        }
    }

    /// Update the discretization interval used for new compositions.
    pub fn with_tail_mass_truncation(mut self, tail_mass_truncation: f64) -> Self {
        self.tail_mass_truncation = tail_mass_truncation;
        self
    }

    /// Record one DP-SGD step with Poisson sampling.
    pub fn step(&mut self, noise_mult: f64, q: f64) {
        if q <= 0.0 {
            return;
        }
        if noise_mult == 0.0 {
            self.contains_non_dp_event = true;
            return;
        }
        let pld = PrivacyLossDistribution::from_gaussian_mechanism(
            noise_mult,
            1.0,
            true,
            self.value_discretization_interval,
            self.log_mass_truncation_bound,
            q,
            self.neighboring_relation,
        );
        self.pld = self.pld.compose(&pld, self.tail_mass_truncation);
    }

    /// Record multiple identical steps.
    pub fn steps(&mut self, noise_mult: f64, q: f64, n: usize) {
        if n == 0 {
            return;
        }
        if noise_mult == 0.0 {
            self.contains_non_dp_event = true;
            return;
        }
        let pld = PrivacyLossDistribution::from_gaussian_mechanism(
            noise_mult,
            1.0,
            true,
            self.value_discretization_interval,
            self.log_mass_truncation_bound,
            q,
            self.neighboring_relation,
        );
        let composed = pld.self_compose(n, self.tail_mass_truncation);
        self.pld = self.pld.compose(&composed, self.tail_mass_truncation);
    }

    /// Record a truncated subsampled Gaussian step.
    pub fn truncated_step(
        &mut self,
        dataset_size: usize,
        sampling_probability: f64,
        truncated_batch_size: usize,
        noise_multiplier: f64,
    ) {
        if noise_multiplier == 0.0 {
            self.contains_non_dp_event = true;
            return;
        }
        let pld = PrivacyLossDistribution::from_truncated_subsampled_gaussian_mechanism(
            dataset_size,
            sampling_probability,
            truncated_batch_size,
            noise_multiplier,
            self.value_discretization_interval,
            self.log_mass_truncation_bound,
            self.neighboring_relation,
        );
        self.pld = self.pld.compose(&pld, self.tail_mass_truncation);
    }

    /// Convert to (epsilon, delta)-DP.
    pub fn epsilon(&self, delta: f64) -> f64 {
        if self.contains_non_dp_event {
            return f64::INFINITY;
        }
        self.pld.get_epsilon_for_delta(delta)
    }

    /// Reset the accountant state.
    pub fn reset(&mut self) {
        self.pld = PrivacyLossDistribution::identity(self.value_discretization_interval, true);
        self.contains_non_dp_event = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pld_epsilon_monotone_in_noise() {
        let mut acc_low = PldAccountant::new();
        let mut acc_high = PldAccountant::new();
        acc_low.step(0.8, 0.1);
        acc_high.step(2.0, 0.1);
        let eps_low = acc_low.epsilon(1e-6);
        let eps_high = acc_high.epsilon(1e-6);
        assert!(eps_high <= eps_low);
    }

    #[test]
    fn pld_steps_matches_repeated_steps() {
        let mut a = PldAccountant::new();
        let mut b = PldAccountant::new();
        a.steps(1.2, 0.2, 3);
        for _ in 0..3 {
            b.step(1.2, 0.2);
        }
        let eps_a = a.epsilon(1e-6);
        let eps_b = b.epsilon(1e-6);
        let rel = (eps_a - eps_b).abs() / eps_b.max(1e-9);
        assert!(rel < 0.05);
    }

    #[test]
    fn truncated_step_returns_finite() {
        let mut acc = PldAccountant::new();
        acc.truncated_step(1000, 0.1, 128, 1.5);
        let eps = acc.epsilon(1e-6);
        assert!(eps.is_finite());
    }
}
