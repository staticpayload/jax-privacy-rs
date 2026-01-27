//! Privacy analysis helpers and higher-level accountants.

use std::collections::{BTreeMap, BTreeSet};

use jax_privacy_core::{DpError, Result};

use crate::accountants::{DpAccountantConfig, PldAccountantConfig, RdpAccountantConfig};
use crate::{pld::PldAccountant, DpParams, RdpAccountant, Sampler, Schedule};

fn ceil_div(a: u64, b: u64) -> u64 {
    (a + b - 1) / b
}

fn normalize_schedule<T: Clone>(schedule: &Schedule<T>, max_step: u64) -> Vec<(u64, T)> {
    let mut points = schedule.normalize(max_step);
    if points.is_empty() {
        return points;
    }
    points.sort_by_key(|(t, _)| *t);
    points
}

fn interleave_schedules(noise: &[(u64, f64)], batches: &[(u64, usize)]) -> Vec<(u64, f64, usize)> {
    let mut steps: BTreeSet<u64> = noise.iter().map(|(t, _)| *t).collect();
    steps.extend(batches.iter().map(|(t, _)| *t));
    let steps: Vec<u64> = steps.into_iter().collect();

    let mut noise_idx = 0usize;
    let mut batch_idx = 0usize;
    let mut out = Vec::with_capacity(steps.len());

    for step in steps {
        while noise_idx + 1 < noise.len() && noise[noise_idx + 1].0 <= step {
            noise_idx += 1;
        }
        while batch_idx + 1 < batches.len() && batches[batch_idx + 1].0 <= step {
            batch_idx += 1;
        }
        out.push((step, noise[noise_idx].1, batches[batch_idx].1));
    }
    out
}

/// Interleave noise multipliers and batch sizes into durations.
pub fn interleave_nm_and_bs(
    noise_schedule: &Schedule<f64>,
    batch_schedule: &Schedule<usize>,
    num_steps: u64,
) -> Vec<(u64, f64, usize)> {
    if num_steps == 0 {
        return Vec::new();
    }
    let mut noise = normalize_schedule(noise_schedule, num_steps);
    let mut batches = normalize_schedule(batch_schedule, num_steps);
    if noise.is_empty() || batches.is_empty() {
        return Vec::new();
    }

    // Ensure the last change point does not exceed num_steps.
    noise.retain(|(t, _)| *t <= num_steps);
    batches.retain(|(t, _)| *t <= num_steps);

    let interleaved = interleave_schedules(&noise, &batches);
    if interleaved.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::with_capacity(interleaved.len());
    for i in 0..interleaved.len() - 1 {
        let (step, nm, bs) = interleaved[i];
        let next_step = interleaved[i + 1].0;
        if next_step > step {
            segments.push((next_step - step, nm, bs));
        }
    }
    let (last_step, last_nm, last_bs) = interleaved[interleaved.len() - 1];
    if num_steps >= last_step {
        segments.push((num_steps - last_step, last_nm, last_bs));
    }

    segments
}

/// Privacy accounting interface for training procedures.
pub trait DpTrainingAccountant {
    /// Whether the number of steps can be calibrated.
    fn can_calibrate_steps(&self) -> bool;
    /// Whether the batch size can be calibrated.
    fn can_calibrate_batch_size(&self) -> bool;
    /// Whether the noise multiplier can be calibrated.
    fn can_calibrate_noise_multipliers(&self) -> bool;

    /// Validate parameter compatibility.
    fn validate(&self, dp_params: &DpParams, num_updates: u64) -> Result<()>;

    /// Compute epsilon for the given updates and parameters.
    fn compute_epsilon_impl(&self, num_updates: u64, dp_params: &DpParams) -> Result<f64>;

    /// Compute epsilon with shared guard rails.
    fn compute_epsilon(&self, num_updates: u64, dp_params: &DpParams) -> Result<f64> {
        if num_updates == 0 {
            return Ok(0.0);
        }
        if !dp_params.is_finite_guarantee {
            return Ok(f64::INFINITY);
        }
        self.validate(dp_params, num_updates)?;
        self.compute_epsilon_impl(num_updates, dp_params)
    }
}

/// Accountant selection for training analysis.
#[derive(Clone, Debug)]
pub enum DpAccountantKind {
    /// Use an RDP accountant configuration.
    Rdp(RdpAccountantConfig),
    /// Use a PLD accountant configuration.
    Pld(PldAccountantConfig),
}

impl Default for DpAccountantKind {
    fn default() -> Self {
        Self::Rdp(RdpAccountantConfig::default())
    }
}

impl DpAccountantKind {
    fn is_pld(&self) -> bool {
        matches!(self, DpAccountantKind::Pld(_))
    }

    fn create(&self) -> TrainingAccountant {
        match self {
            DpAccountantKind::Rdp(cfg) => TrainingAccountant::Rdp(cfg.create_accountant()),
            DpAccountantKind::Pld(cfg) => TrainingAccountant::Pld(cfg.create_accountant()),
        }
    }
}

enum TrainingAccountant {
    Rdp(RdpAccountant),
    Pld(PldAccountant),
}

impl TrainingAccountant {
    fn steps(&mut self, noise_mult: f64, q: f64, n: usize) {
        match self {
            TrainingAccountant::Rdp(acc) => acc.steps(noise_mult, q, n),
            TrainingAccountant::Pld(acc) => acc.steps(noise_mult, q, n),
        }
    }

    fn epsilon(&self, delta: f64) -> f64 {
        match self {
            TrainingAccountant::Rdp(acc) => acc.epsilon(delta),
            TrainingAccountant::Pld(acc) => acc.epsilon(delta),
        }
    }
}

/// DP-SGD style accountant supporting cyclic Poisson sampling.
#[derive(Clone, Debug)]
pub struct DpsgdTrainingAccountant {
    dp_accountant_config: DpAccountantKind,
}

impl Default for DpsgdTrainingAccountant {
    fn default() -> Self {
        Self {
            dp_accountant_config: DpAccountantKind::default(),
        }
    }
}

impl DpsgdTrainingAccountant {
    /// Create an accountant with an explicit configuration.
    pub fn new(dp_accountant_config: DpAccountantKind) -> Self {
        Self {
            dp_accountant_config,
        }
    }

    /// Use an explicit RDP accountant configuration.
    pub fn with_rdp(config: RdpAccountantConfig) -> Self {
        Self::new(DpAccountantKind::Rdp(config))
    }

    /// Use an explicit PLD accountant configuration.
    pub fn with_pld(config: PldAccountantConfig) -> Self {
        Self::new(DpAccountantKind::Pld(config))
    }

    /// Access the configured accountant kind.
    pub fn accountant_kind(&self) -> &DpAccountantKind {
        &self.dp_accountant_config
    }
}

impl DpTrainingAccountant for DpsgdTrainingAccountant {
    fn can_calibrate_steps(&self) -> bool {
        true
    }

    fn can_calibrate_batch_size(&self) -> bool {
        true
    }

    fn can_calibrate_noise_multipliers(&self) -> bool {
        true
    }

    fn validate(&self, dp_params: &DpParams, num_updates: u64) -> Result<()> {
        dp_params.validate()?;
        if dp_params.examples_per_user.unwrap_or(1) != 1 {
            return Err(DpError::unsupported(
                "DpsgdTrainingAccountant requires examples_per_user = 1",
            ));
        }
        if dp_params.truncated_batch_size.is_some() && dp_params.sampler != Sampler::Poisson {
            return Err(DpError::unsupported(
                "truncated_batch_size is only supported with Poisson sampling",
            ));
        }
        if dp_params.truncated_batch_size.is_some() && !self.dp_accountant_config.is_pld() {
            return Err(DpError::invalid(
                "truncated_batch_size requires a PLD accountant configuration",
            ));
        }
        let cycle_length = dp_params.cycle_length.unwrap_or(1).max(1) as u64;
        if cycle_length > 1 {
            // Require constant noise and batch size when cycle_length > 1.
            if let Some(noise) = &dp_params.noise_multipliers {
                if noise.normalize(num_updates).len() != 1 {
                    return Err(DpError::invalid(
                        "cycle_length > 1 requires a single noise multiplier",
                    ));
                }
            }
            if dp_params.batch_sizes.normalize(num_updates).len() != 1 {
                return Err(DpError::invalid(
                    "cycle_length > 1 requires a single batch size",
                ));
            }
            if dp_params.batch_size as u64 * cycle_length > dp_params.num_samples as u64 {
                return Err(DpError::invalid(
                    "batch_size * cycle_length must be <= num_samples",
                ));
            }
        }
        Ok(())
    }

    fn compute_epsilon_impl(&self, num_updates: u64, dp_params: &DpParams) -> Result<f64> {
        let noise_schedule = dp_params
            .noise_multipliers
            .as_ref()
            .ok_or_else(|| DpError::invalid("noise multipliers are required"))?;
        let segments = interleave_nm_and_bs(noise_schedule, &dp_params.batch_sizes, num_updates);

        let cycle_length = dp_params.cycle_length.unwrap_or(1).max(1) as u64;
        let min_group_size = (dp_params.num_samples as u64 / cycle_length).max(1);
        let q_cap = dp_params
            .truncated_batch_size
            .map(|t| t as f64 / min_group_size as f64);
        let sensitivity_multiplier = match dp_params.sampler {
            Sampler::Poisson => 1.0,
            Sampler::Fixed => 2.0,
        };

        let mut accountant = self.dp_accountant_config.create();
        for (duration, nm, bs) in segments {
            if duration == 0 {
                continue;
            }
            let q_raw = bs as f64 / min_group_size as f64;
            let q = q_raw.min(q_cap.unwrap_or(1.0)).clamp(0.0, 1.0);
            let steps = ceil_div(duration, cycle_length) as usize;
            accountant.steps(nm / sensitivity_multiplier, q, steps);
        }

        Ok(accountant.epsilon(dp_params.delta))
    }
}

/// Conservative user-level accountant using worst-case sensitivity.
#[derive(Clone, Debug, Default)]
pub struct DpsgdTrainingUserLevelAccountant;

impl DpTrainingAccountant for DpsgdTrainingUserLevelAccountant {
    fn can_calibrate_steps(&self) -> bool {
        true
    }

    fn can_calibrate_batch_size(&self) -> bool {
        true
    }

    fn can_calibrate_noise_multipliers(&self) -> bool {
        true
    }

    fn validate(&self, dp_params: &DpParams, _num_updates: u64) -> Result<()> {
        dp_params.validate()?;
        if dp_params.examples_per_user.is_none() {
            return Err(DpError::invalid(
                "examples_per_user is required for user-level accounting",
            ));
        }
        if dp_params.cycle_length.unwrap_or(1) != 1 {
            return Err(DpError::unsupported(
                "cycle_length != 1 is not supported for user-level accounting",
            ));
        }
        if dp_params.truncated_batch_size.is_some() && dp_params.sampler != Sampler::Poisson {
            return Err(DpError::unsupported(
                "truncated_batch_size is only supported with Poisson sampling",
            ));
        }
        Ok(())
    }

    fn compute_epsilon_impl(&self, num_updates: u64, dp_params: &DpParams) -> Result<f64> {
        let noise_schedule = dp_params
            .noise_multipliers
            .as_ref()
            .ok_or_else(|| DpError::invalid("noise multipliers are required"))?;
        let segments = interleave_nm_and_bs(noise_schedule, &dp_params.batch_sizes, num_updates);

        let examples_per_user = dp_params.examples_per_user.unwrap_or(1) as f64;
        let sensitivity_multiplier = match dp_params.sampler {
            Sampler::Poisson => examples_per_user,
            Sampler::Fixed => 2.0 * examples_per_user,
        };
        let q_cap = dp_params
            .truncated_batch_size
            .map(|t| t as f64 / dp_params.num_samples as f64);

        let mut accountant = RdpAccountant::new();
        for (duration, nm, bs) in segments {
            if duration == 0 {
                continue;
            }
            let q_raw = bs as f64 / dp_params.num_samples as f64;
            let q = q_raw.min(q_cap.unwrap_or(1.0)).clamp(0.0, 1.0);
            accountant.steps(nm / sensitivity_multiplier, q, duration as usize);
        }
        Ok(accountant.epsilon(dp_params.delta))
    }
}

/// Single-release accountant (no amplification over steps).
#[derive(Clone, Debug, Default)]
pub struct SingleReleaseTrainingAccountant;

impl DpTrainingAccountant for SingleReleaseTrainingAccountant {
    fn can_calibrate_steps(&self) -> bool {
        false
    }

    fn can_calibrate_batch_size(&self) -> bool {
        false
    }

    fn can_calibrate_noise_multipliers(&self) -> bool {
        false
    }

    fn validate(&self, dp_params: &DpParams, _num_updates: u64) -> Result<()> {
        dp_params.validate()?;
        if dp_params.examples_per_user.unwrap_or(1) != 1 {
            return Err(DpError::unsupported(
                "SingleReleaseTrainingAccountant requires examples_per_user = 1",
            ));
        }
        Ok(())
    }

    fn compute_epsilon_impl(&self, num_updates: u64, dp_params: &DpParams) -> Result<f64> {
        let noise_schedule = dp_params
            .noise_multipliers
            .as_ref()
            .ok_or_else(|| DpError::invalid("noise multipliers are required"))?;
        let segments = interleave_nm_and_bs(noise_schedule, &dp_params.batch_sizes, num_updates);

        let mut accountant = RdpAccountant::new();
        for (_duration, nm, _bs) in segments {
            // Compose a single Gaussian event per configuration.
            accountant.step(nm, 1.0);
        }
        Ok(accountant.epsilon(dp_params.delta))
    }
}

/// Pre-compute and cache epsilon for different step counts.
#[derive(Clone, Debug)]
pub struct CachedExperimentAccountant<A: DpTrainingAccountant> {
    accountant: A,
    max_num_updates: u64,
    num_cached_points: usize,
    cache_initialized: bool,
    cached_points: Vec<u64>,
    cached_values: BTreeMap<u64, f64>,
}

impl<A: DpTrainingAccountant> CachedExperimentAccountant<A> {
    /// Create a new cached accountant.
    pub fn new(accountant: A, max_num_updates: u64, num_cached_points: usize) -> Self {
        Self {
            accountant,
            max_num_updates: max_num_updates.max(1),
            num_cached_points: num_cached_points.max(1),
            cache_initialized: false,
            cached_points: Vec::new(),
            cached_values: BTreeMap::new(),
        }
    }

    fn maybe_initialize_cache(&mut self, dp_params: &DpParams) -> Result<()> {
        if self.cache_initialized {
            return Ok(());
        }
        self.cached_points = (0..=self.num_cached_points)
            .map(|j| {
                ceil_div(
                    self.max_num_updates * j as u64,
                    self.num_cached_points as u64,
                )
            })
            .collect();
        self.cached_points.sort_unstable();
        self.cached_points.dedup();

        for &point in &self.cached_points {
            let eps = self.accountant.compute_epsilon(point, dp_params)?;
            self.cached_values.insert(point, eps);
        }
        self.cache_initialized = true;
        Ok(())
    }

    /// Compute epsilon, optionally using an over-approximating cache.
    pub fn compute_epsilon(
        &mut self,
        num_updates: u64,
        dp_params: &DpParams,
        allow_approximate_cache: bool,
    ) -> Result<f64> {
        if !allow_approximate_cache {
            return self.accountant.compute_epsilon(num_updates, dp_params);
        }
        self.maybe_initialize_cache(dp_params)?;
        let idx = ceil_div(
            self.num_cached_points as u64 * num_updates,
            self.max_num_updates,
        ) as usize;
        let closest = *self
            .cached_points
            .get(idx)
            .ok_or_else(|| DpError::invalid("cached point index out of bounds"))?;
        self.cached_values
            .get(&closest)
            .copied()
            .ok_or_else(|| DpError::invalid("cached epsilon missing"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accountants::PldAccountantConfig;

    #[test]
    fn test_interleave_nm_and_bs() {
        let noise = Schedule::timed(vec![(0, 1.0), (10, 2.0)]);
        let bs = Schedule::timed(vec![(0, 100usize), (5, 200usize)]);
        let segments = interleave_nm_and_bs(&noise, &bs, 20);
        assert!(!segments.is_empty());
        let total: u64 = segments.iter().map(|(t, _, _)| *t).sum();
        assert_eq!(total, 20);
    }

    #[test]
    fn test_dpsgd_accountant_runs() {
        let params = DpParams::new(1.0, 10_000, 100, 1e-5).expect("valid params");
        let acc = DpsgdTrainingAccountant::default();
        let eps = acc.compute_epsilon(100, &params).expect("epsilon");
        assert!(eps.is_finite());
        assert!(eps > 0.0);
    }

    #[test]
    fn truncated_batch_size_caps_sampling_rate() {
        let acc = DpsgdTrainingAccountant::with_pld(PldAccountantConfig::default());
        let base = DpParams::new(1.0, 10_000, 1_000, 1e-5).expect("valid params");
        let truncated = base.clone().with_truncated_batch_size(100);

        let eps_base = acc.compute_epsilon(200, &base).expect("base epsilon");
        let eps_trunc = acc
            .compute_epsilon(200, &truncated)
            .expect("truncated epsilon");

        assert!(eps_trunc.is_finite());
        assert!(eps_trunc <= eps_base + 1e-12);
    }

    #[test]
    fn truncated_batch_size_requires_poisson_sampling() {
        let acc = DpsgdTrainingAccountant::with_pld(PldAccountantConfig::default());
        let params = DpParams::new(1.0, 10_000, 500, 1e-5)
            .expect("valid params")
            .with_truncated_batch_size(100)
            .with_sampler(Sampler::Fixed);

        let res = acc.compute_epsilon(10, &params);
        assert!(res.is_err());
    }

    #[test]
    fn user_level_truncation_caps_sampling_rate() {
        let acc = DpsgdTrainingUserLevelAccountant;
        let base = DpParams::new(1.0, 10_000, 1_000, 1e-5)
            .expect("valid params")
            .with_examples_per_user(5);
        let truncated = base.clone().with_truncated_batch_size(100);

        let eps_base = acc.compute_epsilon(200, &base).expect("base epsilon");
        let eps_trunc = acc
            .compute_epsilon(200, &truncated)
            .expect("truncated epsilon");

        assert!(eps_trunc.is_finite());
        assert!(eps_trunc <= eps_base + 1e-12);
    }

    #[test]
    fn user_level_truncation_requires_poisson_sampling() {
        let acc = DpsgdTrainingUserLevelAccountant;
        let params = DpParams::new(1.0, 10_000, 1_000, 1e-5)
            .expect("valid params")
            .with_examples_per_user(5)
            .with_truncated_batch_size(100)
            .with_sampler(Sampler::Fixed);

        let res = acc.compute_epsilon(50, &params);
        assert!(res.is_err());
    }
}
