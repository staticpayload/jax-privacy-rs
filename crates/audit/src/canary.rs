//! Canary-score auditing utilities aligned with the Python reference.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use statrs::distribution::{Beta, ContinuousCDF};

use crate::stats::{epsilon_one_run, epsilon_one_run_fdp};

/// Parameters controlling bootstrap estimates.
#[derive(Clone, Debug)]
pub struct BootstrapParams {
    /// Number of bootstrap samples.
    pub num_samples: usize,
    /// Quantiles to report (in [0, 1]).
    pub quantiles: Vec<f64>,
    /// Optional RNG seed.
    pub seed: Option<u64>,
}

impl BootstrapParams {
    /// Create new bootstrap parameters with validation.
    pub fn new(num_samples: usize, quantiles: Vec<f64>, seed: Option<u64>) -> Self {
        let params = Self {
            num_samples,
            quantiles,
            seed,
        };
        params.validate();
        params
    }

    /// Validate the bootstrap parameters.
    pub fn validate(&self) {
        assert!(self.num_samples > 0, "num_samples must be positive");
        assert!(!self.quantiles.is_empty(), "quantiles cannot be empty");
        for &q in &self.quantiles {
            assert!((0.0..1.0).contains(&q), "quantiles must be in (0, 1)");
        }
    }

    /// Construct parameters corresponding to a confidence interval.
    pub fn confidence_interval(num_samples: usize, confidence: f64, seed: Option<u64>) -> Self {
        assert!(
            (0.0..1.0).contains(&confidence),
            "confidence must be in (0, 1)"
        );
        let significance = 1.0 - confidence;
        let quantiles = vec![significance / 2.0, 1.0 - significance / 2.0];
        Self::new(num_samples, quantiles, seed)
    }
}

/// Threshold selection strategy for canary audits.
#[derive(Clone, Debug)]
pub enum ThresholdStrategy {
    /// Use Bonferroni correction across all thresholds.
    Bonferroni,
    /// Use a specific threshold.
    Explicit(f64),
    /// Split data to choose a threshold and audit on the rest.
    Split {
        /// Fraction of scores used to estimate the threshold.
        threshold_estimation_frac: f64,
        /// Optional RNG seed for deterministic splits.
        seed: Option<u64>,
    },
    /// Repeat split auditing and report the median bound.
    MultiSplit {
        /// Number of split samples to evaluate.
        num_samples: usize,
        /// Fraction of scores used to estimate the threshold.
        threshold_estimation_frac: f64,
        /// Optional RNG seed for deterministic splits.
        seed: Option<u64>,
    },
}

impl Default for ThresholdStrategy {
    fn default() -> Self {
        Self::Bonferroni
    }
}

/// Result of a canary-score audit.
#[derive(Clone, Debug)]
pub struct CanaryAuditResult {
    /// Lower bound on epsilon falsified by the audit.
    pub epsilon_lower: f64,
    /// Threshold used for the audit.
    pub threshold: f64,
}

/// Auditor for held-in and held-out canary scores.
#[derive(Clone, Debug)]
pub struct CanaryScoreAuditor {
    in_scores: Vec<f64>,
    out_scores: Vec<f64>,
    thresholds: Vec<f64>,
    tn_counts: Vec<usize>,
    fn_counts: Vec<usize>,
    in_sorted: Vec<f64>,
    out_sorted: Vec<f64>,
}

impl CanaryScoreAuditor {
    /// Initialize a canary-score auditor.
    pub fn new(in_scores: Vec<f64>, out_scores: Vec<f64>) -> Self {
        assert!(!in_scores.is_empty(), "in_scores must be non-empty");
        assert!(!out_scores.is_empty(), "out_scores must be non-empty");

        let (thresholds, tn_counts, fn_counts, in_sorted, out_sorted) =
            get_tn_fn_counts(&in_scores, &out_scores);

        Self {
            in_scores,
            out_scores,
            thresholds,
            tn_counts,
            fn_counts,
            in_sorted,
            out_sorted,
        }
    }

    fn tp_counts(&self) -> Vec<usize> {
        let total = *self.fn_counts.last().unwrap_or(&0);
        self.fn_counts.iter().rev().map(|v| total - v).collect()
    }

    fn fp_counts(&self) -> Vec<usize> {
        let total = *self.tn_counts.last().unwrap_or(&0);
        self.tn_counts.iter().rev().map(|v| total - v).collect()
    }

    /// Maximum TPR achievable at a given FPR.
    pub fn tpr_at_given_fpr(&self, fpr: f64, bootstrap: Option<BootstrapParams>) -> Vec<f64> {
        assert!((0.0..=1.0).contains(&fpr), "fpr must be in [0, 1]");
        if let Some(params) = bootstrap {
            return bootstrap_quantiles(&params, &self.in_scores, &self.out_scores, |auditor| {
                auditor.tpr_at_given_fpr(fpr, None)[0]
            });
        }
        let tp = self.tp_counts();
        let fp = self.fp_counts();
        vec![tpr_at_given_fpr(fpr, &tp, &fp)]
    }

    /// Maximum TPR achievable at multiple FPR values.
    pub fn tpr_at_given_fpr_many(
        &self,
        fprs: &[f64],
        bootstrap: Option<BootstrapParams>,
    ) -> Vec<f64> {
        if let Some(params) = bootstrap {
            assert!(
                fprs.len() == 1,
                "bootstrap only supports a single FPR value"
            );
            return self.tpr_at_given_fpr(fprs[0], Some(params));
        }
        fprs.iter()
            .map(|&fpr| self.tpr_at_given_fpr(fpr, None)[0])
            .collect()
    }

    /// AUROC of the attack scores.
    pub fn attack_auroc(&self, bootstrap: Option<BootstrapParams>) -> Vec<f64> {
        if let Some(params) = bootstrap {
            return bootstrap_quantiles(&params, &self.in_scores, &self.out_scores, |auditor| {
                auditor.attack_auroc(None)[0]
            });
        }
        let n_pos = *self.fn_counts.last().unwrap_or(&0) as f64;
        let n_neg = *self.tn_counts.last().unwrap_or(&0) as f64;
        if n_pos == 0.0 || n_neg == 0.0 {
            return vec![0.5];
        }
        let tnr: Vec<f64> = self.tn_counts.iter().map(|v| *v as f64 / n_neg).collect();
        let fnr: Vec<f64> = self.fn_counts.iter().map(|v| *v as f64 / n_pos).collect();
        let mut area = 0.0;
        for i in 1..tnr.len() {
            area += 0.5 * (tnr[i - 1] + tnr[i]) * (fnr[i] - fnr[i - 1]);
        }
        vec![area]
    }

    /// Estimate epsilon from raw counts.
    pub fn epsilon_raw_counts(
        &self,
        min_count: usize,
        delta: f64,
        one_sided: bool,
        bootstrap: Option<BootstrapParams>,
    ) -> Vec<f64> {
        assert!(min_count > 0, "min_count must be positive");
        assert!((0.0..=1.0).contains(&delta), "delta must be in [0, 1]");
        if let Some(params) = bootstrap {
            return bootstrap_quantiles(&params, &self.in_scores, &self.out_scores, |auditor| {
                auditor.epsilon_raw_counts(min_count, delta, one_sided, None)[0]
            });
        }
        let tp = self.tp_counts();
        let fp = self.fp_counts();
        let eps = epsilon_raw_counts_helper(&tp, &fp, min_count, delta);
        if one_sided {
            vec![eps.max(0.0)]
        } else {
            let eps_rev =
                epsilon_raw_counts_helper(&self.tn_counts, &self.fn_counts, min_count, delta);
            vec![eps.max(eps_rev).max(0.0)]
        }
    }

    /// Epsilon bound from Clopper-Pearson confidence intervals.
    pub fn epsilon_clopper_pearson(
        &self,
        significance: f64,
        delta: f64,
        one_sided: bool,
        threshold_strategy: ThresholdStrategy,
    ) -> f64 {
        assert!(
            (0.0..0.5).contains(&significance),
            "significance must be in (0, 0.5)"
        );
        assert!((0.0..=1.0).contains(&delta), "delta must be in [0, 1]");
        let audit_fn = CanaryScoreAuditor::epsilon_clopper_pearson_all_thresholds;
        self.audit_with_threshold_strategy(
            threshold_strategy,
            audit_fn,
            significance,
            delta,
            one_sided,
        )
    }

    /// Maximum accuracy achievable by a threshold classifier.
    pub fn max_accuracy(&self, prevalence: Option<f64>, significance: Option<f64>) -> f64 {
        let n_pos = *self.fn_counts.last().unwrap_or(&0) as f64;
        let n_neg = *self.tn_counts.last().unwrap_or(&0) as f64;
        if n_pos == 0.0 || n_neg == 0.0 {
            return 0.0;
        }
        let prevalence = prevalence
            .unwrap_or(n_pos / (n_pos + n_neg))
            .clamp(0.0, 1.0);

        let tp_counts: Vec<usize> = self
            .fn_counts
            .iter()
            .map(|v| (n_pos as usize) - v)
            .collect();
        let (tnr_ubs, tpr_ubs) = if let Some(alpha) = significance {
            (
                clopper_pearson_upper_vec(&self.tn_counts, n_neg as usize, alpha / 2.0),
                clopper_pearson_upper_vec(&tp_counts, n_pos as usize, alpha / 2.0),
            )
        } else {
            (
                self.tn_counts.iter().map(|v| *v as f64 / n_neg).collect(),
                tp_counts.iter().map(|v| *v as f64 / n_pos).collect(),
            )
        };

        let mut best = 0.0;
        for (&tnr, &tpr) in tnr_ubs.iter().zip(tpr_ubs.iter()) {
            let acc = tpr * prevalence + tnr * (1.0 - prevalence);
            if acc > best {
                best = acc;
            }
        }
        best
    }

    /// GDP-based epsilon estimate from canary scores.
    pub fn epsilon_from_gdp(&self, significance: f64, delta: f64, eps_tol: f64) -> f64 {
        assert!(
            (0.0..0.5).contains(&significance),
            "significance must be in (0, 0.5)"
        );
        assert!(
            (0.0..=1.0).contains(&delta) && delta > 0.0,
            "delta must be in (0, 1]"
        );
        assert!(eps_tol > 0.0, "eps_tol must be positive");

        let n_pos = *self.fn_counts.last().unwrap_or(&0);
        let n_neg = *self.tn_counts.last().unwrap_or(&0);
        if n_pos == 0 || n_neg == 0 {
            return 0.0;
        }

        let n = self.fn_counts.len().max(1) as f64;
        let fnr_ubs = clopper_pearson_upper_vec(&self.fn_counts, n_pos, significance / (2.0 * n));
        let fp_counts: Vec<usize> = self.tn_counts.iter().map(|v| n_neg - v).collect();
        let fpr_ubs = clopper_pearson_upper_vec(&fp_counts, n_neg, significance / (2.0 * n));

        let mut max_mu: f64 = 0.0;
        for (&fpr, &fnr) in fpr_ubs.iter().zip(fnr_ubs.iter()) {
            if fpr >= 1.0 - delta || fnr >= 1.0 - delta {
                continue;
            }
            let mu = normal_isf(fpr) - normal_ppf(fnr);
            max_mu = max_mu.max(mu.abs());
        }
        if max_mu == 0.0 {
            return 0.0;
        }

        let delta_gap = |eps: f64| {
            log_sub(
                normal_logcdf(-(eps / max_mu) + max_mu / 2.0),
                eps + normal_logcdf(-(eps / max_mu) - max_mu / 2.0),
            ) - delta.ln()
        };

        let mut lo = 0.0;
        let mut hi = 100.0;
        if delta_gap(lo) <= 0.0 {
            return lo;
        }
        if delta_gap(hi) >= 0.0 {
            return hi;
        }

        for _ in 0..80 {
            if (hi - lo).abs() <= eps_tol {
                break;
            }
            let mid = 0.5 * (lo + hi);
            if delta_gap(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// One-shot audit epsilon bound using the standard DP test.
    pub fn epsilon_one_run(
        &self,
        significance: f64,
        delta: f64,
        one_sided: bool,
        use_fdp: bool,
        threshold_strategy: ThresholdStrategy,
    ) -> f64 {
        assert!(
            (0.0..1.0).contains(&significance),
            "significance must be in (0, 1)"
        );
        assert!((0.0..=1.0).contains(&delta), "delta must be in [0, 1]");
        assert!(one_sided, "one_sided must be true");
        let audit_fn = if use_fdp {
            CanaryScoreAuditor::epsilon_one_run_fdp_all_thresholds
        } else {
            CanaryScoreAuditor::epsilon_one_run_all_thresholds
        };
        self.audit_with_threshold_strategy(
            threshold_strategy,
            audit_fn,
            significance,
            delta,
            one_sided,
        )
    }

    fn audit_with_threshold_strategy(
        &self,
        strategy: ThresholdStrategy,
        audit_fn: fn(&CanaryScoreAuditor, f64, f64, bool, Option<f64>) -> (f64, f64),
        significance: f64,
        delta: f64,
        one_sided: bool,
    ) -> f64 {
        match strategy {
            ThresholdStrategy::Bonferroni => {
                let denom = self.thresholds.len().max(1) as f64;
                let alpha = significance / denom;
                audit_fn(self, alpha, delta, one_sided, None).0
            }
            ThresholdStrategy::Explicit(t) => {
                audit_fn(self, significance, delta, one_sided, Some(t)).0
            }
            ThresholdStrategy::Split {
                threshold_estimation_frac,
                seed,
            } => {
                let (est_in, aud_in) =
                    random_partition(&self.in_scores, threshold_estimation_frac, seed);
                let (est_out, aud_out) = random_partition(
                    &self.out_scores,
                    threshold_estimation_frac,
                    seed.map(|s| s.wrapping_add(1)),
                );
                let auditor_est = CanaryScoreAuditor::new(est_in, est_out);
                let (_, threshold) = audit_fn(&auditor_est, significance, delta, one_sided, None);
                let auditor_aud = CanaryScoreAuditor::new(aud_in, aud_out);
                audit_fn(
                    &auditor_aud,
                    significance,
                    delta,
                    one_sided,
                    Some(threshold),
                )
                .0
            }
            ThresholdStrategy::MultiSplit {
                num_samples,
                threshold_estimation_frac,
                seed,
            } => {
                let samples = num_samples.max(1);
                let mut eps_vals = Vec::with_capacity(samples);
                for i in 0..samples {
                    let seed_i = seed.map(|s| s.wrapping_add(i as u64));
                    let (est_in, aud_in) =
                        random_partition(&self.in_scores, threshold_estimation_frac, seed_i);
                    let (est_out, aud_out) = random_partition(
                        &self.out_scores,
                        threshold_estimation_frac,
                        seed_i.map(|s| s.wrapping_add(1)),
                    );
                    let auditor_est = CanaryScoreAuditor::new(est_in, est_out);
                    let (_, threshold) =
                        audit_fn(&auditor_est, significance / 2.0, delta, one_sided, None);
                    let auditor_aud = CanaryScoreAuditor::new(aud_in, aud_out);
                    let eps = audit_fn(
                        &auditor_aud,
                        significance / 2.0,
                        delta,
                        one_sided,
                        Some(threshold),
                    )
                    .0;
                    eps_vals.push(eps);
                }
                median(&mut eps_vals)
            }
        }
    }

    fn epsilon_clopper_pearson_all_thresholds(
        &self,
        significance: f64,
        delta: f64,
        one_sided: bool,
        threshold: Option<f64>,
    ) -> (f64, f64) {
        let (fn_counts, tn_counts, thresholds) = match threshold {
            Some(t) => {
                let fn_count = lower_bound(&self.in_sorted, t);
                let tn_count = lower_bound(&self.out_sorted, t);
                (vec![fn_count], vec![tn_count], vec![t])
            }
            None => (
                self.fn_counts.clone(),
                self.tn_counts.clone(),
                self.thresholds.clone(),
            ),
        };

        let n_pos = *fn_counts.last().unwrap_or(&0);
        let n_neg = *tn_counts.last().unwrap_or(&0);
        if n_pos == 0 || n_neg == 0 {
            return (0.0, thresholds.get(0).copied().unwrap_or(0.0));
        }

        let fnr_ubs = clopper_pearson_upper_vec(&fn_counts, n_pos, significance / 2.0);
        let fp_counts: Vec<usize> = tn_counts.iter().map(|v| n_neg - v).collect();
        let fpr_ubs = clopper_pearson_upper_vec(&fp_counts, n_neg, significance / 2.0);

        let (mut eps, mut idx) = eps_from_bounds(&fnr_ubs, &fpr_ubs, delta);
        if !one_sided {
            let (eps_rev, idx_rev) = eps_from_bounds(&fpr_ubs, &fnr_ubs, delta);
            if eps_rev > eps {
                eps = eps_rev;
                idx = idx_rev;
            }
        }

        let threshold = thresholds.get(idx).copied().unwrap_or(0.0);
        (eps, threshold)
    }

    fn epsilon_one_run_all_thresholds(
        &self,
        significance: f64,
        delta: f64,
        one_sided: bool,
        threshold: Option<f64>,
    ) -> (f64, f64) {
        assert!(one_sided, "one_sided must be true");
        self.epsilon_one_run_impl(significance, delta, one_sided, threshold, false)
    }

    fn epsilon_one_run_fdp_all_thresholds(
        &self,
        significance: f64,
        delta: f64,
        one_sided: bool,
        threshold: Option<f64>,
    ) -> (f64, f64) {
        assert!(one_sided, "one_sided must be true");
        self.epsilon_one_run_impl(significance, delta, one_sided, threshold, true)
    }

    fn epsilon_one_run_impl(
        &self,
        significance: f64,
        delta: f64,
        one_sided: bool,
        threshold: Option<f64>,
        use_fdp: bool,
    ) -> (f64, f64) {
        let thresholds = if let Some(t) = threshold {
            vec![t]
        } else {
            self.thresholds.clone()
        };

        let mut best_eps = 0.0;
        let mut best_threshold = thresholds.get(0).copied().unwrap_or(0.0);
        let m = self.in_scores.len() + self.out_scores.len();

        for &t in &thresholds {
            let (tp, _fp, n_guess, n_correct) =
                counts_at_threshold(&self.in_sorted, &self.out_sorted, t);
            let eps = if use_fdp {
                epsilon_one_run_fdp(0.0, m, n_guess, n_correct, significance, delta)
            } else {
                epsilon_one_run(0.0, m, n_guess, n_correct, significance, delta)
            };

            let mut eps_best = eps;
            if !one_sided {
                let (tp_rev, fp_rev, n_guess_rev, n_correct_rev) =
                    counts_at_threshold_reverse(&self.in_sorted, &self.out_sorted, t);
                let eps_rev = if use_fdp {
                    epsilon_one_run_fdp(0.0, m, n_guess_rev, n_correct_rev, significance, delta)
                } else {
                    epsilon_one_run(0.0, m, n_guess_rev, n_correct_rev, significance, delta)
                };
                if eps_rev > eps_best {
                    eps_best = eps_rev;
                }
                let _ = (tp_rev, fp_rev, tp);
            }

            if eps_best > best_eps {
                best_eps = eps_best;
                best_threshold = t;
            }
        }

        (best_eps, best_threshold)
    }
}

fn get_tn_fn_counts(
    in_scores: &[f64],
    out_scores: &[f64],
) -> (Vec<f64>, Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>) {
    let mut thresholds: Vec<f64> = in_scores
        .iter()
        .chain(out_scores.iter())
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    thresholds.push(f64::INFINITY);

    let mut in_sorted = in_scores.to_vec();
    let mut out_sorted = out_scores.to_vec();
    in_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut fn_counts = Vec::with_capacity(thresholds.len());
    let mut tn_counts = Vec::with_capacity(thresholds.len());
    for &t in &thresholds {
        fn_counts.push(lower_bound(&in_sorted, t));
        tn_counts.push(lower_bound(&out_sorted, t));
    }

    let counts: Vec<(usize, usize)> = fn_counts
        .iter()
        .zip(tn_counts.iter())
        .map(|(&f, &t)| (f, t))
        .collect();
    let pareto = pareto_frontier(&counts);
    let thresholds = pareto.iter().map(|&i| thresholds[i]).collect::<Vec<_>>();
    let tn_counts = pareto.iter().map(|&i| tn_counts[i]).collect::<Vec<_>>();
    let fn_counts = pareto.iter().map(|&i| fn_counts[i]).collect::<Vec<_>>();

    (thresholds, tn_counts, fn_counts, in_sorted, out_sorted)
}

fn lower_bound(sorted: &[f64], value: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = sorted.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if sorted[mid] < value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn pareto_frontier(points: &[(usize, usize)]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..points.len()).collect();
    loop {
        if indices.len() <= 2 {
            break;
        }
        let mut dominated = vec![false; indices.len() - 2];
        for i in 0..indices.len() - 2 {
            let (x0, y0) = points[indices[i]];
            let (x1, y1) = points[indices[i + 1]];
            let (x2, y2) = points[indices[i + 2]];

            let dx1 = x1 as f64 - x0 as f64;
            let dy1 = y1 as f64 - y0 as f64;
            let dx2 = x2 as f64 - x1 as f64;
            let dy2 = y2 as f64 - y1 as f64;

            let cross = dy1 * dx2 - dy2 * dx1;
            dominated[i] = cross <= 0.0;
        }
        if !dominated.iter().any(|&v| v) {
            break;
        }
        let mut new_indices = Vec::with_capacity(indices.len());
        for (idx, &val) in indices.iter().enumerate() {
            if idx == 0 || idx == indices.len() - 1 {
                new_indices.push(val);
                continue;
            }
            if !dominated[idx - 1] {
                new_indices.push(val);
            }
        }
        indices = new_indices;
    }
    indices
}

fn tpr_at_given_fpr(fpr: f64, tp_counts: &[usize], fp_counts: &[usize]) -> f64 {
    let fpr = fpr.clamp(0.0, 1.0);
    if tp_counts.is_empty() || fp_counts.is_empty() {
        return 0.0;
    }
    let n_pos = *tp_counts.last().unwrap_or(&0) as f64;
    let n_neg = *fp_counts.last().unwrap_or(&0) as f64;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.0;
    }
    let target = n_neg * fpr;
    let mut idx = upper_bound(fp_counts, target);
    if idx >= fp_counts.len() {
        idx = fp_counts.len() - 1;
    }
    if idx == 0 {
        return 0.0;
    }
    let fp_left = fp_counts[idx - 1] as f64;
    let fp_right = fp_counts[idx] as f64;
    let denom = (fp_right - fp_left).max(1.0);
    let q = (target - fp_left) / denom;
    let tp_left = tp_counts[idx - 1] as f64;
    let tp_right = tp_counts[idx] as f64;
    ((tp_left + q * (tp_right - tp_left)) / n_pos).clamp(0.0, 1.0)
}

fn upper_bound(sorted: &[usize], value: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = sorted.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if (sorted[mid] as f64) <= value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn epsilon_raw_counts_helper(
    tp_counts: &[usize],
    fp_counts: &[usize],
    min_count: usize,
    delta: f64,
) -> f64 {
    let n_pos = *tp_counts.last().unwrap_or(&0) as f64;
    let n_neg = *fp_counts.last().unwrap_or(&0) as f64;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.0;
    }
    if min_count as f64 >= n_neg {
        return 0.0;
    }
    let min_fpr = min_count as f64 / n_neg;
    let tpr_at_min = tpr_at_given_fpr(min_fpr, tp_counts, fp_counts);
    if delta == 0.0 {
        return (tpr_at_min / min_fpr).ln();
    }
    let initial_eps = if tpr_at_min > delta {
        ((tpr_at_min - delta).ln() - min_fpr.ln()).max(0.0)
    } else {
        0.0
    };
    let mut best = initial_eps;
    for (&tp, &fp) in tp_counts.iter().zip(fp_counts.iter()) {
        if fp < min_count {
            continue;
        }
        let tpr = tp as f64 / n_pos;
        let fpr = fp as f64 / n_neg;
        if tpr <= delta || fpr <= 0.0 {
            continue;
        }
        let eps = (tpr - delta).ln() - fpr.ln();
        if eps > best {
            best = eps;
        }
    }
    best
}

fn clopper_pearson_upper_vec(counts: &[usize], n: usize, significance: f64) -> Vec<f64> {
    counts
        .iter()
        .map(|&k| {
            if k >= n {
                1.0
            } else {
                let beta = Beta::new((k + 1) as f64, (n - k) as f64)
                    .unwrap_or_else(|e| panic!("beta error: {e}"));
                beta.inverse_cdf((1.0 - significance).clamp(0.0, 1.0))
                    .min(1.0)
                    .max(0.0)
            }
        })
        .collect()
}

fn eps_from_bounds(fnr_ubs: &[f64], fpr_ubs: &[f64], delta: f64) -> (f64, usize) {
    let mut best = 0.0;
    let mut idx = 0usize;
    for (i, (&fnr, &fpr)) in fnr_ubs.iter().zip(fpr_ubs.iter()).enumerate() {
        let tpr = 1.0 - fnr;
        if tpr <= delta || fpr <= 0.0 {
            continue;
        }
        let eps = (tpr - delta).ln() - fpr.ln();
        if eps > best {
            best = eps;
            idx = i;
        }
    }
    (best.max(0.0), idx)
}

fn counts_at_threshold(
    in_sorted: &[f64],
    out_sorted: &[f64],
    threshold: f64,
) -> (usize, usize, usize, usize) {
    let fn_count = lower_bound(in_sorted, threshold);
    let tn_count = lower_bound(out_sorted, threshold);
    let tp = in_sorted.len() - fn_count;
    let fp = out_sorted.len() - tn_count;
    let n_guess = tp + fp;
    let n_correct = tp;
    (tp, fp, n_guess, n_correct)
}

fn counts_at_threshold_reverse(
    in_sorted: &[f64],
    out_sorted: &[f64],
    threshold: f64,
) -> (usize, usize, usize, usize) {
    let fn_count = lower_bound(in_sorted, threshold);
    let tn_count = lower_bound(out_sorted, threshold);
    let tp = fn_count;
    let fp = tn_count;
    let n_guess = tp + fp;
    let n_correct = tp;
    (tp, fp, n_guess, n_correct)
}

fn random_partition(scores: &[f64], p: f64, seed: Option<u64>) -> (Vec<f64>, Vec<f64>) {
    assert!(p > 0.0 && p < 1.0, "p must be in (0, 1)");
    let mut rng = seeded_rng(seed);
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.shuffle(&mut rng);
    let split_idx = ((scores.len() as f64) * p).floor() as usize;
    let mut left = Vec::new();
    let mut right = Vec::new();
    for (idx, &i) in indices.iter().enumerate() {
        if idx < split_idx {
            left.push(scores[i]);
        } else {
            right.push(scores[i]);
        }
    }
    (left, right)
}

fn seeded_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = <StdRng as SeedableRng>::Seed::default();
            rand::thread_rng().fill_bytes(&mut seed_bytes);
            StdRng::from_seed(seed_bytes)
        }
    }
}

fn bootstrap_quantiles(
    params: &BootstrapParams,
    in_scores: &[f64],
    out_scores: &[f64],
    f: impl Fn(&CanaryScoreAuditor) -> f64,
) -> Vec<f64> {
    params.validate();
    let mut rng = seeded_rng(params.seed);
    let mut values = Vec::with_capacity(params.num_samples.max(1));
    for _ in 0..params.num_samples.max(1) {
        let in_resample = resample_scores(&mut rng, in_scores);
        let out_resample = resample_scores(&mut rng, out_scores);
        let auditor = CanaryScoreAuditor::new(in_resample, out_resample);
        values.push(f(&auditor));
    }
    quantiles(&mut values, &params.quantiles)
}

fn resample_scores(rng: &mut impl Rng, scores: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(scores.len());
    if scores.is_empty() {
        return out;
    }
    for _ in 0..scores.len() {
        let idx = rng.gen_range(0..scores.len());
        out.push(scores[idx]);
    }
    out
}

fn quantiles(values: &mut [f64], qs: &[f64]) -> Vec<f64> {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if values.is_empty() {
        return vec![0.0; qs.len()];
    }
    qs.iter()
        .map(|&q| {
            let q = q.clamp(0.0, 1.0);
            let idx = q * (values.len() - 1) as f64;
            let lo = idx.floor() as usize;
            let hi = idx.ceil() as usize;
            if lo == hi {
                values[lo]
            } else {
                let w = idx - lo as f64;
                values[lo] * (1.0 - w) + values[hi] * w
            }
        })
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        0.5 * (values[mid - 1] + values[mid])
    }
}

fn normal_ppf(p: f64) -> f64 {
    let normal = statrs::distribution::Normal::new(0.0, 1.0).expect("normal");
    normal.inverse_cdf(p.clamp(1e-12, 1.0 - 1e-12))
}

fn normal_isf(p: f64) -> f64 {
    normal_ppf(1.0 - p)
}

fn normal_logcdf(x: f64) -> f64 {
    let normal = statrs::distribution::Normal::new(0.0, 1.0).expect("normal");
    let cdf = normal.cdf(x);
    if cdf <= 0.0 {
        f64::NEG_INFINITY
    } else {
        cdf.ln()
    }
}

fn log_sub(x: f64, y: f64) -> f64 {
    if y > x {
        return f64::NEG_INFINITY;
    }
    if x == y {
        return f64::NEG_INFINITY;
    }
    x + (1.0 - (y - x).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonferroni_runs() {
        let in_scores = vec![1.0, 2.0, 3.0, 4.0];
        let out_scores = vec![0.1, 0.2, 0.3, 0.4];
        let auditor = CanaryScoreAuditor::new(in_scores, out_scores);
        let eps = auditor.epsilon_clopper_pearson(0.05, 1e-6, true, ThresholdStrategy::Bonferroni);
        assert!(eps.is_finite());
    }

    #[test]
    fn one_run_fdp_runs() {
        let in_scores = vec![1.0, 2.0, 3.0, 4.0];
        let out_scores = vec![0.1, 0.2, 0.3, 0.4];
        let auditor = CanaryScoreAuditor::new(in_scores, out_scores);
        let eps = auditor.epsilon_one_run(0.05, 1e-6, true, true, ThresholdStrategy::Bonferroni);
        assert!(eps.is_finite());
    }

    #[test]
    fn max_accuracy_runs() {
        let in_scores = vec![1.0, 2.0, 3.0, 4.0];
        let out_scores = vec![0.1, 0.2, 0.3, 0.4];
        let auditor = CanaryScoreAuditor::new(in_scores, out_scores);
        let acc = auditor.max_accuracy(None, None);
        assert!(acc.is_finite());
    }

    #[test]
    fn epsilon_from_gdp_runs() {
        let in_scores = vec![1.0, 2.0, 3.0, 4.0];
        let out_scores = vec![0.1, 0.2, 0.3, 0.4];
        let auditor = CanaryScoreAuditor::new(in_scores, out_scores);
        let eps = auditor.epsilon_from_gdp(0.05, 1e-6, 1e-4);
        assert!(eps.is_finite());
    }
}
