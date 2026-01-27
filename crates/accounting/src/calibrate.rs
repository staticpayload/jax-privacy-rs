//! Calibration utilities for DP hyper-parameters.

use jax_privacy_core::{DpError, Result};

use crate::{DpEvent, DpParams, DpTrainingAccountant, Sampler, Schedule};

fn schedule_at_zero<T: Clone>(schedule: &Schedule<T>) -> Option<T> {
    schedule.normalize(0).first().map(|(_, v)| v.clone())
}

fn scale_noise_schedule(schedule: &Schedule<f64>, factor: f64) -> Schedule<f64> {
    match schedule {
        Schedule::Constant(v) => Schedule::Constant(v * factor),
        Schedule::Timed(points) => {
            Schedule::Timed(points.iter().map(|(t, v)| (*t, v * factor)).collect())
        }
    }
}

fn make_dp_params(
    noise_schedule: Schedule<f64>,
    batch_schedule: Schedule<usize>,
    num_samples: usize,
    delta: f64,
    sampler: Sampler,
    examples_per_user: Option<usize>,
    cycle_length: Option<usize>,
    truncated_batch_size: Option<usize>,
) -> Result<DpParams> {
    let base_noise = schedule_at_zero(&noise_schedule)
        .ok_or_else(|| DpError::invalid("noise schedule must have a value"))?;
    let base_batch = schedule_at_zero(&batch_schedule)
        .ok_or_else(|| DpError::invalid("batch schedule must have a value"))?;

    let mut params = DpParams::new(base_noise, num_samples, base_batch, delta)?;
    params.noise_multipliers = Some(noise_schedule);
    params.batch_sizes = batch_schedule;
    params.sampler = sampler;
    params.examples_per_user = examples_per_user;
    params.cycle_length = cycle_length;
    params.truncated_batch_size = truncated_batch_size;
    params.validate()?;
    Ok(params)
}

/// Calibrate the number of updates to meet a target epsilon.
pub fn calibrate_num_updates<A: DpTrainingAccountant>(
    target_epsilon: f64,
    accountant: &A,
    noise_multipliers: Schedule<f64>,
    batch_sizes: Schedule<usize>,
    num_samples: usize,
    target_delta: f64,
    examples_per_user: Option<usize>,
    cycle_length: Option<usize>,
    truncated_batch_size: Option<usize>,
    initial_max_updates: u64,
    initial_min_updates: u64,
) -> Result<u64> {
    if !accountant.can_calibrate_steps() {
        return Err(DpError::unsupported("accountant cannot calibrate steps"));
    }

    let sampler = Sampler::Poisson;

    let epsilon_for = |num_updates: u64| -> Result<f64> {
        let params = make_dp_params(
            noise_multipliers.clone(),
            batch_sizes.clone(),
            num_samples,
            target_delta,
            sampler,
            examples_per_user,
            cycle_length,
            truncated_batch_size,
        )?;
        accountant.compute_epsilon(num_updates, &params)
    };

    let min_updates = initial_min_updates.max(1);
    let mut max_updates = initial_max_updates.max(min_updates + 1);

    if epsilon_for(min_updates)? > target_epsilon {
        return Err(DpError::invalid(
            "epsilon at the minimum updates already exceeds the target",
        ));
    }

    while epsilon_for(max_updates)? < target_epsilon {
        max_updates = max_updates.saturating_mul(2);
        if max_updates > (1 << 60) {
            break;
        }
    }

    // Binary search on discrete steps.
    let mut lo = min_updates;
    let mut hi = max_updates;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        let eps = epsilon_for(mid)?;
        if eps <= target_epsilon {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    if let Some(cycle) = cycle_length.filter(|c| *c > 1) {
        let cycle = cycle as u64;
        Ok(((lo + cycle - 1) / cycle) * cycle)
    } else {
        Ok(lo)
    }
}

/// Calibrate the noise multiplier to meet a target epsilon.
pub fn calibrate_noise_multiplier<A: DpTrainingAccountant>(
    target_epsilon: f64,
    accountant: &A,
    base_noise_schedule: Schedule<f64>,
    batch_sizes: Schedule<usize>,
    num_updates: u64,
    num_samples: usize,
    target_delta: f64,
    examples_per_user: Option<usize>,
    cycle_length: Option<usize>,
    truncated_batch_size: Option<usize>,
    initial_max_noise: f64,
    initial_min_noise: f64,
    tol: f64,
) -> Result<f64> {
    if !accountant.can_calibrate_noise_multipliers() {
        return Err(DpError::unsupported(
            "accountant cannot calibrate noise multipliers",
        ));
    }

    let sampler = Sampler::Poisson;

    let epsilon_for = |scale: f64| -> Result<f64> {
        let noise_schedule = scale_noise_schedule(&base_noise_schedule, scale);
        let params = make_dp_params(
            noise_schedule,
            batch_sizes.clone(),
            num_samples,
            target_delta,
            sampler,
            examples_per_user,
            cycle_length,
            truncated_batch_size,
        )?;
        accountant.compute_epsilon(num_updates, &params)
    };

    let mut lo = initial_min_noise.max(0.0);
    let mut hi = initial_max_noise.max(lo + tol.max(1e-6));

    while epsilon_for(hi)? > target_epsilon {
        lo = hi;
        hi *= 2.0;
        if hi > 1e6 {
            break;
        }
    }

    // Binary search for smallest scale such that epsilon <= target.
    let tol = tol.max(1e-6);
    for _ in 0..80 {
        if (hi - lo).abs() <= tol {
            break;
        }
        let mid = 0.5 * (lo + hi);
        let eps = epsilon_for(mid)?;
        if eps <= target_epsilon {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    Ok(hi)
}

/// Calibrate the batch size to meet a target epsilon.
pub fn calibrate_batch_size<A: DpTrainingAccountant>(
    target_epsilon: f64,
    accountant: &A,
    noise_multipliers: Schedule<f64>,
    num_updates: u64,
    num_samples: usize,
    target_delta: f64,
    examples_per_user: Option<usize>,
    cycle_length: Option<usize>,
    truncated_batch_size: Option<usize>,
    initial_max_batch_size: usize,
    initial_min_batch_size: usize,
) -> Result<usize> {
    if !accountant.can_calibrate_batch_size() {
        return Err(DpError::unsupported(
            "accountant cannot calibrate batch size",
        ));
    }

    let sampler = Sampler::Poisson;

    let epsilon_for = |batch_size: usize| -> Result<f64> {
        let batch_schedule = Schedule::constant(batch_size);
        let params = make_dp_params(
            noise_multipliers.clone(),
            batch_schedule,
            num_samples,
            target_delta,
            sampler,
            examples_per_user,
            cycle_length,
            truncated_batch_size,
        )?;
        accountant.compute_epsilon(num_updates, &params)
    };

    let min_batch_size = initial_min_batch_size.max(1);
    if epsilon_for(min_batch_size)? > target_epsilon {
        return Err(DpError::invalid(
            "epsilon at the minimum batch size already exceeds the target",
        ));
    }

    let mut lo = min_batch_size;
    let mut hi = initial_max_batch_size.max(lo + 1);
    while epsilon_for(hi)? < target_epsilon && hi < num_samples {
        lo = hi;
        hi = (hi.saturating_mul(2)).min(num_samples);
        if hi == lo {
            break;
        }
    }

    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        let eps = epsilon_for(mid)?;
        if eps <= target_epsilon {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Ok(lo)
}

/// Calibrate a monotone DP mechanism parameter using `DpEvent`.
///
/// This mirrors the spirit of `dp_accounting.calibrate_dp_mechanism` from the
/// Python ecosystem, assuming that increasing the parameter reduces epsilon
/// (e.g., a noise multiplier).
pub fn calibrate_dp_mechanism(
    make_event: impl Fn(f64) -> DpEvent,
    target_epsilon: f64,
    target_delta: f64,
    lower: f64,
    upper: f64,
    tol: f64,
    max_iters: usize,
) -> Result<f64> {
    if !target_epsilon.is_finite() || target_epsilon < 0.0 {
        return Err(DpError::invalid("target_epsilon must be finite and >= 0"));
    }
    if !target_delta.is_finite() || target_delta <= 0.0 || target_delta >= 1.0 {
        return Err(DpError::invalid("target_delta must be in (0, 1)"));
    }

    let tol = tol.max(1e-6);
    let mut lo = lower.max(0.0);
    let mut hi = upper.max(lo + tol);

    let eps_lo = make_event(lo).epsilon(target_delta);
    if eps_lo <= target_epsilon {
        return Ok(lo);
    }

    let mut eps_hi = make_event(hi).epsilon(target_delta);
    let mut expansions = 0usize;
    while eps_hi > target_epsilon && hi < 1e6 && expansions < 60 {
        lo = hi;
        hi *= 2.0;
        eps_hi = make_event(hi).epsilon(target_delta);
        expansions += 1;
    }

    if eps_hi > target_epsilon {
        return Err(DpError::invalid(
            "failed to bracket a parameter achieving the target epsilon",
        ));
    }

    let iters = max_iters.max(1);
    for _ in 0..iters {
        if (hi - lo).abs() <= tol {
            break;
        }
        let mid = 0.5 * (lo + hi);
        let eps_mid = make_event(mid).epsilon(target_delta);
        if eps_mid <= target_epsilon {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    Ok(hi)
}

/// Calibrate a monotone DP mechanism parameter using PLD epsilon evaluation.
pub fn calibrate_dp_mechanism_pld(
    make_event: impl Fn(f64) -> DpEvent,
    target_epsilon: f64,
    target_delta: f64,
    lower: f64,
    upper: f64,
    tol: f64,
    max_iters: usize,
) -> Result<f64> {
    if !target_epsilon.is_finite() || target_epsilon < 0.0 {
        return Err(DpError::invalid("target_epsilon must be finite and >= 0"));
    }
    if !target_delta.is_finite() || target_delta <= 0.0 || target_delta >= 1.0 {
        return Err(DpError::invalid("target_delta must be in (0, 1)"));
    }

    let tol = tol.max(1e-6);
    let mut lo = lower.max(0.0);
    let mut hi = upper.max(lo + tol);

    let eps_lo = make_event(lo).epsilon_pld(target_delta);
    if eps_lo <= target_epsilon {
        return Ok(lo);
    }

    let mut eps_hi = make_event(hi).epsilon_pld(target_delta);
    let mut expansions = 0usize;
    while eps_hi > target_epsilon && hi < 1e6 && expansions < 60 {
        lo = hi;
        hi *= 2.0;
        eps_hi = make_event(hi).epsilon_pld(target_delta);
        expansions += 1;
    }

    if eps_hi > target_epsilon {
        return Err(DpError::invalid(
            "failed to bracket a parameter achieving the target epsilon",
        ));
    }

    let iters = max_iters.max(1);
    for _ in 0..iters {
        if (hi - lo).abs() <= tol {
            break;
        }
        let mid = 0.5 * (lo + hi);
        let eps_mid = make_event(mid).epsilon_pld(target_delta);
        if eps_mid <= target_epsilon {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    Ok(hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dpsgd_event, DpEvent, DpParams, DpsgdTrainingAccountant, PldAccountantConfig};
    use proptest::prelude::*;

    #[test]
    fn calibration_supports_truncation_conservatively() {
        let accountant = DpsgdTrainingAccountant::with_pld(PldAccountantConfig::default());
        let noise = Schedule::constant(1.0);
        let batch = Schedule::constant(64usize);

        let res = calibrate_noise_multiplier(
            2.0,
            &accountant,
            noise,
            batch,
            8,
            2_000,
            1e-6,
            None,
            None,
            Some(64),
            10.0,
            0.1,
            1e-3,
        );
        let sigma = res.expect("truncated calibration should run");
        assert!(sigma.is_finite());
        assert!(sigma > 0.0);
    }

    #[test]
    fn calibrate_dp_mechanism_brackets_noise_multiplier() {
        let steps = 128u64;
        let q = 0.01;
        let delta = 1e-6;
        let target_eps = 5.0;

        let sigma = calibrate_dp_mechanism(
            |s| dpsgd_event(s, steps, q),
            target_eps,
            delta,
            0.1,
            4.0,
            1e-3,
            80,
        )
        .expect("calibrated sigma");

        let eps = dpsgd_event(sigma, steps, q).epsilon(delta);
        assert!(sigma.is_finite() && sigma > 0.0);
        assert!(eps <= target_eps + 1e-6);
    }

    #[test]
    fn calibrate_dp_mechanism_pld_brackets_noise_multiplier() {
        let delta = 1e-6;
        let target_eps = 2.0;

        let sigma = calibrate_dp_mechanism_pld(
            |s| DpEvent::Gaussian {
                noise_multiplier: s,
            },
            target_eps,
            delta,
            0.5,
            4.0,
            1e-1,
            8,
        )
        .expect("calibrated sigma");

        let eps = DpEvent::Gaussian {
            noise_multiplier: sigma,
        }
        .epsilon_pld(delta);
        assert!(sigma.is_finite() && sigma > 0.0);
        assert!(eps <= target_eps + 1e-6);
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 8, .. ProptestConfig::default() })]

        #[test]
        fn calibrate_noise_multiplier_meets_target(
            target_eps in 1.0f64..8.0,
            steps in 16u64..128,
            batch in 64usize..512,
            num_samples in 5_000usize..20_000,
        ) {
            prop_assume!(batch < num_samples);

            let accountant = DpsgdTrainingAccountant::default();
            let sigma = calibrate_noise_multiplier(
                target_eps,
                &accountant,
                Schedule::constant(1.0),
                Schedule::constant(batch),
                steps,
                num_samples,
                1e-6,
                None,
                None,
                None,
                10.0,
                0.1,
                1e-3,
            ).expect("sigma");

            let params = DpParams::new(sigma, num_samples, batch, 1e-6).expect("params");
            let eps = accountant.compute_epsilon(steps, &params).expect("epsilon");
            prop_assert!(eps <= target_eps + 1e-6);
        }
    }
}
