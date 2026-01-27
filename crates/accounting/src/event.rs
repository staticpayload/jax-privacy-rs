//! Experimental DP event helpers.
//!
//! These mirror the intent of `dp_accounting.DpEvent` constructors from the
//! Python implementation while using the in-crate RDP accountant.

use crate::pld::PldAccountant;
use crate::rdp::RdpAccountant;

/// A composable description of a DP mechanism.
#[derive(Clone, Debug)]
pub enum DpEvent {
    /// A Gaussian mechanism with the given noise multiplier.
    Gaussian {
        /// Noise multiplier (standard deviation divided by sensitivity).
        noise_multiplier: f64,
    },
    /// Poisson subsampling applied to an inner event.
    PoissonSampled {
        /// Poisson sampling probability.
        sampling_prob: f64,
        /// The inner event.
        event: Box<DpEvent>,
    },
    /// Truncated Poisson-sampled Gaussian mechanism.
    TruncatedSubsampledGaussian {
        /// Dataset size.
        dataset_size: usize,
        /// Sampling probability before truncation.
        sampling_probability: f64,
        /// Truncation cap.
        truncated_batch_size: usize,
        /// Noise multiplier.
        noise_multiplier: f64,
    },
    /// An event composed with itself multiple times.
    SelfComposed {
        /// The inner event.
        event: Box<DpEvent>,
        /// Number of compositions.
        count: u64,
    },
}

impl DpEvent {
    /// Compute epsilon at the provided delta using the RDP accountant.
    pub fn epsilon(&self, delta: f64) -> f64 {
        if delta <= 0.0 || delta >= 1.0 || !delta.is_finite() {
            return f64::INFINITY;
        }
        let mut acct = RdpAccountant::new();
        self.apply(&mut acct);
        acct.epsilon(delta)
    }

    /// Compute epsilon using the PLD accountant (falls back to RDP when needed).
    pub fn epsilon_pld(&self, delta: f64) -> f64 {
        if delta <= 0.0 || delta >= 1.0 || !delta.is_finite() {
            return f64::INFINITY;
        }
        let mut acct = PldAccountant::new();
        self.apply_pld(&mut acct);
        acct.epsilon(delta)
    }

    fn apply(&self, acct: &mut RdpAccountant) {
        match self {
            DpEvent::Gaussian { noise_multiplier } => {
                acct.step(*noise_multiplier, 1.0);
            }
            DpEvent::PoissonSampled {
                sampling_prob,
                event,
            } => apply_with_sampling(acct, *sampling_prob, event),
            DpEvent::TruncatedSubsampledGaussian {
                dataset_size,
                sampling_probability,
                truncated_batch_size,
                noise_multiplier,
            } => {
                acct.truncated_step(
                    *dataset_size,
                    *sampling_probability,
                    *truncated_batch_size,
                    *noise_multiplier,
                );
            }
            DpEvent::SelfComposed { event, count } => {
                for _ in 0..*count {
                    event.apply(acct);
                }
            }
        }
    }

    fn apply_pld(&self, acct: &mut PldAccountant) {
        match self {
            DpEvent::Gaussian { noise_multiplier } => {
                acct.step(*noise_multiplier, 1.0);
            }
            DpEvent::PoissonSampled {
                sampling_prob,
                event,
            } => apply_with_sampling_pld(acct, *sampling_prob, event),
            DpEvent::TruncatedSubsampledGaussian {
                dataset_size,
                sampling_probability,
                truncated_batch_size,
                noise_multiplier,
            } => {
                let q_cap = (*truncated_batch_size as f64) / (*dataset_size as f64);
                let q = sampling_probability.min(q_cap).clamp(0.0, 1.0);
                acct.step(*noise_multiplier, q);
            }
            DpEvent::SelfComposed { event, count } => {
                for _ in 0..*count {
                    event.apply_pld(acct);
                }
            }
        }
    }
}

fn apply_with_sampling(acct: &mut RdpAccountant, sampling_prob: f64, event: &DpEvent) {
    let q = sampling_prob.clamp(0.0, 1.0);
    match event {
        DpEvent::Gaussian { noise_multiplier } => {
            acct.step(*noise_multiplier, q);
        }
        DpEvent::SelfComposed { event, count } => {
            for _ in 0..*count {
                apply_with_sampling(acct, q, event);
            }
        }
        DpEvent::TruncatedSubsampledGaussian { .. } => {
            // Nested truncation with additional Poisson sampling does not have
            // a tight closed form here; fall back to applying the inner event.
            event.apply(acct);
        }
        DpEvent::PoissonSampled { .. } => {
            // Compose sampling probabilities conservatively.
            let nested_q = combined_sampling_prob(q, event);
            if let Some(noise) = extract_noise_multiplier(event) {
                acct.step(noise, nested_q);
            } else {
                event.apply(acct);
            }
        }
    }
}

fn apply_with_sampling_pld(acct: &mut PldAccountant, sampling_prob: f64, event: &DpEvent) {
    let q = sampling_prob.clamp(0.0, 1.0);
    match event {
        DpEvent::Gaussian { noise_multiplier } => {
            acct.step(*noise_multiplier, q);
        }
        DpEvent::SelfComposed { event, count } => {
            for _ in 0..*count {
                apply_with_sampling_pld(acct, q, event);
            }
        }
        DpEvent::TruncatedSubsampledGaussian { .. } => {
            event.apply_pld(acct);
        }
        DpEvent::PoissonSampled { .. } => {
            let nested_q = combined_sampling_prob(q, event);
            if let Some(noise) = extract_noise_multiplier(event) {
                acct.step(noise, nested_q);
            } else {
                event.apply_pld(acct);
            }
        }
    }
}

fn combined_sampling_prob(q: f64, event: &DpEvent) -> f64 {
    match event {
        DpEvent::PoissonSampled { sampling_prob, .. } => q * sampling_prob.clamp(0.0, 1.0),
        _ => q,
    }
}

fn extract_noise_multiplier(event: &DpEvent) -> Option<f64> {
    match event {
        DpEvent::Gaussian { noise_multiplier } => Some(*noise_multiplier),
        DpEvent::PoissonSampled { event, .. } => extract_noise_multiplier(event),
        DpEvent::SelfComposed { event, .. } => extract_noise_multiplier(event),
        DpEvent::TruncatedSubsampledGaussian {
            noise_multiplier, ..
        } => Some(*noise_multiplier),
    }
}

fn validate_args(noise_multiplier: f64, iterations: u64, sampling_prob: f64) {
    assert!(
        noise_multiplier >= 0.0 && noise_multiplier.is_finite(),
        "noise multiplier must be non-negative and finite"
    );
    assert!(iterations as i128 >= 0, "iterations must be non-negative");
    assert!(
        (0.0..=1.0).contains(&sampling_prob) && sampling_prob.is_finite(),
        "sampling probability must be in [0, 1]"
    );
}

/// DP-SGD event: Poisson sampled Gaussian repeated `iterations` times.
pub fn dpsgd_event(noise_multiplier: f64, iterations: u64, sampling_prob: f64) -> DpEvent {
    validate_args(noise_multiplier, iterations, sampling_prob);
    let gaussian = DpEvent::Gaussian { noise_multiplier };
    let sampled = DpEvent::PoissonSampled {
        sampling_prob,
        event: Box::new(gaussian),
    };
    DpEvent::SelfComposed {
        event: Box::new(sampled),
        count: iterations,
    }
}

/// Truncated DP-SGD event.
pub fn truncated_dpsgd_event(
    noise_multiplier: f64,
    iterations: u64,
    sampling_prob: f64,
    num_examples: usize,
    truncated_batch_size: usize,
) -> DpEvent {
    validate_args(noise_multiplier, iterations, sampling_prob);
    assert!(num_examples > 0, "num_examples must be positive");
    assert!(
        truncated_batch_size > 0,
        "truncated_batch_size must be positive"
    );
    let sampled_gaussian = DpEvent::TruncatedSubsampledGaussian {
        dataset_size: num_examples,
        sampling_probability: sampling_prob,
        truncated_batch_size,
        noise_multiplier,
    };
    DpEvent::SelfComposed {
        event: Box::new(sampled_gaussian),
        count: iterations,
    }
}

/// Amplified BandMF event (grouping into `num_bands`).
pub fn amplified_bandmf_event(
    noise_multiplier: f64,
    iterations: u64,
    num_bands: usize,
    sampling_prob: f64,
) -> DpEvent {
    validate_args(noise_multiplier, iterations, sampling_prob);
    assert!(num_bands >= 1, "num_bands must be >= 1");
    let rounds = div_ceil_u64(iterations, num_bands as u64);
    dpsgd_event(noise_multiplier, rounds, sampling_prob)
}

/// Truncated amplified BandMF event.
pub fn truncated_amplified_bandmf_event(
    noise_multiplier: f64,
    iterations: u64,
    num_bands: usize,
    sampling_prob: f64,
    largest_group_size: usize,
    truncated_batch_size: usize,
) -> DpEvent {
    validate_args(noise_multiplier, iterations, sampling_prob);
    assert!(num_bands >= 1, "num_bands must be >= 1");
    let rounds = div_ceil_u64(iterations, num_bands as u64);
    truncated_dpsgd_event(
        noise_multiplier,
        rounds,
        sampling_prob,
        largest_group_size,
        truncated_batch_size,
    )
}

fn div_ceil_u64(x: u64, y: u64) -> u64 {
    if y == 0 {
        return x;
    }
    (x + y - 1) / y
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn epsilon_increases_with_iterations() {
        let e1 = dpsgd_event(2.0, 10, 0.1).epsilon(1e-6);
        let e2 = dpsgd_event(2.0, 20, 0.1).epsilon(1e-6);
        assert!(e2 >= e1);
    }

    #[test]
    fn epsilon_pld_is_finite() {
        let eps = dpsgd_event(1.0, 5, 1.0).epsilon_pld(1e-6);
        assert!(eps.is_finite());
        assert!(eps > 0.0);
    }

    #[test]
    fn bandmf_rounds_match_div_ceil() {
        let event = amplified_bandmf_event(1.0, 5, 2, 0.5);
        match event {
            DpEvent::SelfComposed { count, .. } => assert_eq!(count, 3),
            _ => panic!("expected self composed event"),
        }
    }

    #[test]
    fn dpsgd_event_matches_rdp_accountant() {
        let sigma = 1.5;
        let q = 0.2;
        let steps = 25u64;
        let delta = 1e-6;

        let event_eps = dpsgd_event(sigma, steps, q).epsilon(delta);

        let mut acct = crate::rdp::RdpAccountant::new();
        acct.steps(sigma, q, steps as usize);
        let acct_eps = acct.epsilon(delta);

        assert!((event_eps - acct_eps).abs() <= 1e-9);
    }

    #[test]
    fn amplified_bandmf_matches_rounded_rdp() {
        let sigma = 1.2;
        let q = 0.3;
        let steps = 17u64;
        let bands = 4usize;
        let delta = 1e-6;

        let event_eps = amplified_bandmf_event(sigma, steps, bands, q).epsilon(delta);

        let rounds = (steps + bands as u64 - 1) / bands as u64;
        let mut acct = crate::rdp::RdpAccountant::new();
        acct.steps(sigma, q, rounds as usize);
        let acct_eps = acct.epsilon(delta);

        assert!((event_eps - acct_eps).abs() <= 1e-9);
    }

    proptest! {
        #[test]
        fn more_noise_reduces_epsilon(
            noise_a in 0.5f64..10.0,
            noise_b in 0.5f64..10.0,
            iterations in 1u64..200,
            sampling_prob in 0.01f64..0.9,
        ) {
            let low = noise_a.min(noise_b);
            let high = noise_a.max(noise_b);
            let eps_low = dpsgd_event(low, iterations, sampling_prob).epsilon(1e-6);
            let eps_high = dpsgd_event(high, iterations, sampling_prob).epsilon(1e-6);
            prop_assert!(eps_high <= eps_low + 1e-9);
        }

        #[test]
        fn higher_sampling_increases_epsilon(
            noise in 0.5f64..5.0,
            iterations in 1u64..200,
            q_a in 0.01f64..0.9,
            q_b in 0.01f64..0.9,
        ) {
            let low_q = q_a.min(q_b);
            let high_q = q_a.max(q_b);
            let eps_low = dpsgd_event(noise, iterations, low_q).epsilon(1e-6);
            let eps_high = dpsgd_event(noise, iterations, high_q).epsilon(1e-6);
            prop_assert!(eps_high >= eps_low - 1e-9);
        }
    }
}
