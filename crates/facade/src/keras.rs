//! Compatibility helpers inspired by the Python Keras API.
//!
//! This module focuses on the configuration, calibration, and validation
//! utilities that are independent of any specific ML framework. It does not
//! attempt to re-implement the Keras training loop itself.

use jax_privacy_accounting::{
    calibrate_noise_multiplier, DpParams, DpTrainingAccountant, DpsgdTrainingAccountant,
    PldAccountantConfig, Schedule,
};
use jax_privacy_core::{DpError, DpSgdAggregator, Result};
use jax_privacy_prng::JaxKey;

/// Parameters for DP-SGD style training configuration.
#[derive(Clone, Debug)]
pub struct DpKerasConfig {
    /// Target epsilon.
    pub epsilon: f64,
    /// Target delta.
    pub delta: f64,
    /// L2 clipping norm.
    pub clipping_norm: f64,
    /// Physical batch size.
    pub batch_size: usize,
    /// Gradient accumulation steps.
    pub gradient_accumulation_steps: usize,
    /// Total training steps.
    pub train_steps: usize,
    /// Total number of training examples.
    pub train_size: usize,
    /// Optional pre-specified noise multiplier.
    pub noise_multiplier: Option<f64>,
    /// Whether to rescale clipped gradients to unit norm.
    pub rescale_to_unit_norm: bool,
    /// Optional microbatch size for per-example gradient computation.
    pub microbatch_size: Option<usize>,
    /// Optional RNG seed.
    pub seed: Option<u64>,
}

impl DpKerasConfig {
    /// Effective batch size (physical batch size * accumulation steps).
    pub fn effective_batch_size(&self) -> usize {
        self.batch_size
            .saturating_mul(self.gradient_accumulation_steps.max(1))
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(DpError::invalid("epsilon must be positive"));
        }
        if !self.delta.is_finite() || self.delta <= 0.0 || self.delta > 1.0 {
            return Err(DpError::invalid("delta must be in (0, 1]"));
        }
        if !self.clipping_norm.is_finite() || self.clipping_norm <= 0.0 {
            return Err(DpError::invalid("clipping_norm must be positive"));
        }
        if self.batch_size == 0 {
            return Err(DpError::invalid("batch_size must be positive"));
        }
        if self.gradient_accumulation_steps == 0 {
            return Err(DpError::invalid(
                "gradient_accumulation_steps must be positive",
            ));
        }
        if self.train_steps == 0 {
            return Err(DpError::invalid("train_steps must be positive"));
        }
        if self.train_size == 0 {
            return Err(DpError::invalid("train_size must be positive"));
        }
        if let Some(micro) = self.microbatch_size {
            if micro == 0 {
                return Err(DpError::invalid("microbatch_size must be positive"));
            }
            if self.batch_size % micro != 0 {
                return Err(DpError::invalid(
                    "batch_size must be divisible by microbatch_size",
                ));
            }
        }

        if let Some(noise) = self.noise_multiplier {
            if !noise.is_finite() || noise <= 0.0 {
                return Err(DpError::invalid("noise_multiplier must be positive"));
            }
            let eps = self.resulting_epsilon(noise)?;
            let tolerance = 1e-1;
            if eps > self.epsilon + tolerance {
                return Err(DpError::invalid(
                    "noise_multiplier exceeds the configured epsilon budget",
                ));
            }
        }

        Ok(())
    }

    /// Resolve a noise multiplier, calibrating if needed.
    pub fn resolve_noise_multiplier(&self) -> Result<f64> {
        if let Some(noise) = self.noise_multiplier {
            return Ok(noise);
        }
        let calibrated = self.update_with_calibrated_noise_multiplier()?;
        calibrated
            .noise_multiplier
            .ok_or_else(|| DpError::invalid("calibration did not return a noise multiplier"))
    }

    /// Return a copy of this config with a calibrated noise multiplier.
    pub fn update_with_calibrated_noise_multiplier(&self) -> Result<Self> {
        let accountant = DpsgdTrainingAccountant::with_pld(PldAccountantConfig::new(1e-3));
        let sigma = calibrate_noise_multiplier(
            self.epsilon,
            &accountant,
            Schedule::constant(1.0),
            Schedule::constant(self.effective_batch_size()),
            self.train_steps as u64,
            self.train_size,
            self.delta,
            None,
            None,
            None,
            50.0,
            1e-3,
            1e-6,
        )?;

        let mut updated = self.clone();
        updated.noise_multiplier = Some(sigma);
        Ok(updated)
    }

    /// Compute the resulting epsilon for a specific noise multiplier.
    pub fn resulting_epsilon(&self, noise_multiplier: f64) -> Result<f64> {
        let accountant = DpsgdTrainingAccountant::with_pld(PldAccountantConfig::new(1e-3));
        let params = DpParams::new(
            noise_multiplier,
            self.train_size,
            self.batch_size,
            self.delta,
        )?;
        accountant.compute_epsilon(self.train_steps as u64, &params)
    }

    /// Build a DP-SGD aggregator using the config settings.
    pub fn build_aggregator(&self, key: JaxKey) -> Result<DpSgdAggregator> {
        self.validate()?;
        let noise_multiplier = self.resolve_noise_multiplier()?;
        let agg = DpSgdAggregator::new(
            self.clipping_norm,
            self.rescale_to_unit_norm,
            self.batch_size as f64,
            noise_multiplier,
            true,
            self.microbatch_size,
            key,
        )
        .with_accumulation_steps(self.gradient_accumulation_steps);
        Ok(agg)
    }
}

/// Calculate the number of optimizer steps performed by a training run.
pub fn calculate_optimizer_steps_to_perform(
    train_size: usize,
    batch_size: usize,
    epochs: usize,
    initial_epoch: usize,
    steps_per_epoch: Option<usize>,
) -> usize {
    let epochs_to_perform = epochs.saturating_sub(initial_epoch);
    let steps_per_epoch = steps_per_epoch.unwrap_or_else(|| train_size / batch_size.max(1));
    steps_per_epoch.saturating_mul(epochs_to_perform)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_batch_size_scales() {
        let cfg = DpKerasConfig {
            epsilon: 100.0,
            delta: 1e-5,
            clipping_norm: 1.0,
            batch_size: 8,
            gradient_accumulation_steps: 4,
            train_steps: 1,
            train_size: 100,
            noise_multiplier: None,
            rescale_to_unit_norm: true,
            microbatch_size: None,
            seed: None,
        };
        assert_eq!(cfg.effective_batch_size(), 32);
    }

    #[test]
    fn optimizer_steps_matches_epoch_math() {
        let steps = calculate_optimizer_steps_to_perform(100, 10, 5, 1, None);
        assert_eq!(steps, 40);
    }

    #[test]
    fn build_aggregator_applies_accumulation() {
        let cfg = DpKerasConfig {
            epsilon: 1.0,
            delta: 1e-5,
            clipping_norm: 1.0,
            batch_size: 8,
            gradient_accumulation_steps: 4,
            train_steps: 10,
            train_size: 80,
            noise_multiplier: Some(100.0),
            rescale_to_unit_norm: true,
            microbatch_size: None,
            seed: None,
        };
        let agg = cfg.build_aggregator(JaxKey::new(0)).expect("aggregator");
        assert_eq!(agg.accumulation_steps, 4);
    }

    #[test]
    fn resulting_epsilon_is_finite() {
        let cfg = DpKerasConfig {
            epsilon: 10.0,
            delta: 1e-5,
            clipping_norm: 1.0,
            batch_size: 8,
            gradient_accumulation_steps: 1,
            train_steps: 5,
            train_size: 80,
            noise_multiplier: None,
            rescale_to_unit_norm: true,
            microbatch_size: None,
            seed: None,
        };
        let eps = cfg.resulting_epsilon(5.0).expect("epsilon");
        assert!(eps.is_finite());
        assert!(eps >= 0.0);
    }
}
