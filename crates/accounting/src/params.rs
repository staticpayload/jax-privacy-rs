//! DP training parameters and schedules.

use std::collections::BTreeMap;

use jax_privacy_core::{DpError, Result};

/// Sampling method for privacy analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampler {
    /// Poisson sampling (each example included independently).
    Poisson,
    /// Fixed batch size sampling (random subset without replacement).
    Fixed,
}

/// A piecewise-constant schedule keyed by update step.
#[derive(Clone, Debug, PartialEq)]
pub enum Schedule<T> {
    /// A constant value across all steps.
    Constant(T),
    /// A sequence of `(step, value)` change points.
    Timed(Vec<(u64, T)>),
}

impl<T: Clone> Schedule<T> {
    /// Construct a constant schedule.
    pub fn constant(value: T) -> Self {
        Self::Constant(value)
    }

    /// Construct a timed schedule.
    pub fn timed(points: Vec<(u64, T)>) -> Self {
        Self::Timed(points)
    }

    /// Normalize to a sorted list of change points up to `max_step`.
    pub fn normalize(&self, max_step: u64) -> Vec<(u64, T)> {
        let mut points: Vec<(u64, T)> = match self {
            Schedule::Constant(v) => vec![(0, v.clone())],
            Schedule::Timed(v) => v.clone(),
        };

        if points.is_empty() {
            return Vec::new();
        }

        // Sort by step and drop points after max_step.
        points.sort_by_key(|(t, _)| *t);
        points.retain(|(t, _)| *t <= max_step);
        if points.is_empty() {
            return Vec::new();
        }

        // Ensure step 0 is present; use the earliest value otherwise.
        if points[0].0 != 0 {
            let first_val = points[0].1.clone();
            points.insert(0, (0, first_val));
        }

        // Deduplicate identical step keys by keeping the last value.
        let mut deduped: Vec<(u64, T)> = Vec::with_capacity(points.len());
        for (step, value) in points {
            if let Some((prev_step, prev_value)) = deduped.last_mut() {
                if *prev_step == step {
                    *prev_value = value;
                    continue;
                }
            }
            deduped.push((step, value));
        }
        deduped
    }
}

/// Batch-size scale schedule keyed by step.
pub type BatchingScaleSchedule = BTreeMap<u64, u64>;

/// Parameters for DP training.
#[derive(Clone, Debug, PartialEq)]
pub struct DpParams {
    /// Noise multipliers schedule (sigma / sensitivity).
    pub noise_multipliers: Option<Schedule<f64>>,
    /// Dataset size.
    pub num_samples: usize,
    /// Target delta.
    pub delta: f64,
    /// Base batch size.
    pub batch_size: usize,
    /// Optional schedule that scales the base batch size over time.
    pub batch_size_scale_schedule: Option<BatchingScaleSchedule>,
    /// Whether the guarantee is expected to be finite.
    pub is_finite_guarantee: bool,
    /// Derived batch-size schedule.
    pub batch_sizes: Schedule<usize>,
    /// Maximum number of examples per user.
    pub examples_per_user: Option<usize>,
    /// Cycle length for cyclic Poisson sampling.
    pub cycle_length: Option<usize>,
    /// Sampling method assumed by accounting.
    pub sampler: Sampler,
    /// Optional truncated batch size for Poisson sampling.
    pub truncated_batch_size: Option<usize>,
}

impl DpParams {
    /// Create new DP parameters with constant noise and batch size.
    pub fn new(noise_mult: f64, num_samples: usize, batch_size: usize, delta: f64) -> Result<Self> {
        let mut params = Self {
            noise_multipliers: Some(Schedule::constant(noise_mult)),
            num_samples,
            delta,
            batch_size,
            batch_size_scale_schedule: None,
            is_finite_guarantee: true,
            batch_sizes: Schedule::constant(batch_size),
            examples_per_user: None,
            cycle_length: None,
            sampler: Sampler::Poisson,
            truncated_batch_size: None,
        };
        params.recompute_batch_sizes();
        params.validate()?;
        Ok(params)
    }

    /// Set a noise schedule.
    pub fn with_noise_schedule(mut self, schedule: Schedule<f64>) -> Result<Self> {
        self.noise_multipliers = Some(schedule);
        self.validate()?;
        Ok(self)
    }

    /// Set the batch-size scale schedule.
    pub fn with_batch_size_scale_schedule(
        mut self,
        schedule: BatchingScaleSchedule,
    ) -> Result<Self> {
        self.batch_size_scale_schedule = Some(schedule);
        self.recompute_batch_sizes();
        self.validate()?;
        Ok(self)
    }

    /// Set the sampling method.
    pub fn with_sampler(mut self, sampler: Sampler) -> Self {
        self.sampler = sampler;
        self
    }

    /// Set the examples-per-user constraint.
    pub fn with_examples_per_user(mut self, examples_per_user: usize) -> Self {
        self.examples_per_user = Some(examples_per_user);
        self
    }

    /// Set the cycle length.
    pub fn with_cycle_length(mut self, cycle_length: usize) -> Self {
        self.cycle_length = Some(cycle_length.max(1));
        self
    }

    /// Set the truncated batch size.
    pub fn with_truncated_batch_size(mut self, truncated_batch_size: usize) -> Self {
        self.truncated_batch_size = Some(truncated_batch_size);
        self
    }

    /// Compute the sampling rate using the base batch size.
    pub fn sampling_rate(&self) -> f64 {
        self.batch_size as f64 / self.num_samples as f64
    }

    fn recompute_batch_sizes(&mut self) {
        self.batch_sizes = match &self.batch_size_scale_schedule {
            None => Schedule::constant(self.batch_size),
            Some(schedule) => make_batch_size_boundaries(self.batch_size, schedule),
        };
    }

    /// Validate parameters.
    pub fn validate(&self) -> Result<()> {
        if self.noise_multipliers.is_none() && self.is_finite_guarantee {
            return Err(DpError::invalid(
                "noise_multipliers must be set for finite guarantees",
            ));
        }
        if let Some(schedule) = &self.noise_multipliers {
            let points = schedule.normalize(u64::MAX);
            if points.is_empty() {
                return Err(DpError::invalid("noise schedule must not be empty"));
            }
            if points.iter().any(|(_, nm)| !nm.is_finite() || *nm <= 0.0) {
                return Err(DpError::invalid(
                    "noise multipliers must be positive and finite",
                ));
            }
        }
        if self.num_samples == 0 {
            return Err(DpError::invalid("num_samples must be positive"));
        }
        if self.batch_size == 0 || self.batch_size > self.num_samples {
            return Err(DpError::invalid("batch_size must be in (0, num_samples]"));
        }
        if !self.delta.is_finite() || self.delta <= 0.0 || self.delta >= 1.0 {
            return Err(DpError::invalid("delta must be in (0, 1)"));
        }
        if let Some(examples_per_user) = self.examples_per_user {
            if examples_per_user == 0 {
                return Err(DpError::invalid("examples_per_user must be positive"));
            }
        }
        if let Some(cycle_length) = self.cycle_length {
            if cycle_length == 0 {
                return Err(DpError::invalid("cycle_length must be positive"));
            }
        }
        if let Some(truncated) = self.truncated_batch_size {
            if truncated == 0 {
                return Err(DpError::invalid("truncated_batch_size must be positive"));
            }
        }
        Ok(())
    }
}

fn make_batch_size_boundaries(
    base_batch_size: usize,
    schedule: &BatchingScaleSchedule,
) -> Schedule<usize> {
    let mut size = base_batch_size;
    let mut points: Vec<(u64, usize)> = Vec::with_capacity(schedule.len() + 1);
    points.push((0, size));
    for (threshold, scale) in schedule {
        size = size.saturating_mul(*scale as usize);
        points.push((*threshold, size));
    }
    Schedule::Timed(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_params() {
        let p = DpParams::new(1.0, 10_000, 100, 1e-5).expect("valid params");
        assert!((p.sampling_rate() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_schedule_normalize_inserts_zero() {
        let sched = Schedule::timed(vec![(10, 2.0), (20, 3.0)]);
        let norm = sched.normalize(15);
        assert_eq!(norm[0].0, 0);
        assert_eq!(norm[1].0, 10);
        assert_eq!(norm.len(), 2);
    }

    #[test]
    fn test_scale_schedule() {
        let mut schedule = BatchingScaleSchedule::new();
        schedule.insert(100, 4);
        schedule.insert(200, 2);
        let mut params = DpParams::new(1.0, 10_000, 10, 1e-5).expect("valid params");
        params = params
            .with_batch_size_scale_schedule(schedule)
            .expect("valid schedule");
        let points = params.batch_sizes.normalize(500);
        assert_eq!(points[0].1, 10);
        assert_eq!(points[1].1, 40);
        assert_eq!(points[2].1, 80);
    }
}
