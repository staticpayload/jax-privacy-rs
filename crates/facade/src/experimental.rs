//! Experimental DP execution-plan helpers.
//!
//! These mirror the intent of the Python experimental module while keeping the
//! implementation Rust-native and dependency-light.

pub use jax_privacy_accounting::DpEvent;

use jax_privacy_accounting::{
    amplified_bandmf_event, calibrate_noise_multiplier, dpsgd_event,
    truncated_amplified_bandmf_event, truncated_dpsgd_event, DpsgdTrainingAccountant,
    PldAccountantConfig, Schedule,
};
use jax_privacy_core::{
    clip_and_aggregate, BoundedSensitivityCallable, CyclicPoissonSampling, GradientTransform,
    NeighboringRelation, PartitionType, Tensor,
};
use jax_privacy_mf::{
    inverse_as_streaming_matrix, optimize_banded_toeplitz, sensitivity_squared,
    MatrixFactorizationPrivatizer, NoiseStrategy, ToeplitzInverseStream,
};
use jax_privacy_prng::JaxKey;

/// Concrete execution-plan wiring for common DP-SGD style flows.
pub struct DpExecutionPlan {
    /// Clipping + aggregation component.
    pub clipped_aggregation_fn:
        BoundedSensitivityCallable<Box<dyn Fn(Vec<Tensor>) -> Tensor + Send + Sync>>,
    /// Batch selection strategy.
    pub batch_selection_strategy: CyclicPoissonSampling,
    /// Noise addition component.
    pub noise_addition_transform: MatrixFactorizationPrivatizer<ToeplitzInverseStream>,
    /// Privacy event descriptor for the composed mechanism.
    pub dp_event: DpEvent,
}

/// Stateful components for executing a DP plan.
#[derive(Clone, Debug)]
pub struct DpExecutionPlanState {
    /// Noise-addition state.
    pub noise_state:
        <MatrixFactorizationPrivatizer<ToeplitzInverseStream> as jax_privacy_core::GradientTransform<
        Tensor,
    >>::State,
}

impl DpExecutionPlan {
    /// Apply the clipping/aggregation function.
    pub fn clipped_aggregate(&self, examples: Vec<Tensor>) -> Tensor {
        self.clipped_aggregation_fn.call(examples)
    }

    /// Compute epsilon for this plan at the provided delta.
    pub fn epsilon(&self, delta: f64) -> f64 {
        self.dp_event.epsilon(delta)
    }

    /// Compute epsilon using the PLD accountant (fallbacks to RDP when needed).
    pub fn epsilon_pld(&self, delta: f64) -> f64 {
        self.dp_event.epsilon_pld(delta)
    }

    /// Initialize state for noise addition.
    pub fn init_state(&self, model: &Tensor) -> DpExecutionPlanState {
        DpExecutionPlanState {
            noise_state: self.noise_addition_transform.init(model),
        }
    }

    /// Execute one DP step on pre-computed per-example gradients.
    pub fn step(
        &self,
        per_example_grads: Vec<Tensor>,
        state: DpExecutionPlanState,
    ) -> (Tensor, DpExecutionPlanState) {
        let clipped = self.clipped_aggregation_fn.call(per_example_grads);
        let (noisy, noise_state) = self
            .noise_addition_transform
            .update(&clipped, state.noise_state);
        (noisy, DpExecutionPlanState { noise_state })
    }
}

/// Configuration for a DP-SGD execution plan.
#[derive(Clone, Debug)]
pub struct DpsgdExecutionPlanConfig {
    /// Number of iterations to plan for.
    pub iterations: usize,
    /// Optional epsilon target; when set, we calibrate the noise multiplier.
    pub epsilon: Option<f64>,
    /// Target delta.
    pub delta: f64,
    /// Optional pre-specified noise multiplier.
    pub noise_multiplier: Option<f64>,
    /// Poisson sampling probability.
    pub sampling_prob: f64,
    /// Dataset size for accounting calibration.
    pub num_samples: usize,
    /// Effective batch size for accounting calibration.
    pub batch_size: usize,
    /// Per-example L2 clip norm.
    pub l2_clip_norm: f64,
    /// Whether to rescale to unit norm after clipping.
    pub rescale_to_unit_norm: bool,
    /// Optional normalization divisor; defaults to batch_size.
    pub normalize_by: Option<f64>,
    /// Optional truncated batch size.
    pub truncated_batch_size: Option<usize>,
    /// Partitioning strategy for batch selection.
    pub partition_type: PartitionType,
    /// Neighboring relation for sensitivity calculations.
    pub neighboring_relation: NeighboringRelation,
    /// Seed used for correlated noise generation.
    pub noise_seed: u64,
    /// Strategy for intermediate noise generation.
    pub noise_strategy: NoiseStrategy,
}

impl Default for DpsgdExecutionPlanConfig {
    fn default() -> Self {
        Self {
            iterations: 1,
            epsilon: None,
            delta: 1e-6,
            noise_multiplier: Some(1.0),
            sampling_prob: 1.0,
            num_samples: 1,
            batch_size: 1,
            l2_clip_norm: 1.0,
            rescale_to_unit_norm: false,
            normalize_by: None,
            truncated_batch_size: None,
            partition_type: PartitionType::Independent,
            neighboring_relation: NeighboringRelation::ReplaceSpecial,
            noise_seed: 0,
            noise_strategy: NoiseStrategy::Default,
        }
    }
}

impl DpsgdExecutionPlanConfig {
    /// Build a concrete execution plan.
    pub fn build(&self) -> DpExecutionPlan {
        self.validate();

        let nm = resolve_noise_multiplier_common(
            self.epsilon,
            self.noise_multiplier,
            self.batch_size,
            self.iterations as u64,
            self.num_samples,
            self.delta,
            Some(1),
            Some(1),
            self.truncated_batch_size,
        );

        let plan = build_plan_components(
            nm,
            self.sampling_prob,
            self.iterations,
            self.truncated_batch_size,
            1,
            self.partition_type,
            self.batch_size,
            self.l2_clip_norm,
            self.rescale_to_unit_norm,
            self.normalize_by,
            self.noise_seed,
            self.noise_strategy,
            1.0,
            self.neighboring_relation,
            vec![1.0],
        );

        let dp_event = if let Some(trunc) = self.truncated_batch_size {
            truncated_dpsgd_event(
                nm,
                self.iterations as u64,
                self.sampling_prob,
                self.num_samples,
                trunc,
            )
        } else {
            dpsgd_event(nm, self.iterations as u64, self.sampling_prob)
        };

        DpExecutionPlan { dp_event, ..plan }
    }

    fn validate(&self) {
        validate_common(
            self.epsilon,
            self.noise_multiplier,
            self.delta,
            self.sampling_prob,
            self.num_samples,
            self.batch_size,
        );
        validate_adjacency_relation(
            self.neighboring_relation,
            self.partition_type,
            self.truncated_batch_size,
        );
        assert!(self.l2_clip_norm > 0.0, "l2_clip_norm must be positive");
    }
}

/// Configuration for a banded matrix-factorization execution plan.
#[derive(Clone, Debug)]
pub struct BandMfExecutionPlanConfig {
    /// Number of iterations to plan for.
    pub iterations: usize,
    /// Number of bands in the strategy matrix.
    pub num_bands: usize,
    /// Optional epsilon target; when set, we calibrate the noise multiplier.
    pub epsilon: Option<f64>,
    /// Target delta.
    pub delta: f64,
    /// Optional pre-specified noise multiplier.
    pub noise_multiplier: Option<f64>,
    /// Poisson sampling probability.
    pub sampling_prob: f64,
    /// Dataset size for accounting calibration.
    pub num_samples: usize,
    /// Effective batch size for accounting calibration.
    pub batch_size: usize,
    /// Per-example L2 clip norm.
    pub l2_clip_norm: f64,
    /// Whether to rescale to unit norm after clipping.
    pub rescale_to_unit_norm: bool,
    /// Optional normalization divisor; defaults to batch_size.
    pub normalize_by: Option<f64>,
    /// Optional truncated batch size.
    pub truncated_batch_size: Option<usize>,
    /// Partitioning strategy for batch selection.
    pub partition_type: PartitionType,
    /// Neighboring relation for sensitivity calculations.
    pub neighboring_relation: NeighboringRelation,
    /// Steps to optimize the banded Toeplitz strategy.
    pub strategy_optimization_steps: usize,
    /// Seed used for correlated noise generation.
    pub noise_seed: u64,
    /// Strategy for intermediate noise generation.
    pub noise_strategy: NoiseStrategy,
}

impl Default for BandMfExecutionPlanConfig {
    fn default() -> Self {
        Self {
            iterations: 1,
            num_bands: 1,
            epsilon: None,
            delta: 1e-6,
            noise_multiplier: Some(1.0),
            sampling_prob: 1.0,
            num_samples: 1,
            batch_size: 1,
            l2_clip_norm: 1.0,
            rescale_to_unit_norm: false,
            normalize_by: None,
            truncated_batch_size: None,
            partition_type: PartitionType::EqualSplit,
            neighboring_relation: NeighboringRelation::ReplaceSpecial,
            strategy_optimization_steps: 500,
            noise_seed: 0,
            noise_strategy: NoiseStrategy::Default,
        }
    }
}

impl BandMfExecutionPlanConfig {
    /// Build a concrete execution plan.
    pub fn build(&self) -> DpExecutionPlan {
        self.validate();

        let bands = self.num_bands.max(1);
        let nm = resolve_noise_multiplier_common(
            self.epsilon,
            self.noise_multiplier,
            self.batch_size,
            self.iterations as u64,
            self.num_samples,
            self.delta,
            Some(1),
            Some(bands),
            self.truncated_batch_size,
        );

        let mf_strategy = optimize_banded_toeplitz(
            self.iterations.max(1),
            bands,
            self.strategy_optimization_steps,
        );
        let max_column_norm = sensitivity_squared(&mf_strategy, None).sqrt();

        let plan = build_plan_components(
            nm,
            self.sampling_prob,
            self.iterations,
            self.truncated_batch_size,
            bands,
            self.partition_type,
            self.batch_size,
            self.l2_clip_norm,
            self.rescale_to_unit_norm,
            self.normalize_by,
            self.noise_seed,
            self.noise_strategy,
            max_column_norm,
            self.neighboring_relation,
            mf_strategy,
        );

        let dp_event = if let Some(trunc) = self.truncated_batch_size {
            let largest_group_size = div_ceil(self.num_samples, bands);
            truncated_amplified_bandmf_event(
                nm,
                self.iterations as u64,
                bands,
                self.sampling_prob,
                largest_group_size,
                trunc,
            )
        } else {
            amplified_bandmf_event(nm, self.iterations as u64, bands, self.sampling_prob)
        };

        DpExecutionPlan { dp_event, ..plan }
    }

    fn validate(&self) {
        validate_common(
            self.epsilon,
            self.noise_multiplier,
            self.delta,
            self.sampling_prob,
            self.num_samples,
            self.batch_size,
        );
        validate_adjacency_relation(
            self.neighboring_relation,
            self.partition_type,
            self.truncated_batch_size,
        );
        assert!(self.num_bands >= 1, "num_bands must be >= 1");
        assert!(
            self.batch_size * self.num_bands.max(1) <= self.num_samples,
            "batch_size * num_bands must be <= num_samples"
        );
        assert!(self.l2_clip_norm > 0.0, "l2_clip_norm must be positive");
    }
}

fn build_plan_components(
    noise_multiplier: f64,
    sampling_prob: f64,
    iterations: usize,
    truncated_batch_size: Option<usize>,
    cycle_length: usize,
    partition_type: PartitionType,
    batch_size: usize,
    l2_clip_norm: f64,
    rescale_to_unit_norm: bool,
    normalize_by: Option<f64>,
    noise_seed: u64,
    noise_strategy: NoiseStrategy,
    max_column_norm: f64,
    neighboring_relation: NeighboringRelation,
    strategy_coeffs: Vec<f64>,
) -> DpExecutionPlan {
    let normalize_by = normalize_by.unwrap_or(batch_size as f64);

    let clipped = make_clipped_fn(l2_clip_norm, rescale_to_unit_norm, normalize_by);
    let norm_bound = if rescale_to_unit_norm {
        1.0
    } else {
        l2_clip_norm
    } / normalize_by.max(1e-12);
    let clipped_aggregation_fn = BoundedSensitivityCallable::new(clipped, norm_bound, false);
    let query_sensitivity = clipped_aggregation_fn.sensitivity(neighboring_relation);
    let stddev = noise_multiplier * query_sensitivity * max_column_norm;

    let batch_selection_strategy = CyclicPoissonSampling {
        sampling_prob,
        iterations,
        truncated_batch_size,
        cycle_length: cycle_length.max(1),
        partition_type,
    };

    let noising_matrix = inverse_as_streaming_matrix(&strategy_coeffs, None);
    let noise_addition_transform =
        MatrixFactorizationPrivatizer::new(noising_matrix, stddev, JaxKey::new(noise_seed))
            .with_strategy(noise_strategy);

    // Placeholder event; callers always replace this via struct update.
    let dp_event = dpsgd_event(noise_multiplier, iterations as u64, sampling_prob);

    DpExecutionPlan {
        clipped_aggregation_fn,
        batch_selection_strategy,
        noise_addition_transform,
        dp_event,
    }
}

fn make_clipped_fn(
    l2_clip_norm: f64,
    rescale_to_unit_norm: bool,
    normalize_by: f64,
) -> Box<dyn Fn(Vec<Tensor>) -> Tensor + Send + Sync> {
    Box::new(move |examples| {
        let (agg, _) = clip_and_aggregate(
            examples.as_slice(),
            l2_clip_norm,
            rescale_to_unit_norm,
            normalize_by,
            true,
            false,
        );
        agg
    })
}

fn validate_adjacency_relation(
    neighboring_relation: NeighboringRelation,
    partition_type: PartitionType,
    truncated_batch_size: Option<usize>,
) {
    if neighboring_relation == NeighboringRelation::AddOrRemoveOne {
        assert!(
            partition_type == PartitionType::Independent,
            "AddOrRemoveOne adjacency requires Independent partitioning"
        );
        assert!(
            truncated_batch_size.is_none(),
            "AddOrRemoveOne adjacency does not support truncated batches"
        );
    } else if partition_type == PartitionType::Independent && truncated_batch_size.is_some() {
        panic!("Independent partitioning is incompatible with truncated batches");
    }
}

fn resolve_noise_multiplier_common(
    epsilon: Option<f64>,
    noise_multiplier: Option<f64>,
    batch_size: usize,
    iterations: u64,
    num_samples: usize,
    delta: f64,
    examples_per_user: Option<usize>,
    cycle_length: Option<usize>,
    truncated_batch_size: Option<usize>,
) -> f64 {
    if let Some(nm) = noise_multiplier {
        return nm;
    }
    let epsilon = epsilon.expect("epsilon must be set when noise_multiplier is None");
    let accountant = if truncated_batch_size.is_some() {
        DpsgdTrainingAccountant::with_pld(PldAccountantConfig::default())
    } else {
        DpsgdTrainingAccountant::default()
    };
    calibrate_noise_multiplier(
        epsilon,
        &accountant,
        Schedule::Constant(1.0),
        Schedule::Constant(batch_size),
        iterations,
        num_samples,
        delta,
        examples_per_user,
        cycle_length,
        truncated_batch_size,
        50.0,
        1e-3,
        1e-6,
    )
    .expect("noise multiplier calibration failed")
}

fn validate_common(
    epsilon: Option<f64>,
    noise_multiplier: Option<f64>,
    delta: f64,
    sampling_prob: f64,
    num_samples: usize,
    batch_size: usize,
) {
    let has_eps = epsilon.is_some();
    let has_nm = noise_multiplier.is_some();
    assert!(
        has_eps ^ has_nm,
        "exactly one of epsilon or noise_multiplier must be set"
    );
    assert!(delta > 0.0 && delta < 1.0, "delta must be in (0, 1)");
    assert!(batch_size >= 1, "batch_size must be >= 1");
    assert!(
        num_samples >= batch_size,
        "num_samples must be >= batch_size"
    );
    assert!(
        (0.0..=1.0).contains(&sampling_prob),
        "sampling_prob must be in [0, 1]"
    );
}

fn div_ceil(x: usize, y: usize) -> usize {
    if y == 0 {
        return x;
    }
    (x + y - 1) / y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bandmf_plan_builds_with_noise_multiplier() {
        let cfg = BandMfExecutionPlanConfig {
            iterations: 4,
            num_bands: 2,
            epsilon: None,
            delta: 1e-6,
            noise_multiplier: Some(1.2),
            sampling_prob: 0.5,
            num_samples: 1000,
            batch_size: 64,
            l2_clip_norm: 1.0,
            rescale_to_unit_norm: false,
            normalize_by: None,
            truncated_batch_size: None,
            partition_type: PartitionType::EqualSplit,
            neighboring_relation: NeighboringRelation::ReplaceSpecial,
            strategy_optimization_steps: 25,
            noise_seed: 7,
            noise_strategy: NoiseStrategy::Default,
        };
        let plan = cfg.build();
        let eps = plan.epsilon(cfg.delta);
        assert!(eps.is_finite());
        assert!(eps > 0.0);
    }

    #[test]
    fn dpsgd_plan_builds_with_noise_multiplier() {
        let cfg = DpsgdExecutionPlanConfig {
            iterations: 8,
            epsilon: None,
            delta: 1e-6,
            noise_multiplier: Some(1.0),
            sampling_prob: 0.2,
            num_samples: 512,
            batch_size: 64,
            l2_clip_norm: 1.0,
            rescale_to_unit_norm: false,
            normalize_by: None,
            truncated_batch_size: None,
            partition_type: PartitionType::Independent,
            neighboring_relation: NeighboringRelation::ReplaceSpecial,
            noise_seed: 1,
            noise_strategy: NoiseStrategy::Default,
        };
        let plan = cfg.build();
        let eps = plan.epsilon(cfg.delta);
        assert!(eps.is_finite());
        assert!(eps > 0.0);
    }
}
