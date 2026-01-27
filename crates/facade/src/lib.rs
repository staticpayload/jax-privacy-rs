//! Facade crate re-exporting stable APIs.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod experimental;
pub mod keras;
pub mod training;

/// Crate version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use jax_privacy_accounting as accounting;
pub use jax_privacy_audit as audit;
pub use jax_privacy_core as core;
pub use jax_privacy_mf as mf;
pub use jax_privacy_prng as prng;

pub use accounting::{
    amplified_bandmf_event, calibrate_batch_size, calibrate_dp_mechanism,
    calibrate_dp_mechanism_pld, calibrate_noise_multiplier, calibrate_num_updates, dpsgd_event,
    optimal_physical_batch_sizes, truncated_amplified_bandmf_event, truncated_dpsgd_event,
    BatchingScaleSchedule, CachedExperimentAccountant, DpAccountantConfig, DpEvent, DpParams,
    DpTrainingAccountant, DpsgdTrainingAccountant, DpsgdTrainingUserLevelAccountant, PldAccountant,
    PldAccountantConfig, RdpAccountant, RdpAccountantConfig, Sampler, Schedule,
    SingleReleaseTrainingAccountant,
};
pub use audit::{
    epsilon_one_run, epsilon_one_run_fdp, one_run_p_value, sigma_for_gaussian_eps_delta,
    synthetic_audit_data, AuditResult, Auditor, CanaryAuditResult, CanaryScoreAuditor,
    ThresholdStrategy,
};
pub use core::clipping::clipped_grad;
pub use core::prelude as core_prelude;
pub use experimental::{
    BandMfExecutionPlanConfig, DpExecutionPlan, DpExecutionPlanState, DpsgdExecutionPlanConfig,
};
pub use keras::{calculate_optimizer_steps_to_perform, DpKerasConfig};
pub use mf::{
    banded_minsep_sensitivity_squared, gaussian_privatizer, inverse_as_streaming_matrix,
    matrix_factorization_privatizer, matrix_factorization_privatizer_dense, max_error_blt,
    max_loss_blt, mean_error, mean_error_blt, min_buf_decay_gap, minsep_sensitivity_squared_blt,
    multiply_streaming_matrices, optimal_max_error_strategy_coefs, optimize,
    optimize_banded_toeplitz, optimize_projected, per_query_error, sensitivity_squared_blt,
    BufferedStreamingMatrixBuilder, BufferedToeplitz, CallbackArgs, ColumnNormalizedBanded,
    Diagonal, Identity, MatrixFactorizationPrivatizer, NoiseStrategy, PrefixSum, StreamingMatrix,
    Toeplitz, ToeplitzInverseStream,
};
pub use prng::JaxKey;
pub use training::{
    dpsgd_aggregate_batch, gather_batch_with_padding, reorder_indices_for_microbatch, BatchData,
    DpSgdRunner, IndexedDataset,
};

/// Convenience prelude covering common DP building blocks.
pub mod prelude {
    pub use crate::experimental::{
        BandMfExecutionPlanConfig, DpExecutionPlan, DpExecutionPlanState, DpsgdExecutionPlanConfig,
    };
    pub use crate::keras::DpKerasConfig;
    pub use crate::training::{
        dpsgd_aggregate_batch, gather_batch_with_padding, reorder_indices_for_microbatch,
        BatchData, DpSgdRunner, IndexedDataset,
    };
    pub use jax_privacy_accounting::prelude::*;
    pub use jax_privacy_audit::prelude::*;
    pub use jax_privacy_core::prelude::*;
    pub use jax_privacy_mf::prelude::*;
    pub use jax_privacy_prng::prelude::*;
}
