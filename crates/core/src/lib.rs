//! Core differentially private ML primitives.
//!
//! This crate provides framework-agnostic building blocks for DP training:
//! clipping, noise addition, sampling, and PyTree-style utilities.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod batch_selection;
pub mod clipping;
pub mod error;
pub mod noise;
pub mod pytree;
pub mod sampling;
pub mod sharding;
pub mod tensor;
pub mod transform;

pub use batch_selection::{
    split_and_pad_global_batch, split_and_pad_global_batch_matrix, BallsInBinsSampling,
    BatchSelectionStrategy, CyclicPoissonSampling, PartitionType, UserSelectionStrategy,
};
pub use clipping::{
    clip_and_aggregate, clip_and_aggregate_with_padding, clip_pytree, clip_tensor, clipped_grad,
    BoundedSensitivityCallable, ClipReport, ClippedGradOutput, NeighboringRelation,
};
pub use error::{DpError, Result};
pub use noise::{
    add_gaussian_noise, add_gaussian_noise_tree, add_laplace_noise, GaussianMechanism,
    LaplaceMechanism,
};
pub use pytree::{global_l2_norm, map_leaves, PyTree, TreeSpec};
pub use sampling::{
    compute_early_stopping_order, fixed_sample, pad_to_multiple_of, poisson_sample, BatchIndices,
};
pub use sharding::{ceiling_to_multiple, flatten_with_zero_redundancy, local_reshape_add};
pub use tensor::{l2_norm, sanitize, Scalar, Tensor};
pub use transform::{DpSgdAggregator, DpSgdState, GaussianPrivatizer, GradientTransform};

/// Common imports for downstream users.
pub mod prelude {
    pub use crate::{
        add_gaussian_noise, add_gaussian_noise_tree, add_laplace_noise, ceiling_to_multiple,
        clip_and_aggregate, clip_and_aggregate_with_padding, clip_pytree, clip_tensor,
        clipped_grad, compute_early_stopping_order, fixed_sample, flatten_with_zero_redundancy,
        global_l2_norm, local_reshape_add, pad_to_multiple_of, poisson_sample,
        split_and_pad_global_batch, split_and_pad_global_batch_matrix, BallsInBinsSampling,
        BatchSelectionStrategy, BoundedSensitivityCallable, ClippedGradOutput,
        CyclicPoissonSampling, DpError, DpSgdAggregator, DpSgdState, GaussianMechanism,
        GaussianPrivatizer, GradientTransform, LaplaceMechanism, NeighboringRelation,
        PartitionType, PyTree, Result, Scalar, Tensor, UserSelectionStrategy,
    };
}
