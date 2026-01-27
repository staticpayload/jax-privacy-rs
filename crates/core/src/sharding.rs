//! Sharding-inspired helpers.
//!
//! The original JAX implementation uses explicit sharding metadata to support
//! multi-host "zero redundancy" noise generation. In this Rust rewrite, we
//! provide single-process analogues that preserve the shape semantics without
//! requiring distributed infrastructure.

use ndarray::{Array, IxDyn};

use crate::sampling::compute_early_stopping_order as compute_early_stopping_order_impl;
use crate::tensor::{Scalar, Tensor};

/// Return the smallest multiple of `multiple` that is >= `size`.
pub fn ceiling_to_multiple(size: usize, multiple: usize) -> usize {
    if multiple == 0 {
        return size;
    }
    let remainder = size % multiple;
    if remainder == 0 {
        size
    } else {
        size + multiple - remainder
    }
}

/// Flatten a tensor and pad with zeros up to a multiple.
pub fn flatten_with_zero_redundancy(x: &Tensor, multiple: usize) -> Tensor {
    let flat: Vec<Scalar> = x.iter().copied().collect();
    let padded_len = ceiling_to_multiple(flat.len(), multiple.max(1));
    let mut padded = Vec::with_capacity(padded_len);
    padded.extend(flat);
    padded.resize(padded_len, 0.0 as Scalar);
    Array::from_shape_vec(IxDyn(&[padded_len]), padded)
        .unwrap_or_else(|e| panic!("failed to flatten/pad tensor: {e}"))
}

/// Add flattened noise to a tensor by reshaping the leading slice.
pub fn local_reshape_add(x: &Tensor, noise_flat: &Tensor) -> Tensor {
    let len = x.len();
    assert!(
        noise_flat.len() >= len,
        "noise must have at least as many elements as x"
    );
    let noise_vec: Vec<Scalar> = noise_flat.iter().copied().take(len).collect();
    let reshaped = Array::from_shape_vec(x.raw_dim(), noise_vec)
        .unwrap_or_else(|e| panic!("failed to reshape noise: {e}"));
    x + &reshaped
}

/// Compute the early-stopping order for microbatching.
pub fn compute_early_stopping_order(
    batch_size: usize,
    microbatch_size: Option<usize>,
) -> Vec<usize> {
    compute_early_stopping_order_impl(batch_size, microbatch_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_and_reshape_round_trip() {
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let flat = flatten_with_zero_redundancy(&x, 4);
        let out = local_reshape_add(&x, &flat);
        assert_eq!(out.shape(), x.shape());
    }
}
