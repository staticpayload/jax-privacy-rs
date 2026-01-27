//! Gradient clipping for bounded sensitivity.

use crate::pytree::{global_l2_norm, leaf_count, map_leaves, scale_tensor, PyTree};
use crate::tensor::{l2_norm, sanitize, Scalar, Tensor};

/// Result of a clipping operation.
#[derive(Clone, Debug)]
pub struct ClipReport {
    /// Original L2 norm before clipping.
    pub original_norm: f64,
    /// Whether clipping was applied.
    pub clipped: bool,
    /// Scale factor applied (1.0 if not clipped).
    pub scale: f64,
}

/// Clip a tensor to a maximum L2 norm in-place.
pub fn clip_tensor(tensor: &mut Tensor, max_norm: f64) -> ClipReport {
    if !max_norm.is_finite() || max_norm <= 0.0 {
        sanitize(tensor, 0.0 as Scalar, 0.0 as Scalar);
        let original_norm = l2_norm(tensor);
        tensor.fill(0.0 as Scalar);
        return ClipReport {
            original_norm,
            clipped: true,
            scale: 0.0,
        };
    }

    sanitize(tensor, 0.0 as Scalar, 0.0 as Scalar);
    let norm = l2_norm(tensor);

    if !norm.is_finite() || norm == 0.0 {
        tensor.fill(0.0 as Scalar);
        return ClipReport {
            original_norm: norm,
            clipped: true,
            scale: 0.0,
        };
    }

    if norm <= max_norm {
        return ClipReport {
            original_norm: norm,
            clipped: false,
            scale: 1.0,
        };
    }

    let scale = max_norm / norm;
    tensor.mapv_inplace(|x| x * (scale as Scalar));
    ClipReport {
        original_norm: norm,
        clipped: true,
        scale,
    }
}

/// Clip a PyTree of tensors based on its global L2 norm.
///
/// This mirrors the key semantics of `jax_privacy.clipping.clip_pytree`.
pub fn clip_pytree<T>(
    pytree: &T,
    clip_norm: f64,
    rescale_to_unit_norm: bool,
    nan_safe: bool,
    return_zero: bool,
) -> (T, f64)
where
    T: PyTree<Leaf = Tensor>,
{
    if clip_norm.is_finite() && clip_norm < 0.0 {
        panic!("clip_norm must be non-negative, got {clip_norm}");
    }

    let sanitized = if nan_safe {
        map_leaves(pytree, |leaf| {
            let mut leaf = leaf.clone();
            sanitize(&mut leaf, 0.0 as Scalar, 0.0 as Scalar);
            leaf
        })
    } else {
        map_leaves(pytree, |leaf| leaf.clone())
    };

    let clip_norm = if clip_norm.is_finite() {
        clip_norm.max(0.0)
    } else {
        clip_norm
    };
    let l2 = global_l2_norm(&sanitized);

    let mut scale = if l2 > 0.0 && l2.is_finite() {
        (clip_norm / l2).min(1.0)
    } else {
        0.0
    };

    if rescale_to_unit_norm {
        scale = if clip_norm > 0.0 {
            scale / clip_norm
        } else if l2 > 0.0 {
            1.0 / l2
        } else {
            0.0
        };
    }

    if !scale.is_finite() {
        scale = 0.0;
    }

    let clipped = map_leaves(&sanitized, |leaf| scale_tensor(leaf, scale));
    let final_tree = if return_zero {
        map_leaves(&clipped, |leaf| Tensor::zeros(leaf.raw_dim()))
    } else {
        clipped
    };

    (final_tree, l2 as f64)
}

/// Neighboring relation used for sensitivity calculations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeighboringRelation {
    /// Add or remove a single element.
    AddOrRemoveOne,
    /// Replace a single element.
    ReplaceOne,
    /// Replace-one style relation with special handling.
    ReplaceSpecial,
}

/// A callable with an explicit L2 sensitivity bound.
#[derive(Clone)]
pub struct BoundedSensitivityCallable<F> {
    /// The wrapped callable.
    pub fun: F,
    /// L2 norm bound.
    pub l2_norm_bound: f64,
    /// Whether the callable returns auxiliary outputs.
    pub has_aux: bool,
}

/// Output of a clipped-gradient aggregation.
#[derive(Clone, Debug)]
pub struct ClippedGradOutput<T> {
    /// Aggregated clipped gradients.
    pub clipped: T,
    /// Optional per-example norms.
    pub norms: Option<Vec<f64>>,
}

impl<F> BoundedSensitivityCallable<F> {
    /// Create a new bounded-sensitivity callable wrapper.
    pub fn new(fun: F, l2_norm_bound: f64, has_aux: bool) -> Self {
        Self {
            fun,
            l2_norm_bound,
            has_aux,
        }
    }

    /// Sensitivity under a neighboring relation.
    pub fn sensitivity(&self, relation: NeighboringRelation) -> f64 {
        match relation {
            NeighboringRelation::AddOrRemoveOne => self.l2_norm_bound,
            NeighboringRelation::ReplaceOne => 2.0 * self.l2_norm_bound,
            NeighboringRelation::ReplaceSpecial => self.l2_norm_bound,
        }
    }

    /// Call the wrapped function.
    pub fn call<A, R>(&self, args: A) -> R
    where
        F: Fn(A) -> R,
    {
        (self.fun)(args)
    }
}

fn add_tensors(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    assert_eq!(
        lhs.raw_dim(),
        rhs.raw_dim(),
        "tensor shapes must match for aggregation"
    );
    lhs + rhs
}

fn scale_in_place(t: &Tensor, factor: f64) -> Tensor {
    let f = factor as Scalar;
    t.mapv(|v| v / f)
}

/// Clip and aggregate a slice of per-example trees.
///
/// This is a framework-agnostic analogue of `clipped_fun`, assuming the
/// per-example values are already computed.
pub fn clip_and_aggregate<T>(
    examples: &[T],
    l2_clip_norm: f64,
    rescale_to_unit_norm: bool,
    normalize_by: f64,
    nan_safe: bool,
    return_norms: bool,
) -> (T, Option<Vec<f64>>)
where
    T: PyTree<Leaf = Tensor>,
{
    assert!(!examples.is_empty(), "examples must be non-empty");
    assert!(normalize_by > 0.0, "normalize_by must be positive");

    let (first_leaves, spec) = examples[0].flatten();
    let leaf_count = leaf_count(&spec);
    assert_eq!(
        first_leaves.len(),
        leaf_count,
        "TreeSpec/leaf mismatch in first example"
    );

    let mut aggregated: Vec<Tensor> = first_leaves
        .iter()
        .map(|leaf| Tensor::zeros(leaf.raw_dim()))
        .collect();
    let mut norms = if return_norms {
        Some(Vec::with_capacity(examples.len()))
    } else {
        None
    };

    for ex in examples {
        let (clipped, norm) = clip_pytree(ex, l2_clip_norm, rescale_to_unit_norm, nan_safe, false);
        if let Some(buf) = norms.as_mut() {
            buf.push(norm);
        }
        let (leaves, ex_spec) = clipped.flatten();
        assert_eq!(ex_spec, spec, "all examples must share the same structure");
        for (agg_leaf, leaf) in aggregated.iter_mut().zip(leaves.iter()) {
            *agg_leaf = add_tensors(agg_leaf, leaf);
        }
    }

    if normalize_by != 1.0 {
        for leaf in &mut aggregated {
            *leaf = scale_in_place(leaf, normalize_by);
        }
    }

    (T::unflatten(&spec, aggregated), norms)
}

/// Clip and aggregate with optional padding and microbatch ordering.
pub fn clip_and_aggregate_with_padding<T>(
    examples: &[T],
    l2_clip_norm: f64,
    rescale_to_unit_norm: bool,
    normalize_by: f64,
    nan_safe: bool,
    return_norms: bool,
    is_padding_example: Option<&[bool]>,
    microbatch_size: Option<usize>,
) -> (T, Option<Vec<f64>>)
where
    T: PyTree<Leaf = Tensor>,
{
    assert!(!examples.is_empty(), "examples must be non-empty");
    assert!(normalize_by > 0.0, "normalize_by must be positive");

    let len = examples.len();
    if let Some(mask) = is_padding_example {
        assert_eq!(mask.len(), len, "is_padding_example length mismatch");
    }

    let order: Vec<usize> = if microbatch_size.is_some() && is_padding_example.is_some() {
        crate::sampling::compute_early_stopping_order(len, microbatch_size)
    } else {
        (0..len).collect()
    };

    let first_idx = order
        .iter()
        .copied()
        .find(|&i| !is_padding_example.map(|m| m[i]).unwrap_or(false))
        .unwrap_or(0);

    let (first_leaves, spec) = examples[first_idx].flatten();
    let leaf_count = leaf_count(&spec);
    assert_eq!(
        first_leaves.len(),
        leaf_count,
        "TreeSpec/leaf mismatch in first example"
    );

    let mut aggregated: Vec<Tensor> = first_leaves
        .iter()
        .map(|leaf| Tensor::zeros(leaf.raw_dim()))
        .collect();

    let mut norms = if return_norms {
        Some(vec![0.0; len])
    } else {
        None
    };

    for &idx in &order {
        let is_pad = is_padding_example.map(|m| m[idx]).unwrap_or(false);
        if is_pad {
            continue;
        }
        let (clipped, norm) = clip_pytree(
            &examples[idx],
            l2_clip_norm,
            rescale_to_unit_norm,
            nan_safe,
            false,
        );
        if let Some(buf) = norms.as_mut() {
            buf[idx] = norm;
        }
        let (leaves, ex_spec) = clipped.flatten();
        assert_eq!(ex_spec, spec, "all examples must share the same structure");
        for (agg_leaf, leaf) in aggregated.iter_mut().zip(leaves.iter()) {
            *agg_leaf = add_tensors(agg_leaf, leaf);
        }
    }

    if normalize_by != 1.0 {
        for leaf in &mut aggregated {
            *leaf = scale_in_place(leaf, normalize_by);
        }
    }

    (T::unflatten(&spec, aggregated), norms)
}

/// Build a clipped-gradient aggregation callable over per-example gradients.
///
/// This is a Rust-native analogue of `clipped_grad` that expects per-example
/// gradients as input. The returned callable consumes a vector of per-example
/// gradients, clips them, and aggregates to a single gradient tensor tree.
pub fn clipped_grad<T>(
    l2_clip_norm: f64,
    rescale_to_unit_norm: bool,
    normalize_by: f64,
    nan_safe: bool,
    return_norms: bool,
) -> BoundedSensitivityCallable<Box<dyn Fn(Vec<T>) -> ClippedGradOutput<T> + Send + Sync>>
where
    T: PyTree<Leaf = Tensor> + Clone + Send + Sync + 'static,
{
    assert!(normalize_by > 0.0, "normalize_by must be positive");
    let norm_bound = if rescale_to_unit_norm {
        1.0
    } else {
        l2_clip_norm
    } / normalize_by;

    let fun = move |examples: Vec<T>| {
        let (clipped, norms) = clip_and_aggregate(
            &examples,
            l2_clip_norm,
            rescale_to_unit_norm,
            normalize_by,
            nan_safe,
            return_norms,
        );
        ClippedGradOutput { clipped, norms }
    };

    BoundedSensitivityCallable::new(Box::new(fun), norm_bound, return_norms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use proptest::prelude::*;

    #[test]
    fn test_no_clipping_needed() {
        let tree = array![0.3, 0.4].into_dyn(); // norm = 0.5
        let (clipped, norm) = clip_pytree(&tree, 1.0, false, true, false);
        assert!((norm - 0.5).abs() < 1e-10);
        assert_eq!(clipped, tree);
    }

    #[test]
    fn test_clipping_applied() {
        let tree = array![3.0, 4.0].into_dyn(); // norm = 5.0
        let (clipped, norm) = clip_pytree(&tree, 1.0, false, true, false);
        assert!((norm - 5.0).abs() < 1e-10);
        let clipped_norm = l2_norm(&clipped);
        assert!((clipped_norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rescale_to_unit_norm() {
        let tree = array![3.0, 4.0].into_dyn(); // norm = 5.0
        let (clipped, _) = clip_pytree(&tree, 2.0, true, true, false);
        let clipped_norm = l2_norm(&clipped);
        assert!(clipped_norm <= 1.0 + 1e-10);
    }

    #[test]
    fn test_return_zero() {
        let tree = array![1.0, 2.0, 3.0].into_dyn();
        let (clipped, _) = clip_pytree(&tree, 1.0, false, true, true);
        assert!(clipped.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_clip_and_aggregate() {
        let ex1 = array![3.0, 4.0].into_dyn();
        let ex2 = array![0.6, 0.8].into_dyn();
        let (agg, norms) = clip_and_aggregate(&[ex1, ex2], 1.0, false, 1.0, true, true);
        let norms = norms.expect("norms requested");
        assert_eq!(norms.len(), 2);
        assert!(l2_norm(&agg) <= 2.0 + 1e-10);
    }

    #[test]
    fn test_clip_and_aggregate_with_padding_skips_examples() {
        let a = array![1.0, 0.0].into_dyn();
        let b = array![0.0, 1.0].into_dyn();
        let c = array![2.0, 2.0].into_dyn();
        let examples = vec![a.clone(), b.clone(), c.clone()];
        let padding = vec![false, true, false];

        let (agg, norms) = clip_and_aggregate_with_padding(
            &examples,
            10.0,
            false,
            1.0,
            true,
            true,
            Some(&padding),
            Some(1),
        );
        let expected = &a + &c;
        assert_eq!(agg, expected);
        let norms = norms.expect("norms requested");
        assert_eq!(norms.len(), examples.len());
        assert_eq!(norms[1], 0.0);
    }

    proptest! {
        #[test]
        fn prop_clip_tensor_respects_bound(
            vals in prop::collection::vec(-100.0f64..100.0, 1..32),
            clip_norm in 0.1f64..50.0,
        ) {
            let t = ndarray::Array1::from_vec(vals).into_dyn();
            let (clipped, _) = clip_pytree(&t, clip_norm, false, true, false);
            let norm = l2_norm(&clipped);
            prop_assert!(norm.is_finite());
            prop_assert!(norm <= clip_norm + 1e-6);
        }
    }
}
