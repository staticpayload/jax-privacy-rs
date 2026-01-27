//! PyTree-style abstractions and utilities.
//!
//! This is a lightweight, trait-based analogue of JAX pytrees. It is
//! intentionally minimal but sufficient for tree-wise DP operations.

use crate::tensor::{l2_norm, Scalar, Tensor};

/// Structural description of a [`PyTree`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TreeSpec {
    /// A single leaf value.
    Leaf,
    /// A homogeneous vector of children.
    Vec {
        /// Number of elements.
        len: usize,
        /// Child structure.
        child: Box<TreeSpec>,
    },
    /// A 2-tuple of potentially distinct structures.
    Tuple2(Box<TreeSpec>, Box<TreeSpec>),
    /// An optional child.
    Option {
        /// Whether the option is `Some`.
        is_some: bool,
        /// Structure of the wrapped value when `is_some`.
        child: Box<TreeSpec>,
    },
}

/// A trait representing tree-structured collections of tensors.
///
/// This mirrors the essential flatten/unflatten behavior of JAX pytrees.
pub trait PyTree: Sized {
    /// Leaf type stored in the tree.
    type Leaf;

    /// Flatten into leaves plus a structural specification.
    fn flatten(&self) -> (Vec<Self::Leaf>, TreeSpec);

    /// Reconstruct from a spec and leaves.
    ///
    /// Implementations may panic if the leaves do not match the spec.
    fn unflatten(spec: &TreeSpec, leaves: Vec<Self::Leaf>) -> Self;
}

impl PyTree for Tensor {
    type Leaf = Tensor;

    fn flatten(&self) -> (Vec<Self::Leaf>, TreeSpec) {
        (vec![self.clone()], TreeSpec::Leaf)
    }

    fn unflatten(spec: &TreeSpec, mut leaves: Vec<Self::Leaf>) -> Self {
        match spec {
            TreeSpec::Leaf => leaves
                .pop()
                .unwrap_or_else(|| panic!("missing leaf for Tensor unflatten")),
            _ => panic!("TreeSpec mismatch for Tensor leaf"),
        }
    }
}

impl<T> PyTree for Vec<T>
where
    T: PyTree<Leaf = Tensor>,
{
    type Leaf = Tensor;

    fn flatten(&self) -> (Vec<Self::Leaf>, TreeSpec) {
        if self.is_empty() {
            return (
                Vec::new(),
                TreeSpec::Vec {
                    len: 0,
                    child: Box::new(TreeSpec::Leaf),
                },
            );
        }

        let mut leaves = Vec::new();
        let mut child_spec: Option<TreeSpec> = None;
        for item in self {
            let (mut item_leaves, spec) = item.flatten();
            if let Some(prev) = &child_spec {
                // We assume homogeneous vectors; mismatches are a logic error.
                if prev != &spec {
                    panic!("heterogeneous Vec PyTree is not supported");
                }
            } else {
                child_spec = Some(spec);
            }
            leaves.append(&mut item_leaves);
        }

        (
            leaves,
            TreeSpec::Vec {
                len: self.len(),
                child: Box::new(child_spec.unwrap_or(TreeSpec::Leaf)),
            },
        )
    }

    fn unflatten(spec: &TreeSpec, leaves: Vec<Self::Leaf>) -> Self {
        let (len, child_spec) = match spec {
            TreeSpec::Vec { len, child } => (*len, child.as_ref()),
            _ => panic!("TreeSpec mismatch for Vec"),
        };

        if len == 0 {
            return Vec::new();
        }

        let leaves_per_elem = leaf_count(child_spec);
        assert!(
            leaves_per_elem > 0,
            "invalid TreeSpec: zero leaves per element"
        );

        let expected = len
            .checked_mul(leaves_per_elem)
            .unwrap_or_else(|| panic!("TreeSpec size overflow for Vec unflatten"));
        assert_eq!(
            leaves.len(),
            expected,
            "leaf count does not match Vec TreeSpec"
        );
        let mut out = Vec::with_capacity(len);
        let mut offset = 0;
        for _ in 0..len {
            let end = offset + leaves_per_elem;
            let chunk = leaves[offset..end].to_vec();
            out.push(T::unflatten(child_spec, chunk));
            offset = end;
        }
        out
    }
}

impl<A, B> PyTree for (A, B)
where
    A: PyTree<Leaf = Tensor>,
    B: PyTree<Leaf = Tensor>,
{
    type Leaf = Tensor;

    fn flatten(&self) -> (Vec<Self::Leaf>, TreeSpec) {
        let (mut a_leaves, a_spec) = self.0.flatten();
        let (mut b_leaves, b_spec) = self.1.flatten();
        a_leaves.append(&mut b_leaves);
        (
            a_leaves,
            TreeSpec::Tuple2(Box::new(a_spec), Box::new(b_spec)),
        )
    }

    fn unflatten(spec: &TreeSpec, leaves: Vec<Self::Leaf>) -> Self {
        let (a_spec, b_spec) = match spec {
            TreeSpec::Tuple2(a, b) => (a.as_ref(), b.as_ref()),
            _ => panic!("TreeSpec mismatch for tuple"),
        };

        let a_count = leaf_count(a_spec);
        let (a_leaves, b_leaves) = leaves.split_at(a_count);
        (
            A::unflatten(a_spec, a_leaves.to_vec()),
            B::unflatten(b_spec, b_leaves.to_vec()),
        )
    }
}

impl<T> PyTree for Option<T>
where
    T: PyTree<Leaf = Tensor>,
{
    type Leaf = Tensor;

    fn flatten(&self) -> (Vec<Self::Leaf>, TreeSpec) {
        match self {
            Some(inner) => {
                let (leaves, spec) = inner.flatten();
                (
                    leaves,
                    TreeSpec::Option {
                        is_some: true,
                        child: Box::new(spec),
                    },
                )
            }
            None => (
                Vec::new(),
                TreeSpec::Option {
                    is_some: false,
                    child: Box::new(TreeSpec::Leaf),
                },
            ),
        }
    }

    fn unflatten(spec: &TreeSpec, leaves: Vec<Self::Leaf>) -> Self {
        match spec {
            TreeSpec::Option { is_some, child } => {
                if *is_some {
                    Some(T::unflatten(child, leaves))
                } else {
                    None
                }
            }
            _ => panic!("TreeSpec mismatch for Option"),
        }
    }
}

/// Apply a function to every leaf tensor.
pub fn map_leaves<T>(tree: &T, mut f: impl FnMut(&Tensor) -> Tensor) -> T
where
    T: PyTree<Leaf = Tensor>,
{
    let (leaves, spec) = tree.flatten();
    let mapped = leaves.into_iter().map(|leaf| f(&leaf)).collect();
    T::unflatten(&spec, mapped)
}

/// Compute the global L2 norm across all leaves.
pub fn global_l2_norm<T>(tree: &T) -> f64
where
    T: PyTree<Leaf = Tensor>,
{
    let (leaves, _) = tree.flatten();
    let mut sum_sq = 0.0_f64;
    for leaf in leaves {
        let norm = l2_norm(&leaf);
        if !norm.is_finite() {
            return norm;
        }
        sum_sq += norm * norm;
    }
    sum_sq.sqrt()
}

/// Count the number of leaves described by a spec.
pub fn leaf_count(spec: &TreeSpec) -> usize {
    match spec {
        TreeSpec::Leaf => 1,
        TreeSpec::Vec { len, child } => len.saturating_mul(leaf_count(child)),
        TreeSpec::Tuple2(a, b) => leaf_count(a) + leaf_count(b),
        TreeSpec::Option { is_some, child } => {
            if *is_some {
                leaf_count(child)
            } else {
                0
            }
        }
    }
}

/// Scale a tensor by a floating-point factor, preserving dtype.
pub(crate) fn scale_tensor(t: &Tensor, scale: f64) -> Tensor {
    let s = scale as Scalar;
    t.mapv(|v| v * s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_global_norm_tensor() {
        let t = array![3.0, 4.0].into_dyn();
        assert!((global_l2_norm(&t) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_map_vec_tree() {
        let tree = vec![array![1.0, 2.0].into_dyn(), array![3.0].into_dyn()];
        let doubled = map_leaves(&tree, |t| t.mapv(|v| v * 2.0));
        assert_eq!(doubled[0][[0]], 2.0);
        assert_eq!(doubled[1][[0]], 6.0);
    }
}
