//! Correlated noise addition via matrix factorization.
//!
//! This mirrors the spirit of `noise_addition.py` by generating noise whose
//! covariance is defined by a (typically lower-triangular) noising matrix.

use ndarray::{Array, Array2, Dimension, IxDyn};
use rand_distr::{Distribution, Normal};

use jax_privacy_core::pytree::PyTree;
use jax_privacy_core::sharding::ceiling_to_multiple;
use jax_privacy_core::tensor::{Scalar, Tensor};
use jax_privacy_core::transform::GradientTransform;
use jax_privacy_prng::JaxKey;

use crate::banded::DenseStreamingMatrix;
use crate::checks::assert_lower_triangular;
use crate::streaming::{Identity, StreamingMatrix};

/// Strategy for intermediate noise structure.
#[derive(Clone, Copy, Debug)]
pub enum NoiseStrategy {
    /// Generate noise with the same shape as the value.
    Default,
    /// Generate flattened, padded noise and locally reshape it back.
    ZeroRedundancy {
        /// Pad the flattened noise length up to a multiple of this value.
        multiple: usize,
    },
}

impl Default for NoiseStrategy {
    fn default() -> Self {
        Self::Default
    }
}

/// A correlated-noise privatizer driven by a noising matrix.
#[derive(Clone, Debug)]
pub struct MatrixFactorizationPrivatizer<M> {
    /// Noising matrix `A` such that covariance is `A^T A`.
    pub matrix: M,
    /// Standard deviation multiplier applied to the generated noise.
    pub stddev: f64,
    /// Root PRNG key.
    pub key: JaxKey,
    /// Strategy for intermediate noise generation.
    pub strategy: NoiseStrategy,
}

impl<M> MatrixFactorizationPrivatizer<M> {
    /// Construct a new privatizer.
    pub fn new(matrix: M, stddev: f64, key: JaxKey) -> Self {
        Self {
            matrix,
            stddev,
            key,
            strategy: NoiseStrategy::Default,
        }
    }

    /// Configure the intermediate noise strategy.
    pub fn with_strategy(mut self, strategy: NoiseStrategy) -> Self {
        self.strategy = strategy;
        self
    }
}

/// Construct a matrix-factorization privatizer with a chosen strategy.
pub fn matrix_factorization_privatizer<M: StreamingMatrix>(
    matrix: M,
    stddev: f64,
    key: JaxKey,
    strategy: NoiseStrategy,
) -> MatrixFactorizationPrivatizer<M> {
    MatrixFactorizationPrivatizer::new(matrix, stddev, key).with_strategy(strategy)
}

/// Construct an isotropic Gaussian privatizer.
pub fn gaussian_privatizer(
    stddev: f64,
    key: JaxKey,
    strategy: NoiseStrategy,
) -> MatrixFactorizationPrivatizer<Identity> {
    matrix_factorization_privatizer(Identity, stddev, key, strategy)
}

/// Construct a privatizer from a dense lower-triangular matrix.
pub fn matrix_factorization_privatizer_dense(
    matrix: Array2<f64>,
    stddev: f64,
    key: JaxKey,
    strategy: NoiseStrategy,
) -> MatrixFactorizationPrivatizer<DenseStreamingMatrix> {
    assert_lower_triangular(&matrix, 1e-10);
    let dense = DenseStreamingMatrix::from_dense(matrix);
    matrix_factorization_privatizer(dense, stddev, key, strategy)
}

/// State for the correlated-noise privatizer.
#[derive(Clone, Debug)]
pub struct MfNoiseState {
    key: JaxKey,
    step: u64,
}

impl<T, M> GradientTransform<T> for MatrixFactorizationPrivatizer<M>
where
    T: PyTree<Leaf = Tensor>,
    M: StreamingMatrix,
{
    type State = MfNoiseState;

    fn init(&self, _model: &T) -> Self::State {
        MfNoiseState {
            key: self.key,
            step: 0,
        }
    }

    fn update(&self, grad: &T, mut state: Self::State) -> (T, Self::State) {
        let row = state.step as usize;
        let row_coefs = matrix_row(&self.matrix, row);

        let (leaves, spec) = grad.flatten();
        let mut out_leaves = Vec::with_capacity(leaves.len());
        for (leaf_idx, leaf) in leaves.iter().enumerate() {
            let leaf_key = state.key.fold_in(state.step).fold_in(leaf_idx as u64);
            let noise = correlated_noise(&row_coefs, leaf_key, leaf, self.stddev, self.strategy);
            let mut out = leaf.clone();
            out.zip_mut_with(&noise, |a, b| *a += *b);
            out_leaves.push(out);
        }

        state.step = state.step.saturating_add(1);
        (T::unflatten(&spec, out_leaves), state)
    }
}

fn matrix_row<M: StreamingMatrix>(matrix: &M, row: usize) -> Vec<f64> {
    (0..=row).map(|col| matrix.get_coeff(row, col)).collect()
}

fn correlated_noise(
    row: &[f64],
    key: JaxKey,
    value: &Tensor,
    stddev: f64,
    strategy: NoiseStrategy,
) -> Tensor {
    let (noise_shape, target_len) = match strategy {
        NoiseStrategy::Default => (value.raw_dim(), value.len()),
        NoiseStrategy::ZeroRedundancy { multiple } => {
            let padded = ceiling_to_multiple(value.len(), multiple.max(1));
            (IxDyn(&[padded]), value.len())
        }
    };

    let noise_len = Dimension::size(&noise_shape);
    let mut accum = vec![0.0_f64; noise_len];
    let normal = Normal::new(0.0, 1.0).expect("normal distribution parameters are valid");

    for (idx, &coef) in row.iter().enumerate() {
        if coef == 0.0 {
            continue;
        }
        let subkey = key.fold_in(idx as u64);
        let mut rng = subkey.to_rng();
        for v in &mut accum {
            *v += coef * normal.sample(&mut rng);
        }
    }

    for v in &mut accum {
        *v *= stddev;
    }

    // Reshape depending on the strategy.
    match strategy {
        NoiseStrategy::Default => Array::from_shape_vec(noise_shape, cast_vec(&accum))
            .unwrap_or_else(|e| panic!("noise shape mismatch: {e}")),
        NoiseStrategy::ZeroRedundancy { .. } => {
            let trimmed = accum.into_iter().take(target_len).collect::<Vec<_>>();
            Array::from_shape_vec(value.raw_dim(), cast_vec(&trimmed))
                .unwrap_or_else(|e| panic!("noise reshape mismatch: {e}"))
        }
    }
}

fn cast_vec(values: &[f64]) -> Vec<Scalar> {
    values.iter().map(|&v| v as Scalar).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toeplitz::Toeplitz;

    #[test]
    fn correlated_noise_is_deterministic() {
        let matrix = Toeplitz::identity(3);
        let priv1 = MatrixFactorizationPrivatizer::new(matrix.clone(), 1.0, JaxKey::new(7));
        let priv2 = MatrixFactorizationPrivatizer::new(matrix, 1.0, JaxKey::new(7));

        let grad = ndarray::array![0.0, 0.0].into_dyn();
        let s1 = priv1.init(&grad);
        let s2 = priv2.init(&grad);
        let (g1, _) = priv1.update(&grad, s1);
        let (g2, _) = priv2.update(&grad, s2);
        assert_eq!(g1, g2);
    }

    #[test]
    fn gaussian_privatizer_is_deterministic() {
        let grad = ndarray::array![0.0, 0.0].into_dyn();
        let p1 = gaussian_privatizer(1.0, JaxKey::new(9), NoiseStrategy::Default);
        let p2 = gaussian_privatizer(1.0, JaxKey::new(9), NoiseStrategy::Default);
        let s1 = p1.init(&grad);
        let s2 = p2.init(&grad);
        let (g1, _) = p1.update(&grad, s1);
        let (g2, _) = p2.update(&grad, s2);
        assert_eq!(g1, g2);
    }

    #[test]
    fn dense_identity_matches_gaussian_on_first_step() {
        let grad = ndarray::array![0.0, 0.0].into_dyn();
        let dense = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
        let pd = matrix_factorization_privatizer_dense(
            dense,
            1.0,
            JaxKey::new(123),
            NoiseStrategy::Default,
        );
        let pg = gaussian_privatizer(1.0, JaxKey::new(123), NoiseStrategy::Default);
        let (gd, _) = pd.update(&grad, pd.init(&grad));
        let (gg, _) = pg.update(&grad, pg.init(&grad));
        assert_eq!(gd, gg);
    }
}
