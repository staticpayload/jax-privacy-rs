//! Banded strategy representations.

use ndarray::{Array1, Array2};

use crate::checks::normalize_columns;
use crate::dense::inverse_lower_triangular;
use crate::optimization::{optimize_projected, CallbackArgs};
use crate::sensitivity;
use crate::streaming::{multiply_streaming_matrices, PrefixSum, StreamingMatrix};

/// A column-normalized banded lower-triangular matrix.
#[derive(Clone, Debug)]
pub struct ColumnNormalizedBanded {
    /// Parameter matrix of shape (n, bands).
    pub params: Array2<f64>,
}

impl ColumnNormalizedBanded {
    /// Number of rows/columns in the implied square matrix.
    pub fn n(&self) -> usize {
        self.params.nrows()
    }

    /// Number of bands in the strategy.
    pub fn bands(&self) -> usize {
        self.params.ncols()
    }

    /// Construct from banded Toeplitz coefficients.
    pub fn from_banded_toeplitz(n: usize, coefs: &[f64]) -> Self {
        let bands = coefs.len();
        assert!(bands >= 1 && bands <= n, "bands must be in [1, n]");

        let norm = coefs.iter().map(|v| v * v).sum::<f64>().sqrt();
        let scale = if norm > 0.0 { 1.0 / norm } else { 1.0 };

        let mut params = Array2::zeros((n, bands));
        for i in 0..n {
            for b in 0..bands {
                // Mirror the lower-right triangle masking used in the JAX code.
                if i + b < n {
                    params[(i, b)] = coefs[b] * scale;
                }
            }
        }
        Self { params }
    }

    /// Default initialization inspired by Fichtenberger et al. (2023).
    pub fn default(n: usize, bands: usize) -> Self {
        assert!(bands >= 1 && bands <= n, "bands must be in [1, n]");
        let mut coefs = Vec::with_capacity(bands);
        let mut acc = 1.0;
        for k in 0..bands {
            if k == 0 {
                acc = 1.0;
            } else {
                let num = (2 * k - 1) as f64;
                let den = (2 * k) as f64;
                acc *= num / den;
            }
            coefs.push(acc);
        }
        Self::from_banded_toeplitz(n, &coefs)
    }

    /// Materialize as a dense column-normalized matrix.
    pub fn materialize(&self) -> Array2<f64> {
        let n = self.n();
        let bands = self.bands();
        let mut mat = Array2::zeros((n, n));
        for i in 0..n {
            for b in 0..bands {
                if i + b >= n || i < b {
                    continue;
                }
                let col = i - b;
                mat[(i, col)] = self.params[(i, b)];
            }
        }
        normalize_columns(&mut mat);
        mat
    }

    /// Convert the inverse to a streaming matrix via dense inversion.
    pub fn inverse_as_streaming_matrix(&self) -> DenseStreamingMatrix {
        let mat = self.materialize();
        let inv = inverse_lower_triangular(&mat);
        DenseStreamingMatrix { mat: inv }
    }

    /// Toeplitz-like diagonal coefficients for debugging/inspection.
    pub fn diagonal_coefs(&self) -> Array1<f64> {
        let n = self.n();
        let mut out = Array1::zeros(n);
        for i in 0..n {
            out[i] = self.params[(i, 0)];
        }
        out
    }
}

/// Sensitivity squared under min-separation constraints.
pub fn minsep_sensitivity_squared(
    strategy: &ColumnNormalizedBanded,
    min_sep: usize,
    max_participations: Option<usize>,
    n: Option<usize>,
) -> usize {
    let bands = strategy.bands();
    let n = n.unwrap_or(strategy.n());
    let max_participations =
        sensitivity::minsep_true_max_participations(n, min_sep, max_participations);
    assert!(
        min_sep >= bands,
        "min_sep must be >= bands for banded sensitivity upper bound"
    );
    max_participations
}

/// Expected per-query squared error for a column-normalized banded strategy.
pub fn per_query_error(strategy: &ColumnNormalizedBanded) -> Array1<f64> {
    per_query_error_with(strategy, PrefixSum)
}

/// Expected per-query squared error for a custom workload.
pub fn per_query_error_with<A: StreamingMatrix>(
    strategy: &ColumnNormalizedBanded,
    workload: A,
) -> Array1<f64> {
    let inv = strategy.inverse_as_streaming_matrix();
    let composed = multiply_streaming_matrices(workload, inv);
    composed.row_norms_sq(strategy.n())
}

/// Mean per-query squared error for the prefix workload.
pub fn mean_error(strategy: &ColumnNormalizedBanded) -> f64 {
    let errs = per_query_error(strategy);
    if errs.is_empty() {
        0.0
    } else {
        errs.iter().sum::<f64>() / errs.len() as f64
    }
}

/// Max per-query squared error for the prefix workload.
pub fn max_error(strategy: &ColumnNormalizedBanded) -> f64 {
    let errs = per_query_error(strategy);
    errs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Last-iterate per-query squared error for the prefix workload.
pub fn last_error(strategy: &ColumnNormalizedBanded) -> f64 {
    let errs = per_query_error(strategy);
    errs.last().copied().unwrap_or(0.0)
}

/// Optimize a banded strategy using finite-difference gradient descent.
pub fn optimize(
    n: usize,
    bands: usize,
    init: Option<ColumnNormalizedBanded>,
    max_optimizer_steps: usize,
    step_size: f64,
    reduction_fn: impl Fn(&Array1<f64>) -> f64,
    mut callback: impl FnMut(CallbackArgs<'_>) -> bool,
) -> ColumnNormalizedBanded {
    let init = init.unwrap_or_else(|| ColumnNormalizedBanded::default(n, bands));
    let params = init.params.clone().into_raw_vec();

    let loss_fn = |p: &[f64]| {
        let mat = Array2::from_shape_vec((n, bands), p.to_vec())
            .unwrap_or_else(|_| Array2::zeros((n, bands)));
        let candidate = ColumnNormalizedBanded { params: mat };
        reduction_fn(&per_query_error(&candidate))
    };

    let optimized = optimize_projected(
        loss_fn,
        params,
        max_optimizer_steps,
        step_size,
        |_| {},
        |args| callback(args),
    );

    let params =
        Array2::from_shape_vec((n, bands), optimized).unwrap_or_else(|_| Array2::zeros((n, bands)));
    ColumnNormalizedBanded { params }
}

/// A dense streaming matrix backed by a materialized lower-triangular matrix.
#[derive(Clone, Debug)]
pub struct DenseStreamingMatrix {
    mat: Array2<f64>,
}

impl DenseStreamingMatrix {
    /// Create a dense streaming matrix from a materialized lower-triangular matrix.
    pub fn from_dense(mat: Array2<f64>) -> Self {
        Self { mat }
    }
}

/// Streaming state that records all previously seen inputs.
#[derive(Clone, Debug, Default)]
pub struct DenseState {
    index: usize,
    history: Vec<f64>,
}

impl StreamingMatrix for DenseStreamingMatrix {
    type State = DenseState;

    fn init(&self, n: usize) -> Self::State {
        DenseState {
            index: 0,
            history: Vec::with_capacity(n),
        }
    }

    fn next(&self, x: f64, mut state: Self::State) -> (f64, Self::State) {
        let i = state.index;
        state.history.push(x);

        let mut y = 0.0;
        for j in 0..=i {
            y += self.mat[(i, j)] * state.history[j];
        }
        state.index = state.index.saturating_add(1);
        (y, state)
    }

    fn get_coeff(&self, row: usize, col: usize) -> f64 {
        if col > row {
            0.0
        } else {
            self.mat[(row, col)]
        }
    }

    fn materialize(&self, _n: usize) -> Array2<f64> {
        self.mat.clone()
    }

    fn row_norms_sq(&self, _n: usize) -> Array1<f64> {
        let n = self.mat.nrows();
        let mut out = Array1::zeros(n);
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..=i {
                let v = self.mat[(i, j)];
                acc += v * v;
            }
            out[i] = acc;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banded_materialize_is_lower_triangular() {
        let strategy = ColumnNormalizedBanded::default(5, 3);
        let mat = strategy.materialize();
        for i in 0..5 {
            for j in (i + 1)..5 {
                assert_eq!(mat[(i, j)], 0.0);
            }
        }
    }
}
