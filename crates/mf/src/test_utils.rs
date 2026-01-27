//! Test utilities for matrix-factorization strategies.

use ndarray::{Array1, Array2};

use crate::streaming::StreamingMatrix;

/// Example decreasing Toeplitz coefficients with unit diagonal.
pub fn example_toeplitz_coefs(bands: usize) -> Vec<f64> {
    let bands = bands.max(1);
    let mut coefs = Vec::with_capacity(bands);
    coefs.push(1.0);
    for k in 1..bands {
        coefs.push(1.0 / (k as f64 + 1.0));
    }
    coefs
}

/// Materialize a streaming matrix and return it as dense.
pub fn dense_from_streaming<M: StreamingMatrix>(m: &M, n: usize) -> Array2<f64> {
    m.materialize(n)
}

/// Compute relative L2 error between two vectors.
pub fn relative_l2_error(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..a.len() {
        let da = a[i] - b[i];
        num += da * da;
        den += a[i] * a[i];
    }
    if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}
