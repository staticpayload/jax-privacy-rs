//! Basic structural checks for matrix-factorization strategies.

use ndarray::{Array1, Array2};

/// Returns true if all entries strictly above the diagonal are near zero.
pub fn is_lower_triangular(mat: &Array2<f64>, tol: f64) -> bool {
    let n = mat.nrows();
    if n != mat.ncols() {
        return false;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if mat[(i, j)].abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Panics if the matrix is not lower triangular within tolerance.
pub fn assert_lower_triangular(mat: &Array2<f64>, tol: f64) {
    assert!(
        is_lower_triangular(mat, tol),
        "matrix is not lower triangular"
    );
}

/// Returns true if the matrix is lower-triangular Toeplitz within tolerance.
pub fn is_lower_toeplitz(mat: &Array2<f64>, tol: f64) -> bool {
    let n = mat.nrows();
    if n != mat.ncols() {
        return false;
    }
    for i in 0..n {
        for j in 0..=i {
            if i == 0 || j == 0 {
                continue;
            }
            let a = mat[(i, j)];
            let b = mat[(i - 1, j - 1)];
            if (a - b).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Compute per-column L2 norms.
pub fn column_norms(mat: &Array2<f64>) -> Array1<f64> {
    let n = mat.ncols();
    let mut norms = Array1::zeros(n);
    for j in 0..n {
        let mut acc = 0.0;
        for i in 0..mat.nrows() {
            acc += mat[(i, j)] * mat[(i, j)];
        }
        norms[j] = acc.sqrt();
    }
    norms
}

/// Normalize each column in-place, skipping zero-norm columns.
pub fn normalize_columns(mat: &mut Array2<f64>) -> Array1<f64> {
    let norms = column_norms(mat);
    for j in 0..mat.ncols() {
        let norm = norms[j];
        if norm > 0.0 {
            for i in 0..mat.nrows() {
                mat[(i, j)] /= norm;
            }
        }
    }
    norms
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn lower_triangular_check() {
        let good = array![[1.0, 0.0], [2.0, 3.0]];
        assert!(is_lower_triangular(&good, 1e-12));

        let bad = array![[1.0, 1.0], [0.0, 1.0]];
        assert!(!is_lower_triangular(&bad, 1e-12));
    }

    #[test]
    fn lower_toeplitz_check() {
        let toeplitz = array![[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 2.0, 1.0]];
        assert!(is_lower_toeplitz(&toeplitz, 1e-12));
    }
}
