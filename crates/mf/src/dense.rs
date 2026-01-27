//! Dense matrix helpers used by strategy implementations.

use ndarray::{Array1, Array2};

use crate::checks::{assert_lower_triangular, normalize_columns};
use crate::optimization::optimize_projected;
use crate::sensitivity::banded_symmetric_mask;
use crate::toeplitz::materialize_lower_triangular;

/// Materialize a lower-triangular Toeplitz matrix as dense.
pub fn toeplitz_dense(coef: &[f64], n: usize) -> Array2<f64> {
    materialize_lower_triangular(coef, Some(n))
}

/// Invert a lower-triangular matrix via forward substitution.
///
/// Panics if the matrix is not square or has zero diagonal entries.
pub fn inverse_lower_triangular(mat: &Array2<f64>) -> Array2<f64> {
    let n = mat.nrows();
    assert_eq!(n, mat.ncols(), "matrix must be square");
    assert_lower_triangular(mat, 1e-12);

    let mut inv = Array2::zeros((n, n));
    for j in 0..n {
        let mut x = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..i {
                sum += mat[(i, k)] * x[k];
            }
            let rhs = if i == j { 1.0 } else { 0.0 };
            let diag = mat[(i, i)];
            assert!(diag != 0.0, "zero diagonal encountered during inversion");
            x[i] = (rhs - sum) / diag;
        }
        for i in 0..n {
            inv[(i, j)] = x[i];
        }
    }
    inv
}

/// Column-normalize a dense matrix, returning the norms.
pub fn column_normalize(mat: &mut Array2<f64>) -> Array1<f64> {
    normalize_columns(mat)
}

/// Expected per-query squared error for a general matrix mechanism.
pub fn per_query_error(
    strategy_matrix: Option<&Array2<f64>>,
    noising_matrix: Option<&Array2<f64>>,
    workload_matrix: Option<&Array2<f64>>,
    skip_checks: bool,
) -> Array1<f64> {
    if !skip_checks {
        assert!(
            strategy_matrix.is_some() ^ noising_matrix.is_some(),
            "specify exactly one of strategy_matrix or noising_matrix"
        );
    }

    let (n, a) = if let Some(c) = strategy_matrix {
        let n = c.nrows();
        if !skip_checks {
            assert_eq!(n, c.ncols(), "strategy_matrix must be square");
        }
        let a = workload_matrix
            .cloned()
            .unwrap_or_else(|| prefix_workload(n));
        (n, a)
    } else if let Some(c_inv) = noising_matrix {
        let n = c_inv.nrows();
        let a = workload_matrix
            .cloned()
            .unwrap_or_else(|| prefix_workload(n));
        (n, a)
    } else {
        return Array1::zeros(0);
    };

    let b = if let Some(c) = strategy_matrix {
        let c_t = c.t().to_owned();
        let a_t = a.t().to_owned();
        let inv = invert_square(&c_t);
        let x = inv.dot(&a_t);
        x.t().to_owned()
    } else {
        let c_inv = noising_matrix.expect("noising_matrix checked");
        a.dot(c_inv)
    };

    let mut out = Array1::zeros(n);
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..b.ncols() {
            let v = b[(i, j)];
            acc += v * v;
        }
        out[i] = acc;
    }
    out
}

/// Max-over-iterations squared error for a general matrix mechanism.
pub fn max_error(
    strategy_matrix: Option<&Array2<f64>>,
    noising_matrix: Option<&Array2<f64>>,
    workload_matrix: Option<&Array2<f64>>,
    skip_checks: bool,
) -> f64 {
    let errs = per_query_error(
        strategy_matrix,
        noising_matrix,
        workload_matrix,
        skip_checks,
    );
    errs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Mean-over-iterations squared error for a general matrix mechanism.
pub fn mean_error(
    strategy_matrix: Option<&Array2<f64>>,
    noising_matrix: Option<&Array2<f64>>,
    workload_matrix: Option<&Array2<f64>>,
    skip_checks: bool,
) -> f64 {
    let errs = per_query_error(
        strategy_matrix,
        noising_matrix,
        workload_matrix,
        skip_checks,
    );
    if errs.is_empty() {
        0.0
    } else {
        errs.iter().sum::<f64>() / errs.len() as f64
    }
}

/// Mask enforcing orthogonality constraints for fixed-epoch order.
pub fn get_orthogonal_mask(n: usize, epochs: usize) -> Array2<f64> {
    let mut mask = Array2::ones((n, n));
    if epochs == 0 {
        return mask;
    }
    let b = n / epochs.max(1);
    if b == 0 {
        return mask;
    }
    for i in 0..b {
        for e1 in 0..epochs {
            for e2 in 0..epochs {
                let row = i + e1 * b;
                let col = i + e2 * b;
                if row < n && col < n {
                    mask[(row, col)] = if e1 == e2 { 1.0 } else { 0.0 };
                }
            }
        }
    }
    mask
}

/// Construct a lower-triangular strategy matrix C from its Gram matrix X.
pub fn strategy_from_x(x: &Array2<f64>) -> Array2<f64> {
    let xr = reverse_matrix(x);
    let l = cholesky_lower(&xr);
    let lt = l.t().to_owned();
    reverse_matrix(&lt)
}

/// Optimize a dense strategy matrix for mean loss under simple constraints.
pub fn optimize(
    n: usize,
    epochs: usize,
    bands: Option<usize>,
    equal_norm: bool,
    workload_matrix: Option<&Array2<f64>>,
    max_optimizer_steps: usize,
) -> Array2<f64> {
    let mut mask = get_orthogonal_mask(n, epochs.max(1));
    if let Some(b) = bands {
        let band_mask = banded_symmetric_mask(n, b).mapv(|v| v as f64);
        mask = mask * &band_mask;
    }

    let init = Array2::eye(n) / epochs.max(1) as f64;
    let params = init.into_raw_vec();

    let loss_fn = |p: &[f64]| {
        if p.len() != n * n {
            return f64::INFINITY;
        }
        let mut x = match Array2::from_shape_vec((n, n), p.to_vec()) {
            Ok(v) => v,
            Err(_) => return f64::INFINITY,
        };
        project_x(&mut x, &mask, equal_norm);

        let c = match strategy_from_x_safe(&x) {
            Some(v) => v,
            None => return f64::INFINITY,
        };
        let errs = per_query_error(Some(&c), None, workload_matrix, true);
        if errs.is_empty() {
            0.0
        } else {
            errs.iter().sum::<f64>() / errs.len() as f64
        }
    };

    let optimized = optimize_projected(
        loss_fn,
        params,
        max_optimizer_steps,
        0.1,
        |p| {
            if p.len() != n * n {
                return;
            }
            if let Ok(mut x) = Array2::from_shape_vec((n, n), p.to_vec()) {
                project_x(&mut x, &mask, equal_norm);
                let flat = x.into_raw_vec();
                p.copy_from_slice(&flat);
            }
        },
        |_| false,
    );

    let x = Array2::from_shape_vec((n, n), optimized).unwrap_or_else(|_| Array2::eye(n));
    strategy_from_x_safe(&x).unwrap_or_else(|| Array2::eye(n))
}

fn project_x(x: &mut Array2<f64>, mask: &Array2<f64>, equal_norm: bool) {
    // Symmetrize.
    let xt = x.t().to_owned();
    *x = (x.to_owned() + xt) * 0.5;
    // Apply masks.
    *x = x.to_owned() * mask;
    // Enforce positive diagonal.
    for i in 0..x.nrows() {
        if !x[(i, i)].is_finite() || x[(i, i)] <= 1e-6 {
            x[(i, i)] = 1e-6;
        }
    }
    if equal_norm {
        let mut sum = 0.0;
        for i in 0..x.nrows() {
            sum += x[(i, i)];
        }
        let mean = sum / x.nrows().max(1) as f64;
        for i in 0..x.nrows() {
            x[(i, i)] = mean;
        }
    }
}

fn strategy_from_x_safe(x: &Array2<f64>) -> Option<Array2<f64>> {
    let xr = reverse_matrix(x);
    let l = cholesky_lower_safe(&xr)?;
    let lt = l.t().to_owned();
    Some(reverse_matrix(&lt))
}

fn prefix_workload(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| if i >= j { 1.0 } else { 0.0 })
}

fn reverse_matrix(mat: &Array2<f64>) -> Array2<f64> {
    let (n, m) = mat.dim();
    let mut out = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            out[(i, j)] = mat[(n - 1 - i, m - 1 - j)];
        }
    }
    out
}

fn cholesky_lower(mat: &Array2<f64>) -> Array2<f64> {
    let n = mat.nrows();
    assert_eq!(n, mat.ncols(), "matrix must be square");
    let mut l = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = mat[(i, j)];
            for k in 0..j {
                sum -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                assert!(sum > 0.0, "matrix is not positive definite");
                l[(i, j)] = sum.sqrt();
            } else {
                l[(i, j)] = sum / l[(j, j)];
            }
        }
    }
    l
}

fn cholesky_lower_safe(mat: &Array2<f64>) -> Option<Array2<f64>> {
    let n = mat.nrows();
    if n != mat.ncols() {
        return None;
    }
    let mut l = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = mat[(i, j)];
            for k in 0..j {
                sum -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return None;
                }
                l[(i, j)] = sum.sqrt();
            } else {
                let denom = l[(j, j)];
                if denom == 0.0 {
                    return None;
                }
                l[(i, j)] = sum / denom;
            }
        }
    }
    Some(l)
}

fn invert_square(mat: &Array2<f64>) -> Array2<f64> {
    let n = mat.nrows();
    assert_eq!(n, mat.ncols(), "matrix must be square");
    let mut a = mat.as_standard_layout().to_owned().into_raw_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }

    for i in 0..n {
        let mut pivot_row = i;
        let mut pivot_val = a[i * n + i].abs();
        for r in (i + 1)..n {
            let val = a[r * n + i].abs();
            if val > pivot_val {
                pivot_val = val;
                pivot_row = r;
            }
        }
        assert!(pivot_val > 1e-12, "matrix is singular");
        if pivot_row != i {
            for c in 0..n {
                a.swap(i * n + c, pivot_row * n + c);
                inv.swap(i * n + c, pivot_row * n + c);
            }
        }

        let pivot = a[i * n + i];
        for c in 0..n {
            a[i * n + c] /= pivot;
            inv[i * n + c] /= pivot;
        }

        for r in 0..n {
            if r == i {
                continue;
            }
            let factor = a[r * n + i];
            if factor == 0.0 {
                continue;
            }
            for c in 0..n {
                a[r * n + c] -= factor * a[i * n + c];
                inv[r * n + c] -= factor * inv[i * n + c];
            }
        }
    }

    Array2::from_shape_vec((n, n), inv).expect("inverse shape mismatch")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn inverse_lower_triangular_identity() {
        let id = array![[1.0, 0.0], [0.0, 1.0]];
        let inv = inverse_lower_triangular(&id);
        assert_eq!(inv, id);
    }
}
