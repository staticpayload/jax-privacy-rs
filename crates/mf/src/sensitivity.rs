//! Sensitivity calculations for matrix mechanisms.

use ndarray::{Array1, Array2};

fn ceil_div(x: usize, y: usize) -> usize {
    (x + y - 1) / y
}

/// Maximum column L2 norm (single-participation sensitivity).
pub fn single_participation_sensitivity(c: &Array2<f64>) -> f64 {
    let mut max_norm: f64 = 0.0;
    for col in c.columns() {
        let norm = col.iter().map(|v| v * v).sum::<f64>().sqrt();
        max_norm = max_norm.max(norm);
    }
    max_norm
}

/// Maximum number of participations under min-separation constraints.
pub fn minsep_true_max_participations(
    n: usize,
    min_sep: usize,
    max_participations: Option<usize>,
) -> usize {
    let min_sep = min_sep.max(1);
    let max_part_ub = ceil_div(n, min_sep);
    match max_participations {
        None => max_part_ub,
        Some(k) => k.min(max_part_ub),
    }
}

/// Maximize `<x, u>` with min-separation participation constraints.
pub fn max_participation_for_linear_fn(
    x: &Array1<f64>,
    min_sep: usize,
    max_participations: Option<usize>,
) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let min_sep = min_sep.max(1);
    let max_participations = minsep_true_max_participations(n, min_sep, max_participations);

    // DP over suffixes with cumulative max scans.
    let mut f = vec![0.0f64; n + min_sep];
    for _ in 0..max_participations {
        for i in 0..n {
            f[i] = x[i] + f[i + min_sep];
        }
        for i in (0..n).rev() {
            f[i] = f[i].max(f[i + 1]);
        }
    }
    f[0]
}

/// Lower-triangular band mask with `num_bands` bands.
pub fn banded_lower_triangular_mask(n: usize, num_bands: usize) -> Array2<i32> {
    let b = num_bands.max(1);
    let mut mask = Array2::zeros((n, n));
    for i in 0..n {
        let start = i.saturating_sub(b - 1);
        for j in start..=i {
            mask[(i, j)] = 1;
        }
    }
    mask
}

/// Symmetric band mask with `2 * num_bands - 1` bands.
pub fn banded_symmetric_mask(n: usize, num_bands: usize) -> Array2<i32> {
    let b = num_bands.max(1);
    let mut mask = Array2::zeros((n, n));
    for i in 0..n {
        let start = i.saturating_sub(b - 1);
        let end = (i + b).min(n);
        for j in start..end {
            mask[(i, j)] = 1;
        }
    }
    mask
}

/// Upper bound on min-separation sensitivity from a Gram matrix.
pub fn get_min_sep_sensitivity_upper_bound_for_x(
    x: &Array2<f64>,
    min_sep: usize,
    max_participations: Option<usize>,
) -> f64 {
    let n = x.shape()[0];
    assert_eq!(x.shape(), &[n, n], "Gram matrix must be square");

    let mut row_max = Array1::zeros(n);
    for i in 0..n {
        let row_abs: Array1<f64> = x.row(i).iter().map(|v| v.abs()).collect();
        row_max[i] = max_participation_for_linear_fn(&row_abs, min_sep, max_participations);
    }
    let value = max_participation_for_linear_fn(&row_max, min_sep, max_participations);
    value.sqrt()
}

/// Upper bound on min-separation sensitivity from an encoder matrix.
pub fn get_min_sep_sensitivity_upper_bound(
    c: &Array2<f64>,
    min_sep: usize,
    max_participations: Option<usize>,
) -> f64 {
    let x = c.t().dot(c);
    get_min_sep_sensitivity_upper_bound_for_x(&x, min_sep, max_participations)
}

/// Sensitivity for banded Gram matrices.
pub fn get_sensitivity_banded_for_x(
    x: &Array2<f64>,
    min_sep: usize,
    max_participations: Option<usize>,
) -> f64 {
    let n = x.shape()[0];
    assert_eq!(x.shape(), &[n, n], "Gram matrix must be square");
    assert!(min_sep >= 1 && min_sep <= n, "min_sep out of range");

    let mask = banded_symmetric_mask(n, min_sep);
    let tol = 1e-12;
    for i in 0..n {
        for j in 0..n {
            if mask[(i, j)] == 0 && x[(i, j)].abs() > tol {
                panic!("Gram matrix violates banded orthogonality condition");
            }
        }
    }

    let diag: Array1<f64> = (0..n).map(|i| x[(i, i)]).collect();
    let value = max_participation_for_linear_fn(&diag, min_sep, max_participations);
    value.sqrt()
}

/// Sensitivity for banded encoder matrices.
pub fn get_sensitivity_banded(
    c: &Array2<f64>,
    min_sep: usize,
    max_participations: Option<usize>,
) -> f64 {
    let x = c.t().dot(c);
    get_sensitivity_banded_for_x(&x, min_sep, max_participations)
}

/// Fixed-epoch sensitivity from a Gram matrix.
pub fn fixed_epoch_sensitivity_for_x(x: &Array2<f64>, epochs: usize) -> f64 {
    let n = x.shape()[0];
    assert_eq!(x.shape(), &[n, n], "Gram matrix must be square");
    assert!(epochs > 0, "epochs must be positive");
    assert!(n % epochs == 0, "epochs must divide n");

    let rounds_per_epoch = n / epochs;
    let mut max_sum = 0.0f64;
    for offset in 0..rounds_per_epoch {
        let indices: Vec<usize> = (0..epochs).map(|e| e * rounds_per_epoch + offset).collect();
        let mut sum = 0.0;
        for &i in &indices {
            for &j in &indices {
                sum += x[(i, j)].abs();
            }
        }
        max_sum = max_sum.max(sum);
    }
    max_sum.sqrt()
}

/// Fixed-epoch sensitivity from an encoder matrix.
pub fn fixed_epoch_sensitivity(c: &Array2<f64>, epochs: usize) -> f64 {
    let x = c.t().dot(c);
    fixed_epoch_sensitivity_for_x(&x, epochs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_max_participation_simple() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let v = max_participation_for_linear_fn(&x, 2, None);
        assert!(v >= 5.0);
    }

    #[test]
    fn test_banded_masks() {
        let mask = banded_symmetric_mask(5, 2);
        assert_eq!(mask[(0, 3)], 0);
        assert_eq!(mask[(2, 3)], 1);
    }
}
