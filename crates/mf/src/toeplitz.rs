//! Toeplitz matrices and coefficient-space utilities.

use ndarray::Array2;

use crate::optimization::optimize_projected;
use crate::sensitivity;
use crate::streaming::StreamingMatrix;

fn l2_norm_squared(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

fn reconcile(coef: &[f64], n: Option<usize>) -> (Vec<f64>, usize) {
    let n = n.unwrap_or(coef.len().max(1));
    let mut coef = coef.iter().copied().take(n).collect::<Vec<_>>();
    if coef.is_empty() {
        coef.push(0.0);
    }
    (coef, n)
}

/// Pad Toeplitz coefficients to length `n`.
pub fn pad_coefs_to_n(coef: &[f64], n: Option<usize>) -> Vec<f64> {
    let (coef, n) = reconcile(coef, n);
    let mut out = vec![0.0; n];
    let len = coef.len().min(n);
    out[..len].copy_from_slice(&coef[..len]);
    out
}

/// Materialize a lower-triangular Toeplitz matrix.
pub fn materialize_lower_triangular(coef: &[f64], n: Option<usize>) -> Array2<f64> {
    let coef = pad_coefs_to_n(coef, n);
    let n = coef.len();
    let mut mat = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            mat[(i, j)] = coef[i - j];
        }
    }
    mat
}

/// Multiply two lower-triangular Toeplitz matrices in coefficient space.
pub fn multiply(lhs_coef: &[f64], rhs_coef: &[f64], n: Option<usize>) -> Vec<f64> {
    let inferred_n = n.unwrap_or_else(|| lhs_coef.len().max(rhs_coef.len()).max(1));
    let lhs = pad_coefs_to_n(lhs_coef, Some(inferred_n));
    let rhs = pad_coefs_to_n(rhs_coef, Some(inferred_n));
    let mut out = vec![0.0; inferred_n];

    for i in 0..inferred_n {
        let mut sum = 0.0;
        for k in 0..=i {
            sum += lhs[k] * rhs[i - k];
        }
        out[i] = sum;
    }
    out
}

fn column_norms_from_coef(coef: &[f64], n: usize) -> Vec<f64> {
    let full = pad_coefs_to_n(coef, Some(n));
    let mut prefix = vec![0.0; n];
    let mut running = 0.0;
    for (i, v) in full.iter().enumerate() {
        running += v * v;
        prefix[i] = running;
    }
    (0..n)
        .map(|i| {
            let idx = n - 1 - i;
            prefix[idx].sqrt()
        })
        .collect()
}

/// A streaming representation of the inverse of a banded Toeplitz matrix.
#[derive(Clone, Debug)]
pub struct ToeplitzInverseStream {
    coef: Vec<f64>,
    row_scale: Option<Vec<f64>>,
}

/// Streaming state for the Toeplitz inverse solver.
#[derive(Clone, Debug)]
pub struct ToeplitzInverseState {
    index: usize,
    buffer: Vec<f64>,
}

impl ToeplitzInverseStream {
    /// Create a new streaming inverse.
    pub fn new(coef: &[f64], column_normalize_for_n: Option<usize>) -> Self {
        let (coef, _) = reconcile(coef, column_normalize_for_n);
        let row_scale = column_normalize_for_n.map(|n| column_norms_from_coef(&coef, n));
        Self { coef, row_scale }
    }

    fn scale_for_index(&self, index: usize) -> f64 {
        match &self.row_scale {
            None => 1.0,
            Some(scales) => scales
                .get(index)
                .copied()
                .or_else(|| scales.last().copied())
                .unwrap_or(1.0),
        }
    }
}

impl StreamingMatrix for ToeplitzInverseStream {
    type State = ToeplitzInverseState;

    fn init(&self, _n: usize) -> Self::State {
        let bands = self.coef.len();
        ToeplitzInverseState {
            index: 0,
            buffer: vec![0.0; bands.saturating_sub(1)],
        }
    }

    fn next(&self, y: f64, mut state: Self::State) -> (f64, Self::State) {
        let bands = self.coef.len();
        if bands == 0 {
            return (0.0, state);
        }
        let coef0 = self.coef[0];
        assert!(
            coef0 != 0.0,
            "leading Toeplitz coefficient must be non-zero"
        );

        let mut inner = 0.0;
        for k in 1..bands {
            let prev = state.buffer.get(k - 1).copied().unwrap_or(0.0);
            inner += self.coef[k] * prev;
        }

        let x = (y - inner) / coef0;

        // Update the buffer with the newly solved x.
        if !state.buffer.is_empty() {
            for i in (1..state.buffer.len()).rev() {
                state.buffer[i] = state.buffer[i - 1];
            }
            state.buffer[0] = x;
        }

        let scaled = x * self.scale_for_index(state.index);
        state.index = state.index.saturating_add(1);
        (scaled, state)
    }

    fn get_coeff(&self, row: usize, col: usize) -> f64 {
        if col > row {
            return 0.0;
        }
        // Materialize a prefix and read off the coefficient.
        let n = row + 1;
        let mat = materialize_lower_triangular(&inverse_coef(&self.coef, Some(n)), Some(n));
        mat[(row, col)]
    }
}

/// Represent the inverse as a streaming matrix.
pub fn inverse_as_streaming_matrix(
    coef: &[f64],
    column_normalize_for_n: Option<usize>,
) -> ToeplitzInverseStream {
    ToeplitzInverseStream::new(coef, column_normalize_for_n)
}

/// Solve `T_coef x = rhs` for a lower-triangular Toeplitz matrix.
pub fn solve_banded(coef: &[f64], rhs: &[f64]) -> Vec<f64> {
    let stream = inverse_as_streaming_matrix(coef, None);
    let mut state = stream.init(rhs.len());
    let mut out = Vec::with_capacity(rhs.len());
    for &y in rhs {
        let (x, new_state) = stream.next(y, state);
        state = new_state;
        out.push(x);
    }
    out
}

/// Toeplitz coefficients of the inverse matrix.
pub fn inverse_coef(coef: &[f64], n: Option<usize>) -> Vec<f64> {
    let (_, n) = reconcile(coef, n);
    let mut rhs = vec![0.0; n];
    rhs[0] = 1.0;
    solve_banded(coef, &rhs)
}

/// Sensitivity squared under single participation.
pub fn sensitivity_squared(coef: &[f64], n: Option<usize>) -> f64 {
    let (coef, _) = reconcile(coef, n);
    l2_norm_squared(&coef)
}

/// Sensitivity squared under min-separation participation constraints.
pub fn minsep_sensitivity_squared(
    strategy_coef: &[f64],
    min_sep: usize,
    max_participations: Option<usize>,
    n: Option<usize>,
    skip_checks: bool,
) -> f64 {
    let min_sep = min_sep.max(1);
    let (coef, n) = reconcile(strategy_coef, n);

    if !skip_checks {
        if coef.iter().any(|v| *v < 0.0) {
            panic!("Toeplitz coefficients must be non-negative");
        }
        for w in coef.windows(2) {
            if w[1] > w[0] {
                panic!("Toeplitz coefficients must be non-increasing");
            }
        }
    }

    let k = sensitivity::minsep_true_max_participations(n, min_sep, max_participations);

    let padding = (min_sep - n % min_sep) % min_sep;
    let mut padded = coef;
    padded.resize(n + padding, 0.0);
    let blocks = padded.len() / min_sep;

    let mut vector = vec![0.0; padded.len()];
    for pos in 0..min_sep {
        let mut running = 0.0;
        for block in 0..blocks {
            let idx = block * min_sep + pos;
            running += padded[idx];
            vector[idx] = running;
        }
    }

    let shift = min_sep * k;
    if shift > 0 && shift < vector.len() {
        for i in shift..vector.len() {
            vector[i] -= vector[i - shift];
        }
    }

    vector.iter().take(n).map(|v| v * v).sum()
}

/// Expected per-query squared error for Toeplitz mechanisms.
pub fn per_query_error(
    strategy_coef: Option<&[f64]>,
    noising_coef: Option<&[f64]>,
    n: Option<usize>,
    workload_coef: Option<&[f64]>,
    skip_checks: bool,
) -> Vec<f64> {
    if !skip_checks {
        match (strategy_coef.is_some(), noising_coef.is_some()) {
            (true, true) | (false, false) => {
                panic!("Specify exactly one of strategy_coef or noising_coef")
            }
            _ => {}
        }
    }

    let (b_coef, n) = if let Some(strategy) = strategy_coef {
        let (strategy, n) = reconcile(strategy, n);
        let workload = workload_coef
            .map(|w| pad_coefs_to_n(w, Some(n)))
            .unwrap_or_else(|| vec![1.0; n]);
        (solve_banded(&strategy, &workload), n)
    } else {
        let noising = noising_coef.expect("noising coefficients required");
        let (noising, n) = reconcile(noising, n);
        if let Some(workload) = workload_coef {
            let workload = pad_coefs_to_n(workload, Some(n));
            (multiply(&workload, &noising, Some(n)), n)
        } else {
            let mut cumsum = Vec::with_capacity(n);
            let mut running = 0.0;
            for v in noising {
                running += v;
                cumsum.push(running);
            }
            (cumsum, n)
        }
    };

    let mut out = Vec::with_capacity(n);
    let mut running = 0.0;
    for v in b_coef.into_iter().take(n) {
        running += v * v;
        out.push(running);
    }
    out
}

/// Max-over-iterations squared error.
pub fn max_error(
    strategy_coef: Option<&[f64]>,
    noising_coef: Option<&[f64]>,
    n: Option<usize>,
    workload_coef: Option<&[f64]>,
    skip_checks: bool,
) -> f64 {
    per_query_error(strategy_coef, noising_coef, n, workload_coef, skip_checks)
        .last()
        .copied()
        .unwrap_or(0.0)
}

/// Mean-over-iterations squared error.
pub fn mean_error(
    strategy_coef: Option<&[f64]>,
    noising_coef: Option<&[f64]>,
    n: Option<usize>,
    workload_coef: Option<&[f64]>,
    skip_checks: bool,
) -> f64 {
    let errors = per_query_error(strategy_coef, noising_coef, n, workload_coef, skip_checks);
    if errors.is_empty() {
        0.0
    } else {
        errors.iter().sum::<f64>() / errors.len() as f64
    }
}

/// Error times sensitivity for the prefix workload.
pub fn mean_loss(strategy_coef: &[f64], n: Option<usize>) -> f64 {
    let (strategy, n) = reconcile(strategy_coef, n);
    mean_error(Some(&strategy), None, Some(n), None, true) * sensitivity_squared(&strategy, Some(n))
}

/// Max error times sensitivity for the prefix workload.
pub fn max_loss(strategy_coef: &[f64], n: Option<usize>) -> f64 {
    let (strategy, n) = reconcile(strategy_coef, n);
    max_error(Some(&strategy), None, Some(n), None, true) * sensitivity_squared(&strategy, Some(n))
}

/// Optimal Toeplitz strategy coefficients for max error.
pub fn optimal_max_error_strategy_coefs(n: usize) -> Vec<f64> {
    let mut coefs = Vec::with_capacity(n);
    let mut running = 1.0;
    for k in 0..n {
        if k == 0 {
            running = 1.0;
        } else {
            let kf = k as f64;
            running *= (2.0 * kf - 1.0) / (2.0 * kf);
        }
        coefs.push(running);
    }
    coefs
}

/// Optimal Toeplitz noising coefficients for max error.
pub fn optimal_max_error_noising_coefs(n: usize) -> Vec<f64> {
    let c = optimal_max_error_strategy_coefs(n);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if i == 0 {
            out.push(c[0]);
        } else {
            out.push(c[i] - c[i - 1]);
        }
    }
    out
}

fn project_nonnegative_unit_l2(params: &mut [f64]) {
    for v in params.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
        if v.is_sign_negative() {
            *v = -*v;
        }
    }
    let norm = l2_norm_squared(params).sqrt();
    if norm > 0.0 && norm.is_finite() {
        for v in params.iter_mut() {
            *v /= norm;
        }
    } else if !params.is_empty() {
        params.fill(0.0);
        params[0] = 1.0;
    }
}

/// Heuristic optimization over banded Toeplitz strategies.
pub fn optimize_banded_toeplitz(n: usize, bands: usize, max_optimizer_steps: usize) -> Vec<f64> {
    let bands = bands.max(1).min(n.max(1));
    let init = optimal_max_error_strategy_coefs(bands);
    let steps = max_optimizer_steps.max(1);

    let loss_fn = |p: &[f64]| mean_loss(p, Some(n));
    let mut best = (f64::INFINITY, init.clone());

    let out = optimize_projected(
        loss_fn,
        init,
        steps,
        0.2,
        project_nonnegative_unit_l2,
        |args| {
            if args.loss.is_finite() && args.loss < best.0 {
                best = (args.loss, args.params.to_vec());
            }
            false
        },
    );

    if best.0.is_finite() {
        let mut best_params = best.1;
        project_nonnegative_unit_l2(&mut best_params);
        best_params
    } else {
        out
    }
}

/// Lower-triangular Toeplitz streaming matrix.
#[derive(Clone, Debug)]
pub struct Toeplitz {
    /// Coefficients: coeffs[0] is diagonal, coeffs[k] is k-th sub-diagonal.
    coeffs: Vec<f64>,
}

impl Toeplitz {
    /// Create a Toeplitz matrix from coefficients.
    pub fn new(coeffs: Vec<f64>) -> Self {
        Self { coeffs }
    }

    /// Create a banded identity (coeffs = [1, 0, 0, ...]).
    pub fn identity(bandwidth: usize) -> Self {
        let mut coeffs = vec![0.0; bandwidth.max(1)];
        coeffs[0] = 1.0;
        Self { coeffs }
    }

    /// Number of bands (including the diagonal).
    pub fn bandwidth(&self) -> usize {
        self.coeffs.len()
    }
}

/// State for streaming Toeplitz multiplication.
#[derive(Clone, Debug)]
pub struct ToeplitzState {
    buffer: Vec<f64>,
    pos: usize,
    filled: usize,
}

impl StreamingMatrix for Toeplitz {
    type State = ToeplitzState;

    fn init(&self, _n: usize) -> Self::State {
        ToeplitzState {
            buffer: vec![0.0; self.coeffs.len().max(1)],
            pos: 0,
            filled: 0,
        }
    }

    fn next(&self, x: f64, mut state: Self::State) -> (f64, Self::State) {
        let b = self.coeffs.len();
        if b == 0 {
            return (0.0, state);
        }

        state.buffer[state.pos] = x;
        state.pos = (state.pos + 1) % b;
        state.filled = (state.filled + 1).min(b);

        let mut result = 0.0;
        for k in 0..state.filled {
            let idx = (state.pos + b - 1 - k) % b;
            result += self.coeffs[k] * state.buffer[idx];
        }

        (result, state)
    }

    fn get_coeff(&self, row: usize, col: usize) -> f64 {
        if col > row {
            return 0.0;
        }
        let lag = row - col;
        self.coeffs.get(lag).copied().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_and_materialize() {
        let coef = vec![1.0, 0.5];
        let mat = materialize_lower_triangular(&coef, Some(4));
        assert_eq!(mat[(0, 0)], 1.0);
        assert_eq!(mat[(1, 0)], 0.5);
        assert_eq!(mat[(2, 0)], 0.0);
    }

    #[test]
    fn test_inverse_coef_identity() {
        let coef = vec![1.0, 0.0, 0.0];
        let inv = inverse_coef(&coef, Some(3));
        assert!((inv[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_optimal_coefs_monotone() {
        let coefs = optimal_max_error_strategy_coefs(5);
        for w in coefs.windows(2) {
            assert!(w[1] <= w[0] + 1e-12);
        }
    }

    #[test]
    fn optimize_banded_is_projected() {
        let coefs = optimize_banded_toeplitz(32, 4, 25);
        assert_eq!(coefs.len(), 4);
        assert!(coefs.iter().all(|v| v.is_finite() && *v >= 0.0));
        let norm = l2_norm_squared(&coefs).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
