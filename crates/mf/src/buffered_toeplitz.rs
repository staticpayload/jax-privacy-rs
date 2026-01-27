//! Buffered Linear Toeplitz (BLT) helpers.
//!
//! This is a streamlined version of the JAX implementation that keeps the core
//! ideas: a small number of exponentially decayed buffers that induce a
//! Toeplitz strategy with efficient streaming evaluation.

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::streaming::StreamingMatrix;
use crate::toeplitz::{
    max_error as toeplitz_max_error, max_loss as toeplitz_max_loss, mean_error,
    minsep_sensitivity_squared, Toeplitz,
};

/// Builder that turns BLT parameters into streaming matrices.
#[derive(Clone, Debug)]
pub struct StreamingMatrixBuilder {
    /// Per-buffer decay factors.
    pub buf_decay: Vec<f64>,
    /// Per-buffer readout scales.
    pub output_scale: Vec<f64>,
}

impl StreamingMatrixBuilder {
    fn assert_valid(&self) {
        assert_eq!(
            self.buf_decay.len(),
            self.output_scale.len(),
            "buf_decay and output_scale must match in length"
        );
    }

    fn read(&self, bufs: &[f64]) -> f64 {
        self.output_scale
            .iter()
            .zip(bufs.iter())
            .map(|(s, b)| s * b)
            .sum()
    }

    fn update(&self, bufs: &mut [f64], next_rhs: f64) {
        for ((b, d), _) in bufs
            .iter_mut()
            .zip(self.buf_decay.iter())
            .zip(self.output_scale.iter())
        {
            *b = *d * *b + next_rhs;
        }
    }

    /// Streaming matrix for `C`.
    pub fn build(&self) -> BufferedStreamingMatrix {
        self.assert_valid();
        BufferedStreamingMatrix {
            builder: self.clone(),
            inverse: false,
        }
    }

    /// Streaming matrix for `C^{-1}`.
    pub fn build_inverse(&self) -> BufferedStreamingMatrix {
        self.assert_valid();
        BufferedStreamingMatrix {
            builder: self.clone(),
            inverse: true,
        }
    }
}

/// BLT parameterization of a lower-triangular Toeplitz matrix.
#[derive(Clone, Debug)]
pub struct BufferedToeplitz {
    /// Buffer decay factors.
    pub buf_decay: Vec<f64>,
    /// Output scales.
    pub output_scale: Vec<f64>,
}

impl BufferedToeplitz {
    /// Validate basic BLT invariants.
    pub fn validate(&self) {
        assert_eq!(
            self.buf_decay.len(),
            self.output_scale.len(),
            "buf_decay and output_scale must match in length"
        );
    }

    /// Build a BLT and canonicalize the parameter ordering.
    pub fn build(buf_decay: Vec<f64>, output_scale: Vec<f64>) -> Self {
        let blt = Self {
            buf_decay,
            output_scale,
        };
        blt.validate();
        blt.canonicalize()
    }

    /// Construct a BLT directly from parameters.
    pub fn new(buf_decay: Vec<f64>, output_scale: Vec<f64>) -> Self {
        let blt = Self {
            buf_decay,
            output_scale,
        };
        blt.validate();
        blt
    }

    /// Returns a BLT with `buf_decay` sorted in descending order.
    pub fn canonicalize(&self) -> Self {
        self.validate();
        let mut pairs: Vec<(f64, f64)> = self
            .buf_decay
            .iter()
            .copied()
            .zip(self.output_scale.iter().copied())
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let (buf_decay, output_scale): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
        Self {
            buf_decay,
            output_scale,
        }
    }

    /// Create a streaming builder.
    pub fn builder(&self) -> StreamingMatrixBuilder {
        StreamingMatrixBuilder {
            buf_decay: self.buf_decay.clone(),
            output_scale: self.output_scale.clone(),
        }
    }

    /// Induced Toeplitz coefficients for a given horizon `n`.
    pub fn toeplitz_coefs(&self, n: usize) -> Vec<f64> {
        self.validate();
        let mut coefs = Vec::with_capacity(n.max(1));
        coefs.push(1.0);
        for k in 1..n {
            let power = (k - 1) as i32;
            let tk = self
                .output_scale
                .iter()
                .zip(self.buf_decay.iter())
                .map(|(s, d)| s * d.powi(power))
                .sum();
            coefs.push(tk);
        }
        coefs
    }

    /// Materialize the Toeplitz matrix at horizon `n`.
    pub fn materialize(&self, n: usize) -> ndarray::Array2<f64> {
        self.to_toeplitz(n).materialize(n)
    }

    /// Convert to a Toeplitz strategy at horizon `n`.
    pub fn to_toeplitz(&self, n: usize) -> Toeplitz {
        Toeplitz::new(self.toeplitz_coefs(n))
    }

    /// Streaming matrix representation of `C`.
    pub fn as_streaming_matrix(&self) -> BufferedStreamingMatrix {
        self.builder().build()
    }

    /// Streaming matrix representation of `C^{-1}`.
    pub fn inverse_as_streaming_matrix(&self) -> BufferedStreamingMatrix {
        self.builder().build_inverse()
    }

    /// Compute a BLT parameterization of the inverse.
    pub fn inverse(&self) -> Self {
        self.validate();
        let n = self.buf_decay.len();
        if n == 0 {
            return self.clone();
        }
        let (theta, omega) = sort_pairs(&self.buf_decay, &self.output_scale);

        if n == 1 {
            let eval = theta[0] - omega[0];
            let out = BufferedToeplitz::new(vec![eval], vec![-omega[0]]);
            return out.canonicalize();
        }

        let evals = eigenvalues_rank1(&theta, &omega);
        let evecs = eigenvectors_from_evals(&theta, &omega, &evals);
        let einv = invert_matrix(&evecs, n);

        let col_sums = column_sums(&evecs, n);
        let omega2: Vec<f64> = omega.iter().map(|v| -v).collect();
        let omega_left = mat_vec_mul(&einv, &omega2, n);
        let mut out_scale = Vec::with_capacity(n);
        for j in 0..n {
            out_scale.push(omega_left[j] * col_sums[j]);
        }

        BufferedToeplitz::build(evals, out_scale)
    }

    /// Sum(output_scale / buf_decay).
    pub fn pillutla_score(&self) -> f64 {
        self.buf_decay
            .iter()
            .zip(self.output_scale.iter())
            .map(|(d, s)| s / d)
            .sum()
    }

    /// Construct a BLT via a rational approximation to 1/sqrt(1-x).
    pub fn from_rational_approx_to_sqrt_x(
        num_buffers: usize,
        max_buf_decay: f64,
        max_pillutla_score: Option<f64>,
        buf_decay_scale: f64,
        buf_decay_shift: i32,
    ) -> Self {
        assert!(num_buffers >= 1, "num_buffers must be >= 1");

        let degree = num_buffers as i32;
        let d1 = (degree + 1) / 2;
        let h = buf_decay_scale * PI / (2.0 * (d1 + 1) as f64).sqrt();

        let mut buf_decay = Vec::with_capacity(num_buffers);
        let mut output_scale = Vec::with_capacity(num_buffers);
        let mut constant_term = 0.0;

        for i in 0..num_buffers {
            let k = -d1 + 1 + i as i32 + buf_decay_shift;
            let kf = k as f64;
            let decay = 1.0 / (1.0 + (2.0 * h * kf).exp());
            let scale = -(decay * decay) * (3.0 * h * kf).exp();
            buf_decay.push(decay);
            output_scale.push(scale);
            constant_term += (h * kf).exp() * decay;
        }

        for s in &mut output_scale {
            *s /= constant_term;
        }

        let inv_blt = BufferedToeplitz::build(buf_decay, output_scale);
        let mut blt = inv_blt.inverse();

        let largest = blt.buf_decay[0];
        let scale = (max_buf_decay / largest).min(1.0);
        for d in &mut blt.buf_decay {
            *d *= scale;
        }

        let score: f64 = blt
            .output_scale
            .iter()
            .zip(blt.buf_decay.iter())
            .map(|(s, d)| s / d)
            .sum();
        if let Some(max_score) = max_pillutla_score {
            let score_scale = (max_score / score).min(1.0);
            for s in &mut blt.output_scale {
                *s *= score_scale;
            }
        }

        let blt = BufferedToeplitz::build(blt.buf_decay, blt.output_scale);
        if let Some(max_score) = max_pillutla_score {
            assert!(
                blt.pillutla_score() <= max_score + 1e-12,
                "pillutla_score exceeds maximum"
            );
        }
        blt
    }
}

fn sort_pairs(buf_decay: &[f64], output_scale: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut pairs: Vec<(f64, f64)> = buf_decay
        .iter()
        .copied()
        .zip(output_scale.iter().copied())
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let (buf_decay, output_scale): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
    (buf_decay, output_scale)
}

fn secular(lambda: f64, theta: &[f64], omega: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (t, w) in theta.iter().zip(omega.iter()) {
        sum += w / (t - lambda);
    }
    1.0 - sum
}

fn bisect_root(theta: &[f64], omega: &[f64], mut lo: f64, mut hi: f64) -> f64 {
    let f_lo = secular(lo, theta, omega);
    let f_hi = secular(hi, theta, omega);
    if f_lo.is_nan() || f_hi.is_nan() {
        panic!("secular function returned NaN during bracketing");
    }
    assert!(f_lo > 0.0 && f_hi < 0.0, "root not bracketed");
    for _ in 0..120 {
        let mid = 0.5 * (lo + hi);
        let f_mid = secular(mid, theta, omega);
        if !f_mid.is_finite() {
            break;
        }
        if f_mid > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() < 1e-10 {
            break;
        }
    }
    0.5 * (lo + hi)
}

fn eigenvalues_rank1(theta: &[f64], omega: &[f64]) -> Vec<f64> {
    assert_eq!(theta.len(), omega.len());
    let n = theta.len();
    let mut evals = Vec::with_capacity(n);

    let eps_base = 1e-10;

    // Root below the smallest diagonal entry.
    let right = theta[0] - eps_base * (1.0 + theta[0].abs());
    let mut left = right - 1.0;
    let mut f_left = secular(left, theta, omega);
    let mut step = 1.0;
    let mut guard = 0usize;
    while f_left <= 0.0 && guard < 200 {
        left -= step;
        step *= 2.0;
        f_left = secular(left, theta, omega);
        guard += 1;
    }
    if f_left <= 0.0 {
        left = right - 1e6;
    }
    evals.push(bisect_root(theta, omega, left, right));

    // Roots between consecutive diagonal entries.
    for i in 0..(n - 1) {
        let t_lo = theta[i];
        let t_hi = theta[i + 1];
        let eps = eps_base * (1.0 + t_lo.abs().max(t_hi.abs()));
        let left = t_lo + eps;
        let right = t_hi - eps;
        evals.push(bisect_root(theta, omega, left, right));
    }

    evals
}

fn eigenvectors_from_evals(theta: &[f64], omega: &[f64], evals: &[f64]) -> Vec<f64> {
    let n = theta.len();
    let mut evecs = vec![0.0; n * n];
    for i in 0..n {
        for (j, &eval) in evals.iter().enumerate() {
            let denom = eval - theta[i];
            evecs[i * n + j] = omega[i] / denom;
        }
    }
    evecs
}

fn invert_matrix(matrix: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(matrix.len(), n * n);
    let mut a = matrix.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }

    for i in 0..n {
        // Pivot selection.
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

    inv
}

fn mat_vec_mul(matrix: &[f64], vector: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(matrix.len(), n * n);
    assert_eq!(vector.len(), n);
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += matrix[i * n + j] * vector[j];
        }
        out[i] = acc;
    }
    out
}

fn column_sums(matrix: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(matrix.len(), n * n);
    let mut sums = vec![0.0; n];
    for j in 0..n {
        let mut acc = 0.0;
        for i in 0..n {
            acc += matrix[i * n + j];
        }
        sums[j] = acc;
    }
    sums
}

/// Streaming state for BLT strategies.
#[derive(Clone, Debug, Default)]
pub struct BufferedState {
    index: usize,
    buffers: Vec<f64>,
}

/// Streaming BLT matrix, either `C` or `C^{-1}`.
#[derive(Clone, Debug)]
pub struct BufferedStreamingMatrix {
    builder: StreamingMatrixBuilder,
    inverse: bool,
}

impl StreamingMatrix for BufferedStreamingMatrix {
    type State = BufferedState;

    fn init(&self, _n: usize) -> Self::State {
        let k = self.builder.buf_decay.len();
        BufferedState {
            index: 0,
            buffers: vec![0.0; k],
        }
    }

    fn next(&self, value: f64, mut state: Self::State) -> (f64, Self::State) {
        let read = self.builder.read(&state.buffers);
        let out = if self.inverse {
            value - read
        } else {
            value + read
        };
        let rhs = if self.inverse { out } else { value };
        self.builder.update(&mut state.buffers, rhs);
        state.index = state.index.saturating_add(1);
        (out, state)
    }

    fn get_coeff(&self, row: usize, col: usize) -> f64 {
        if col > row {
            return 0.0;
        }
        // Materialize a Toeplitz approximation for coefficient inspection.
        let n = row + 1;
        let toeplitz = BufferedToeplitz::new(
            self.builder.buf_decay.clone(),
            self.builder.output_scale.clone(),
        )
        .to_toeplitz(n);
        toeplitz.materialize(n)[(row, col)]
    }
}

/// Convenience: BLT sensitivity squared at horizon `n`.
pub fn sensitivity_squared_blt(blt: &BufferedToeplitz, n: usize) -> f64 {
    if blt.buf_decay.iter().any(|v| *v > 1.0) {
        return f64::INFINITY;
    }
    let num = n.saturating_sub(1) as f64;
    let mut acc = 0.0;
    for (&omega_i, &theta_i) in blt.output_scale.iter().zip(blt.buf_decay.iter()) {
        for (&omega_j, &theta_j) in blt.output_scale.iter().zip(blt.buf_decay.iter()) {
            acc += geometric_sum(omega_i * omega_j, theta_i * theta_j, num);
        }
    }
    1.0 + acc
}

/// Limit of sensitivity squared as n -> infinity.
pub fn sensitivity_squared_blt_limit(blt: &BufferedToeplitz) -> f64 {
    if blt.buf_decay.iter().any(|v| *v > 1.0) {
        return f64::INFINITY;
    }
    let mut acc = 0.0;
    for (&omega_i, &theta_i) in blt.output_scale.iter().zip(blt.buf_decay.iter()) {
        for (&omega_j, &theta_j) in blt.output_scale.iter().zip(blt.buf_decay.iter()) {
            acc += geometric_sum(omega_i * omega_j, theta_i * theta_j, f64::INFINITY);
        }
    }
    1.0 + acc
}

/// Convenience: BLT min-separation sensitivity squared at horizon `n`.
pub fn minsep_sensitivity_squared_blt(
    blt: &BufferedToeplitz,
    n: usize,
    min_sep: usize,
    max_participations: Option<usize>,
) -> f64 {
    let coefs = blt.toeplitz_coefs(n);
    minsep_sensitivity_squared(&coefs, min_sep, max_participations, Some(n), false)
}

/// Convenience: BLT mean error at horizon `n`.
pub fn mean_error_blt(blt: &BufferedToeplitz, n: usize) -> f64 {
    let coefs = blt.toeplitz_coefs(n);
    mean_error(Some(&coefs), None, Some(n), None, false)
}

/// Convenience: BLT max error at horizon `n`.
pub fn max_error_blt(blt: &BufferedToeplitz, n: usize) -> f64 {
    let coefs = blt.toeplitz_coefs(n);
    toeplitz_max_error(Some(&coefs), None, Some(n), None, false)
}

/// Convenience: BLT max loss at horizon `n`.
pub fn max_loss_blt(blt: &BufferedToeplitz, n: usize) -> f64 {
    let coefs = blt.toeplitz_coefs(n);
    toeplitz_max_loss(&coefs, Some(n))
}

/// Iteration error for a BLT inverse at iteration `i` (0-indexed).
pub fn iteration_error(inv_blt: &BufferedToeplitz, i: usize) -> f64 {
    let n = i.saturating_add(1) as f64;
    let omega = &inv_blt.output_scale;
    let theta = &inv_blt.buf_decay;
    let mut s1 = 0.0;
    for (&w, &t) in omega.iter().zip(theta.iter()) {
        s1 += robust_max_error_gamma_j(w, t, n);
    }
    let mut s2 = 0.0;
    for (&w1, &t1) in omega.iter().zip(theta.iter()) {
        for (&w2, &t2) in omega.iter().zip(theta.iter()) {
            s2 += robust_max_error_gamma_jk(w1, t1, w2, t2, n);
        }
    }
    n * (1.0 + 2.0 * s1 + s2)
}

/// Max squared error for any iteration 0..n-1.
pub fn max_error(inv_blt: &BufferedToeplitz, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    iteration_error(inv_blt, n - 1)
}

/// Limit of (1/n) max_error as n -> infinity (BLT formula).
pub fn limit_max_error(inv_blt: &BufferedToeplitz) -> f64 {
    let omega = &inv_blt.output_scale;
    let theta = &inv_blt.buf_decay;
    if omega.is_empty() {
        return 0.0;
    }

    let mut sum1 = 0.0;
    for (w, t) in omega.iter().zip(theta.iter()) {
        let denom = 1.0 - t;
        if denom.abs() < 1e-12 {
            continue;
        }
        sum1 += w / denom;
    }

    let mut sum2 = 0.0;
    for i in 0..omega.len() {
        for j in 0..omega.len() {
            let denom = (1.0 - theta[i]) * (1.0 - theta[j]);
            if denom.abs() < 1e-12 {
                continue;
            }
            sum2 += omega[i] * omega[j] / denom;
        }
    }

    1.0 + 2.0 * sum1 + sum2
}

/// Max squared error scaled by sensitivity for a BLT.
pub fn max_loss(blt: &BufferedToeplitz, n: usize) -> f64 {
    let sens = sensitivity_squared_blt(blt, n);
    let inv = blt.inverse();
    max_error(&inv, n) * sens
}

/// Limit of (1/n) max squared error scaled by sensitivity.
pub fn limit_max_loss(blt: &BufferedToeplitz) -> f64 {
    let sens = sensitivity_squared_blt_limit(blt);
    let inv = blt.inverse();
    limit_max_error(&inv) * sens
}

/// Smallest gap between distinct buf_decay values.
pub fn min_buf_decay_gap(buf_decay: &[f64]) -> f64 {
    if buf_decay.len() < 2 {
        return f64::INFINITY;
    }
    let mut min_gap = f64::INFINITY;
    for i in 0..buf_decay.len() {
        for j in (i + 1)..buf_decay.len() {
            let gap = (buf_decay[i] - buf_decay[j]).abs();
            if gap < min_gap {
                min_gap = gap;
            }
        }
    }
    min_gap
}

fn geometric_sum(a: f64, r: f64, num: f64) -> f64 {
    if !num.is_finite() {
        return a / (1.0 - r);
    }
    finite_n_geo_sum(a, num, r)
}

fn finite_n_geo_sum(a: f64, n: f64, r: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    // Thresholds from the JAX implementation.
    const SLOPE: f64 = 0.53018965;
    const INTERCEPT: f64 = 3.33503185;
    let pow_threshold = INTERCEPT + SLOPE * n.ln();
    let threshold = 1.0 - 10f64.powf(-pow_threshold);
    if r < threshold {
        a * (1.0 - r.powf(n)) / (1.0 - r)
    } else {
        let x0 = n - 1.0;
        let x1 = r - 1.0;
        (1.0 / 6.0) * a * n * (x0 * x1 * x1 * (n - 2.0) + 3.0 * x0 * x1 + 6.0)
    }
}

fn max_error_gamma_j(omega: f64, theta: f64, n: f64) -> f64 {
    let denom = 1.0 - theta;
    if denom == 0.0 {
        return f64::INFINITY;
    }
    (omega / denom) * (1.0 - geometric_sum(1.0, theta, n) / n)
}

fn max_error_gamma_j_series(omega: f64, theta: f64, n: f64) -> f64 {
    let x0 = theta - 1.0;
    let x1 = omega * (n - 2.0) * (n - 1.0);
    -omega * (0.5 - 0.5 * n) + (1.0 / 24.0) * x0 * x0 * x1 * (n - 3.0) + (1.0 / 6.0) * x0 * x1
}

fn robust_max_error_gamma_j(omega: f64, theta: f64, n: f64) -> f64 {
    const J_SLOPE: f64 = 0.43877484;
    const J_INTERCEPT: f64 = 2.91215085;
    let power = J_INTERCEPT + J_SLOPE * n.ln();
    let threshold = 1.0 - 10f64.powf(-power);
    if theta < threshold {
        max_error_gamma_j(omega, theta, n)
    } else {
        max_error_gamma_j_series(omega, theta, n)
    }
}

fn max_error_gamma_jk(omega1: f64, theta1: f64, omega2: f64, theta2: f64, n: f64) -> f64 {
    let denom = (1.0 - theta1) * (1.0 - theta2);
    if denom == 0.0 {
        return f64::INFINITY;
    }
    let temp1 = omega1 * omega2 / denom;
    let temp2 = (n - geometric_sum(1.0, theta1, n) - geometric_sum(1.0, theta2, n)
        + geometric_sum(1.0, theta1 * theta2, n))
        / n;
    temp1 * temp2
}

fn max_error_gamma_jk_series_j(omega1: f64, theta1: f64, omega2: f64, theta2: f64, n: f64) -> f64 {
    let x0 = theta2 - 1.0;
    let x1 = theta2.powf(n + 1.0);
    let x2 = -x1;
    let x3 = 6.0 * x0;
    let x4 = theta2.powf(n);
    let x5 = n - 1.0;
    let x6 = theta1 - 1.0;
    let x7 = theta2.powf(n + 2.0);
    (-1.0 / 6.0)
        * omega1
        * omega2
        * (n * x0.powi(3) * (3.0 * n + x5 * x6 * (n - 2.0) - 3.0) + n * x3 * (x2 + x4)
            - x3 * (theta2 + x2)
            + 3.0
                * x6
                * (n * x0 * (-x1 * x5 + x4 * x5) - 2.0 * n * (x1 - x7) + 2.0 * theta2.powi(2)
                    - 2.0 * x7))
        / (n * x0.powi(4))
}

fn max_error_gamma_jk_series_jk(omega1: f64, theta1: f64, omega2: f64, theta2: f64, n: f64) -> f64 {
    let x0 = n * n;
    let x1 = 3.0 * n * n * n + 9.0 * n - 10.0 * x0 - 2.0;
    (1.0 / 24.0)
        * omega1
        * omega2
        * (-12.0 * n + 8.0 * x0 + x1 * (theta1 - 1.0) + x1 * (theta2 - 1.0) + 4.0)
}

fn robust_max_error_gamma_jk(omega1: f64, theta1: f64, omega2: f64, theta2: f64, n: f64) -> f64 {
    let (theta1, theta2) = if theta1 >= theta2 {
        (theta1, theta2)
    } else {
        (theta2, theta1)
    };
    const JK_SLOPE: f64 = 0.35321577;
    const JK_INTERCEPT: f64 = 2.81518052;
    let power = JK_INTERCEPT + JK_SLOPE * n.ln();
    let threshold = 1.0 - 10f64.powf(-power);
    let v0_predicate = theta1 < threshold;
    let v1_predicate = theta2 < threshold;
    if v0_predicate {
        max_error_gamma_jk(omega1, theta1, omega2, theta2, n)
    } else if v1_predicate {
        max_error_gamma_jk_series_j(omega1, theta1, omega2, theta2, n)
    } else {
        max_error_gamma_jk_series_jk(omega1, theta1, omega2, theta2, n)
    }
}

fn gt_zero_penalty(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |acc, &v| {
        if v > 0.0 && v.is_finite() {
            acc - v.ln()
        } else {
            f64::INFINITY
        }
    })
}

fn lt_zero_penalty(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |acc, &v| {
        if v < 0.0 && v.is_finite() {
            acc - (-v).ln()
        } else {
            f64::INFINITY
        }
    })
}

fn lt_penalty(value: f64, upper_bound: f64) -> f64 {
    let diff = upper_bound - value;
    if diff > 0.0 && diff.is_finite() {
        -diff.ln()
    } else {
        f64::INFINITY
    }
}

fn lt_one_penalty(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |acc, &v| {
        let diff = 1.0 - v;
        if diff > 0.0 && diff.is_finite() {
            acc - diff.ln()
        } else {
            f64::INFINITY
        }
    })
}

/// Compute BLTs (C, C^{-1}) from buf_decay pairs.
pub fn blt_pair_from_theta_pair(
    theta: &[f64],
    theta_hat: &[f64],
) -> (BufferedToeplitz, BufferedToeplitz) {
    fn get_omega(theta: &[f64], theta_hat: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(theta.len());
        for (i, &t_i) in theta.iter().enumerate() {
            let mut numer = 1.0;
            for &t_hat in theta_hat {
                numer *= t_i - t_hat;
            }
            let mut denom = 1.0;
            for (j, &t_j) in theta.iter().enumerate() {
                if i == j {
                    continue;
                }
                denom *= t_i - t_j;
            }
            out.push(numer / denom);
        }
        out
    }

    let omega = get_omega(theta, theta_hat);
    let omega_hat = get_omega(theta_hat, theta);
    (
        BufferedToeplitz::build(theta.to_vec(), omega),
        BufferedToeplitz::build(theta_hat.to_vec(), omega_hat),
    )
}

/// Loss configuration for BLT optimization.
pub struct LossFn {
    error_for_inv: Box<dyn Fn(&BufferedToeplitz) -> f64 + Send + Sync>,
    sensitivity_squared: Box<dyn Fn(&BufferedToeplitz) -> f64 + Send + Sync>,
    /// Number of iterations.
    pub n: usize,
    /// Minimum separation of participations.
    pub min_sep: usize,
    /// Effective maximum participation count.
    pub max_participations: usize,
    /// Strength of the penalty term.
    pub penalty_strength: f64,
    /// Per-penalty multipliers.
    pub penalty_multipliers: HashMap<String, f64>,
    /// Upper bound on the second Toeplitz coefficient.
    pub max_second_coef: f64,
    /// Minimum allowed gap between buf_decay parameters.
    pub min_theta_gap: f64,
}

impl LossFn {
    /// Construct a loss for single-participation max-error.
    pub fn build_closed_form_single_participation(n: usize) -> Self {
        let error_for_inv = Box::new(move |inv_blt: &BufferedToeplitz| max_error(inv_blt, n));
        let sensitivity_squared =
            Box::new(move |blt: &BufferedToeplitz| sensitivity_squared_blt(blt, n));
        Self {
            error_for_inv,
            sensitivity_squared,
            n,
            min_sep: 1,
            max_participations: 1,
            penalty_strength: 1e-8,
            penalty_multipliers: HashMap::new(),
            max_second_coef: 1.0,
            min_theta_gap: 1e-12,
        }
    }

    /// Construct a loss for min-separation participation.
    pub fn build_min_sep(
        n: usize,
        error: &str,
        min_sep: usize,
        max_participations: Option<usize>,
    ) -> Self {
        let err_mode = error.to_lowercase();
        let error_for_inv: Box<dyn Fn(&BufferedToeplitz) -> f64 + Send + Sync> =
            if err_mode == "mean" {
                Box::new(move |inv_blt: &BufferedToeplitz| {
                    let coefs = inv_blt.toeplitz_coefs(n);
                    mean_error(None, Some(&coefs), Some(n), None, true)
                })
            } else {
                Box::new(move |inv_blt: &BufferedToeplitz| {
                    let coefs = inv_blt.toeplitz_coefs(n);
                    toeplitz_max_error(None, Some(&coefs), Some(n), None, true)
                })
            };

        let max_part =
            crate::sensitivity::minsep_true_max_participations(n, min_sep, max_participations);
        let sens_fn = Box::new(move |blt: &BufferedToeplitz| {
            minsep_sensitivity_squared_blt(blt, n, min_sep, max_participations)
        });

        Self {
            error_for_inv,
            sensitivity_squared: sens_fn,
            n,
            min_sep,
            max_participations: max_part,
            penalty_strength: 1e-8,
            penalty_multipliers: HashMap::new(),
            max_second_coef: 1.0,
            min_theta_gap: 1e-12,
        }
    }

    /// Compute penalties for BLT constraints.
    pub fn compute_penalties(
        &self,
        blt: &BufferedToeplitz,
        inv_blt: &BufferedToeplitz,
    ) -> HashMap<String, f64> {
        let mut penalties = HashMap::new();
        penalties.insert("buf_decay>0".to_string(), gt_zero_penalty(&blt.buf_decay));
        penalties.insert("buf_decay<1".to_string(), lt_one_penalty(&blt.buf_decay));
        penalties.insert(
            "output_scale>0".to_string(),
            gt_zero_penalty(&blt.output_scale),
        );
        penalties.insert(
            "inv_buf_decay>0".to_string(),
            gt_zero_penalty(&inv_blt.buf_decay),
        );
        penalties.insert(
            "inv_buf_decay<1".to_string(),
            lt_one_penalty(&inv_blt.buf_decay),
        );
        penalties.insert(
            "inv_output_scale<0".to_string(),
            lt_zero_penalty(&inv_blt.output_scale),
        );

        if blt.buf_decay.len() > 1 {
            let gap = min_buf_decay_gap(&blt.buf_decay) - self.min_theta_gap;
            let inv_gap = min_buf_decay_gap(&inv_blt.buf_decay) - self.min_theta_gap;
            penalties.insert(
                "theta_gap".to_string(),
                gt_zero_penalty(&[gap]) + gt_zero_penalty(&[inv_gap]),
            );
        }

        let second_coef: f64 = blt.output_scale.iter().sum();
        penalties.insert(
            "second_coef".to_string(),
            lt_penalty(second_coef, self.max_second_coef),
        );
        penalties.insert(
            "pillutla_score".to_string(),
            lt_one_penalty(&[blt.pillutla_score()]),
        );
        penalties
    }

    /// Compute the penalized loss for a BLT and its inverse.
    pub fn penalized_loss(
        &self,
        blt: &BufferedToeplitz,
        inv_blt: &BufferedToeplitz,
        normalize_by_approx_optimal_loss: bool,
    ) -> f64 {
        let error = (self.error_for_inv)(inv_blt);
        let sens = (self.sensitivity_squared)(blt);
        let penalties = self.compute_penalties(blt, inv_blt);

        let mut total_penalty = 0.0;
        for (name, value) in penalties {
            let multiplier = self.penalty_multipliers.get(&name).copied().unwrap_or(1.0);
            if multiplier != 0.0 {
                total_penalty += multiplier * value;
            }
        }
        total_penalty *= self.penalty_strength;

        let mut loss = error * sens;
        if normalize_by_approx_optimal_loss {
            let approx_optimal =
                self.max_participations as f64 + (1.0 + (self.n as f64).ln() / PI).powi(2);
            loss /= approx_optimal;
        }
        loss + total_penalty
    }

    /// Compute the (unpenalized) loss for a BLT.
    pub fn loss(&self, blt: &BufferedToeplitz) -> f64 {
        let inv = blt.inverse();
        let error = (self.error_for_inv)(&inv);
        let sens = (self.sensitivity_squared)(blt);
        error * sens
    }
}

/// Parameterization for BLT optimization.
pub enum Parameterization {
    /// Optimize directly over (buf_decay, output_scale).
    StrategyBlt {
        /// Number of buffers in the BLT parameterization.
        num_buffers: usize,
    },
    /// Optimize a pair of buf_decay parameters for C and C^{-1}.
    BufDecayPair {
        /// Number of buffers in the BLT parameterization.
        num_buffers: usize,
    },
}

impl Parameterization {
    /// Strategy BLT parameterization.
    pub fn strategy_blt(num_buffers: usize) -> Self {
        Self::StrategyBlt { num_buffers }
    }

    /// Buf-decay-pair parameterization.
    pub fn buf_decay_pair(num_buffers: usize) -> Self {
        Self::BufDecayPair { num_buffers }
    }

    /// Extract parameters from a BLT.
    pub fn params_from_blt(&self, blt: &BufferedToeplitz) -> Vec<f64> {
        match *self {
            Parameterization::StrategyBlt { .. } => blt
                .buf_decay
                .iter()
                .chain(blt.output_scale.iter())
                .copied()
                .collect(),
            Parameterization::BufDecayPair { .. } => {
                let inv = blt.inverse();
                blt.buf_decay
                    .iter()
                    .chain(inv.buf_decay.iter())
                    .copied()
                    .collect()
            }
        }
    }

    /// Construct BLT and inverse from parameters.
    pub fn blt_and_inverse_from_params(
        &self,
        params: &[f64],
    ) -> (BufferedToeplitz, BufferedToeplitz) {
        match *self {
            Parameterization::StrategyBlt { num_buffers } => {
                let mut buf_decay = vec![0.0; num_buffers];
                let mut output_scale = vec![0.0; num_buffers];
                let split = num_buffers.min(params.len());
                buf_decay[..split].copy_from_slice(&params[..split]);
                let remaining = params.len().saturating_sub(split);
                let take = remaining.min(num_buffers);
                if take > 0 {
                    output_scale[..take].copy_from_slice(&params[split..split + take]);
                }
                let blt = BufferedToeplitz::build(buf_decay, output_scale);
                let inv = blt.inverse();
                (blt, inv)
            }
            Parameterization::BufDecayPair { num_buffers } => {
                let split = num_buffers.min(params.len());
                let mut theta = vec![0.0; num_buffers];
                let mut theta_hat = vec![0.0; num_buffers];
                theta[..split].copy_from_slice(&params[..split]);
                let remaining = params.len().saturating_sub(split);
                let take = remaining.min(num_buffers);
                if take > 0 {
                    theta_hat[..take].copy_from_slice(&params[split..split + take]);
                }
                blt_pair_from_theta_pair(&theta, &theta_hat)
            }
        }
    }
}

/// Initialize a BLT for optimization.
pub fn get_init_blt(num_buffers: usize, init_blt: Option<BufferedToeplitz>) -> BufferedToeplitz {
    let mut blt = if let Some(init) = init_blt {
        init
    } else if num_buffers == 0 {
        BufferedToeplitz::build(Vec::new(), Vec::new())
    } else {
        BufferedToeplitz::from_rational_approx_to_sqrt_x(
            num_buffers,
            1.0 - 1e-6,
            Some(1.0 - 1e-6),
            1.0,
            -1,
        )
    };

    if blt.buf_decay.len() != num_buffers {
        blt = BufferedToeplitz::build(
            blt.buf_decay.iter().cloned().take(num_buffers).collect(),
            blt.output_scale.iter().cloned().take(num_buffers).collect(),
        );
    }
    blt
}

/// Optimize a BLT loss using the provided parameterization.
pub fn optimize_loss(
    loss_fn: &LossFn,
    num_buffers: usize,
    init_blt: Option<BufferedToeplitz>,
    parameterization: Option<Parameterization>,
    max_optimizer_steps: usize,
    step_size: f64,
) -> (BufferedToeplitz, f64) {
    let parameterization = match parameterization {
        Some(Parameterization::StrategyBlt { .. }) => Parameterization::StrategyBlt { num_buffers },
        Some(Parameterization::BufDecayPair { .. }) => {
            Parameterization::BufDecayPair { num_buffers }
        }
        None => Parameterization::BufDecayPair { num_buffers },
    };
    let init = get_init_blt(num_buffers, init_blt);
    let params = parameterization.params_from_blt(&init);
    if params.is_empty() {
        let loss = loss_fn.loss(&init);
        return (init, loss);
    }
    let loss = |p: &[f64]| {
        let (blt, inv) = parameterization.blt_and_inverse_from_params(p);
        loss_fn.penalized_loss(&blt, &inv, true)
    };
    let optimized = crate::optimization::optimize_projected(
        loss,
        params,
        max_optimizer_steps,
        step_size,
        |_| {},
        |_| false,
    );
    let (blt, _) = parameterization.blt_and_inverse_from_params(&optimized);
    let loss = loss_fn.loss(&blt);
    (blt, loss)
}

fn optimize_increasing_nbuf(
    mut opt_fn: impl FnMut(usize) -> (BufferedToeplitz, f64),
    min_buffers: usize,
    max_buffers: usize,
    rtol: f64,
) -> BufferedToeplitz {
    let (mut best_blt, mut best_loss) = opt_fn(min_buffers);
    for nbuf in (min_buffers + 1)..=max_buffers {
        let (candidate, loss) = opt_fn(nbuf);
        if rtol * loss < best_loss {
            best_blt = candidate;
            best_loss = loss;
        } else {
            break;
        }
    }
    best_blt
}

/// Optimize a BLT using a dynamically chosen number of buffers.
pub fn optimize(
    n: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    error: &str,
    min_buffers: usize,
    max_buffers: usize,
    rtol: f64,
    max_optimizer_steps: usize,
    step_size: f64,
) -> BufferedToeplitz {
    let loss_fn = if min_sep == 1 && max_participations.unwrap_or(1) == 1 && error == "max" {
        LossFn::build_closed_form_single_participation(n)
    } else {
        LossFn::build_min_sep(n, error, min_sep, max_participations)
    };
    let mut opt_fn = |nbuf| {
        optimize_loss(
            &loss_fn,
            nbuf,
            None,
            Some(Parameterization::buf_decay_pair(nbuf)),
            max_optimizer_steps,
            step_size,
        )
    };
    optimize_increasing_nbuf(&mut opt_fn, min_buffers, max_buffers, rtol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blt_toeplitz_has_unit_diagonal() {
        let blt = BufferedToeplitz::new(vec![0.9, 0.5], vec![0.1, 0.05]);
        let toeplitz = blt.to_toeplitz(5);
        let mat = toeplitz.materialize(5);
        for i in 0..5 {
            assert_eq!(mat[(i, i)], 1.0);
        }
    }

    #[test]
    fn canonicalize_sorts_descending() {
        let blt = BufferedToeplitz::build(vec![0.2, 0.9, 0.5], vec![1.0, 2.0, 3.0]);
        assert!(blt.buf_decay.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn inverse_matches_identity_small() {
        let blt = BufferedToeplitz::new(vec![0.7], vec![0.2]);
        let inv = blt.inverse();
        let n = 6;
        let a = blt.toeplitz_coefs(n);
        let b = inv.toeplitz_coefs(n);
        let prod = crate::toeplitz::multiply(&a, &b, Some(n));
        assert!((prod[0] - 1.0).abs() < 1e-6);
        for v in prod.iter().skip(1) {
            assert!(v.abs() < 1e-6);
        }
    }
}
