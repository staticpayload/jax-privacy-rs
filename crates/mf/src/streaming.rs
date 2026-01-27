//! Streaming matrix abstractions.

use ndarray::{Array1, Array2};

/// A streaming lower-triangular matrix.
///
/// This trait models linear transforms that can be applied one element at a
/// time, where each output depends only on the current input and prior state.
pub trait StreamingMatrix: Clone {
    /// State maintained between steps.
    type State: Clone;

    /// Initialize state for a stream of length `n`.
    fn init(&self, n: usize) -> Self::State;

    /// Process the next input value and update the state.
    fn next(&self, x: f64, state: Self::State) -> (f64, Self::State);

    /// Get the coefficient at position `(row, col)`.
    fn get_coeff(&self, row: usize, col: usize) -> f64;

    /// Materialize as a dense n x n matrix (for debugging).
    fn materialize(&self, n: usize) -> Array2<f64> {
        let mut mat = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                mat[(i, j)] = self.get_coeff(i, j);
            }
        }
        mat
    }

    /// Compute row norms squared.
    fn row_norms_sq(&self, n: usize) -> Array1<f64> {
        let mut norms = Array1::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..=i {
                let c = self.get_coeff(i, j);
                sum += c * c;
            }
            norms[i] = sum;
        }
        norms
    }
}

/// Identity matrix (no transformation).
#[derive(Clone, Debug, Default)]
pub struct Identity;

impl StreamingMatrix for Identity {
    type State = ();

    fn init(&self, _n: usize) -> Self::State {}

    fn next(&self, x: f64, _state: Self::State) -> (f64, Self::State) {
        (x, ())
    }

    fn get_coeff(&self, row: usize, col: usize) -> f64 {
        if row == col {
            1.0
        } else {
            0.0
        }
    }
}

/// Prefix sum matrix (lower triangular of ones).
#[derive(Clone, Debug, Default)]
pub struct PrefixSum;

impl StreamingMatrix for PrefixSum {
    type State = f64;

    fn init(&self, _n: usize) -> Self::State {
        0.0
    }

    fn next(&self, x: f64, state: Self::State) -> (f64, Self::State) {
        let new_sum = state + x;
        (new_sum, new_sum)
    }

    fn get_coeff(&self, row: usize, col: usize) -> f64 {
        if col <= row {
            1.0
        } else {
            0.0
        }
    }
}

/// Diagonal streaming matrix.
#[derive(Clone, Debug)]
pub struct Diagonal {
    diag: Vec<f64>,
}

impl Diagonal {
    /// Create a diagonal matrix.
    pub fn new(diag: Vec<f64>) -> Self {
        Self { diag }
    }

    fn coeff_at(&self, i: usize) -> f64 {
        self.diag
            .get(i)
            .copied()
            .or_else(|| self.diag.last().copied())
            .unwrap_or(0.0)
    }
}

impl StreamingMatrix for Diagonal {
    type State = usize;

    fn init(&self, _n: usize) -> Self::State {
        0
    }

    fn next(&self, x: f64, idx: Self::State) -> (f64, Self::State) {
        (x * self.coeff_at(idx), idx + 1)
    }

    fn get_coeff(&self, row: usize, col: usize) -> f64 {
        if row == col {
            self.coeff_at(row)
        } else {
            0.0
        }
    }
}

/// A composed streaming matrix representing `A @ B`.
#[derive(Clone, Debug)]
pub struct Composed<A: StreamingMatrix, B: StreamingMatrix> {
    a: A,
    b: B,
}

impl<A: StreamingMatrix, B: StreamingMatrix> Composed<A, B> {
    fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: StreamingMatrix, B: StreamingMatrix> StreamingMatrix for Composed<A, B> {
    type State = (A::State, B::State);

    fn init(&self, n: usize) -> Self::State {
        (self.a.init(n), self.b.init(n))
    }

    fn next(&self, x: f64, state: Self::State) -> (f64, Self::State) {
        let (a_state, b_state) = state;
        let (inner, b_state) = self.b.next(x, b_state);
        let (outer, a_state) = self.a.next(inner, a_state);
        (outer, (a_state, b_state))
    }

    fn get_coeff(&self, row: usize, col: usize) -> f64 {
        if col > row {
            return 0.0;
        }
        // Materialize small prefixes to compute exact coefficients.
        // This is not optimized but is correct for debugging use.
        let n = row + 1;
        let a = self.a.materialize(n);
        let b = self.b.materialize(n);
        (a.dot(&b))[(row, col)]
    }
}

/// Compose two streaming matrices as `A @ B`.
pub fn multiply_streaming_matrices<A: StreamingMatrix, B: StreamingMatrix>(
    a: A,
    b: B,
) -> Composed<A, B> {
    Composed::new(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_materialize() {
        let id = Identity;
        let mat = id.materialize(3);
        assert_eq!(mat[(0, 0)], 1.0);
        assert_eq!(mat[(1, 0)], 0.0);
    }

    #[test]
    fn test_prefix_sum_streaming() {
        let ps = PrefixSum;
        let mut state = ps.init(3);
        let (y0, s) = ps.next(1.0, state);
        state = s;
        assert_eq!(y0, 1.0);
        let (y1, s) = ps.next(2.0, state);
        state = s;
        assert_eq!(y1, 3.0);
        let (y2, _) = ps.next(3.0, state);
        assert_eq!(y2, 6.0);
    }

    #[test]
    fn test_diagonal_coeffs() {
        let d = Diagonal::new(vec![2.0, 3.0]);
        assert_eq!(d.get_coeff(0, 0), 2.0);
        assert_eq!(d.get_coeff(1, 1), 3.0);
        // Clamp to last value.
        assert_eq!(d.get_coeff(5, 5), 3.0);
    }
}
