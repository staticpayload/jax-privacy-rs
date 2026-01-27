//! Batch sampling strategies for DP-SGD.

use rand::seq::SliceRandom;
use rand::Rng;

/// Index type that supports padding sentinels like `-1`.
pub type Index = i64;

/// Batch indices for a training step.
pub type BatchIndices = Vec<Index>;

/// Sample indices using Poisson sampling.
///
/// Each index is included independently with probability `q`.
pub fn poisson_sample<R: Rng>(n: usize, q: f64, rng: &mut R) -> BatchIndices {
    if n == 0 || !q.is_finite() || q <= 0.0 {
        return Vec::new();
    }

    let q = q.clamp(0.0, 1.0);
    let mut indices = Vec::new();
    for i in 0..n {
        if rng.gen_bool(q) {
            indices.push(i as Index);
        }
    }
    indices
}

/// Sample a fixed number of indices without replacement.
pub fn fixed_sample<R: Rng>(n: usize, batch_size: usize, rng: &mut R) -> BatchIndices {
    if n == 0 || batch_size == 0 {
        return Vec::new();
    }

    let batch_size = batch_size.min(n);
    let mut indices: Vec<Index> = (0..n as Index).collect();
    indices.shuffle(rng);
    indices.truncate(batch_size);
    indices
}

/// Pad indices to a multiple of `multiple` using `-1` sentinels.
pub fn pad_to_multiple_of(indices: &[Index], multiple: usize) -> BatchIndices {
    assert!(multiple > 0, "multiple must be positive");
    let curr_size = indices.len();
    let pad_size = (multiple - curr_size % multiple) % multiple;
    let mut out = Vec::with_capacity(curr_size + pad_size);
    out.extend_from_slice(indices);
    out.extend(std::iter::repeat(-1).take(pad_size));
    out
}

/// Compute a permutation compatible with microbatch early stopping.
///
/// This matches the semantics of JAX Privacy's Fortran-order reshape:
/// data is reshaped to `(num_microbatches, microbatch_size)` and then
/// processed along the first axis.
pub fn compute_early_stopping_order(
    batch_size: usize,
    microbatch_size: Option<usize>,
) -> Vec<usize> {
    if microbatch_size.is_none() {
        return (0..batch_size).collect();
    }
    let micro = microbatch_size.unwrap();
    assert!(micro > 0, "microbatch_size must be positive");
    assert!(
        batch_size % micro == 0,
        "batch_size must be divisible by microbatch_size"
    );

    let num_micro = batch_size / micro;
    let mut order = Vec::with_capacity(batch_size);
    for i in 0..micro {
        for j in 0..num_micro {
            order.push(j * micro + i);
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_poisson_sample_probability() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let n = 10_000;
        let q = 0.1;
        let samples = poisson_sample(n, q, &mut rng);
        let ratio = samples.len() as f64 / n as f64;
        assert!((ratio - q).abs() < 0.02);
    }

    #[test]
    fn test_fixed_sample_size() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let samples = fixed_sample(100, 10, &mut rng);
        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn test_pad_to_multiple() {
        let indices = vec![0, 1, 2, 3, 4];
        let padded = pad_to_multiple_of(&indices, 4);
        assert_eq!(padded.len(), 8);
        assert_eq!(padded[5], -1);
    }

    #[test]
    fn test_early_stopping_order() {
        let order = compute_early_stopping_order(10, Some(2));
        assert_eq!(order, vec![0, 2, 4, 6, 8, 1, 3, 5, 7, 9]);
    }
}
