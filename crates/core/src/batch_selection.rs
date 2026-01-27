//! Batch selection strategies.

use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::Binomial;

use crate::sampling::pad_to_multiple_of as pad_to_multiple_of_impl;
use crate::sampling::{compute_early_stopping_order, Index};

/// A batch of indices.
pub type Batch = Vec<Index>;

/// A user-level batch where each row belongs to the same user.
pub type UserBatch = Vec<Batch>;

/// Partitioning strategy for cyclic sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionType {
    /// Assign each example independently at random.
    Independent,
    /// Shuffle then split into equally sized groups, discarding remainder.
    EqualSplit,
}

/// A framework-agnostic batch selection interface.
pub trait BatchSelectionStrategy {
    /// Produce all batches for `num_examples` using the provided RNG.
    fn batches<R: Rng + ?Sized>(&self, num_examples: usize, rng: &mut R) -> Vec<Batch>;
}

fn independent_partition<R: Rng + ?Sized>(
    num_examples: usize,
    num_groups: usize,
    rng: &mut R,
) -> Vec<Batch> {
    let mut indices: Vec<Index> = (0..num_examples as Index).collect();
    indices.shuffle(rng);
    let mut groups = vec![Vec::new(); num_groups.max(1)];
    for idx in indices {
        let g = rng.gen_range(0..groups.len());
        groups[g].push(idx);
    }
    groups
}

fn equal_split_partition<R: Rng + ?Sized>(
    num_examples: usize,
    num_groups: usize,
    rng: &mut R,
) -> Vec<Batch> {
    let groups = num_groups.max(1);
    let mut indices: Vec<Index> = (0..num_examples as Index).collect();
    indices.shuffle(rng);
    let group_size = num_examples / groups;
    let mut out = Vec::with_capacity(groups);
    for g in 0..groups {
        let start = g * group_size;
        let end = start + group_size;
        out.push(indices[start..end].to_vec());
    }
    out
}

/// Split a global batch into fixed-size minibatches and pad the last.
pub fn split_and_pad_global_batch(
    indices: &[Index],
    minibatch_size: usize,
    microbatch_size: Option<usize>,
) -> Vec<Batch> {
    assert!(minibatch_size > 0, "minibatch_size must be positive");
    if indices.is_empty() {
        return vec![vec![-1; minibatch_size]];
    }

    let mut minibatches = Vec::new();
    let mut start = 0usize;
    while start < indices.len() {
        let end = (start + minibatch_size).min(indices.len());
        minibatches.push(indices[start..end].to_vec());
        start = end;
    }

    if let Some(prev) = minibatches.pop() {
        let mut last = vec![-1; minibatch_size];
        let copy_len = prev.len().min(minibatch_size);
        last[..copy_len].copy_from_slice(&prev[..copy_len]);
        if microbatch_size.is_some() {
            let order = compute_early_stopping_order(minibatch_size, microbatch_size);
            let reordered: Batch = order.into_iter().map(|i| last[i]).collect();
            minibatches.push(reordered);
        } else if prev.len() == minibatch_size {
            minibatches.push(prev);
        } else {
            minibatches.push(last);
        }
    }

    minibatches
}

/// Pad indices to a multiple of `multiple` using -1 sentinel values.
pub fn pad_to_multiple_of(indices: &[Index], multiple: usize) -> Batch {
    pad_to_multiple_of_impl(indices, multiple)
}

/// Split and pad a 2D batch of indices into fixed-size minibatches.
pub fn split_and_pad_global_batch_matrix(
    indices: &[Batch],
    minibatch_size: usize,
    microbatch_size: Option<usize>,
) -> Vec<UserBatch> {
    assert!(minibatch_size > 0, "minibatch_size must be positive");
    if indices.is_empty() {
        return vec![vec![vec![-1; 0]; minibatch_size]];
    }

    let row_len = indices[0].len();
    let mut minibatches: Vec<UserBatch> = Vec::new();
    let mut start = 0usize;
    while start < indices.len() {
        let end = (start + minibatch_size).min(indices.len());
        minibatches.push(indices[start..end].to_vec());
        start = end;
    }

    if let Some(prev) = minibatches.pop() {
        let mut last = vec![vec![-1; row_len]; minibatch_size];
        for (row_idx, row) in prev.iter().enumerate().take(minibatch_size) {
            let copy_len = row.len().min(row_len);
            last[row_idx][..copy_len].copy_from_slice(&row[..copy_len]);
        }
        if microbatch_size.is_some() {
            let order = compute_early_stopping_order(minibatch_size, microbatch_size);
            let reordered: UserBatch = order.into_iter().map(|i| last[i].clone()).collect();
            minibatches.push(reordered);
        } else if prev.len() == minibatch_size {
            minibatches.push(prev);
        } else {
            minibatches.push(last);
        }
    }

    minibatches
}

/// Cyclic Poisson sampling with optional truncation.
#[derive(Clone, Debug)]
pub struct CyclicPoissonSampling {
    /// Inclusion probability when eligible.
    pub sampling_prob: f64,
    /// Total number of iterations.
    pub iterations: usize,
    /// Optional hard cap on batch size.
    pub truncated_batch_size: Option<usize>,
    /// Cycle length (number of partitions).
    pub cycle_length: usize,
    /// How to partition examples into groups.
    pub partition_type: PartitionType,
}

impl CyclicPoissonSampling {
    fn partition<R: Rng + ?Sized>(&self, num_examples: usize, rng: &mut R) -> Vec<Batch> {
        match self.partition_type {
            PartitionType::Independent => {
                independent_partition(num_examples, self.cycle_length, rng)
            }
            PartitionType::EqualSplit => {
                equal_split_partition(num_examples, self.cycle_length, rng)
            }
        }
    }
}

impl BatchSelectionStrategy for CyclicPoissonSampling {
    fn batches<R: Rng + ?Sized>(&self, num_examples: usize, rng: &mut R) -> Vec<Batch> {
        let cycle = self.cycle_length.max(1);
        let mut partition = self.partition(num_examples, rng);
        if partition.len() != cycle {
            partition.resize(cycle, Vec::new());
        }

        let p = self.sampling_prob.clamp(0.0, 1.0);
        let mut out = Vec::with_capacity(self.iterations);

        for i in 0..self.iterations {
            let group = &partition[i % cycle];
            if group.is_empty() || p == 0.0 {
                out.push(Vec::new());
                continue;
            }

            let binom = Binomial::new(group.len() as u64, p).expect("valid binomial");
            let mut sample_size = binom.sample(rng) as usize;
            if let Some(limit) = self.truncated_batch_size {
                sample_size = sample_size.min(limit);
            }

            let mut candidates = group.clone();
            candidates.shuffle(rng);
            candidates.truncate(sample_size);
            out.push(candidates);
        }

        out
    }
}

/// Balls-in-bins sampling strategy.
#[derive(Clone, Debug)]
pub struct BallsInBinsSampling {
    /// Total number of iterations.
    pub iterations: usize,
    /// Number of bins / cycle length.
    pub cycle_length: usize,
}

impl BatchSelectionStrategy for BallsInBinsSampling {
    fn batches<R: Rng + ?Sized>(&self, num_examples: usize, rng: &mut R) -> Vec<Batch> {
        let cycle = self.cycle_length.max(1);
        let groups = independent_partition(num_examples, cycle, rng);
        let mut out = Vec::with_capacity(self.iterations);
        for i in 0..self.iterations {
            out.push(groups[i % cycle].clone());
        }
        out
    }
}

/// Apply a base strategy at the user level.
#[derive(Clone, Debug)]
pub struct UserSelectionStrategy<B: BatchSelectionStrategy> {
    /// Base strategy used to sample user IDs.
    pub base_strategy: B,
    /// Number of examples per user per batch.
    pub examples_per_user_per_batch: usize,
    /// Whether to shuffle each user's examples before cycling.
    pub shuffle_per_user: bool,
}

impl<B: BatchSelectionStrategy> UserSelectionStrategy<B> {
    /// Generate user-level batches.
    pub fn user_batches<R: Rng + ?Sized>(&self, user_ids: &[Index], rng: &mut R) -> Vec<UserBatch> {
        if user_ids.is_empty() {
            return Vec::new();
        }

        // Map users to the examples they own.
        let mut unique_users: Vec<Index> = user_ids.to_vec();
        unique_users.sort_unstable();
        unique_users.dedup();

        let num_users = unique_users.len();
        let mut owned: Vec<Vec<Index>> = vec![Vec::new(); num_users];
        for (example_idx, &uid) in user_ids.iter().enumerate() {
            if let Ok(pos) = unique_users.binary_search(&uid) {
                owned[pos].push(example_idx as Index);
            }
        }

        if self.shuffle_per_user {
            for examples in &mut owned {
                examples.shuffle(rng);
            }
        }

        let per_user = self.examples_per_user_per_batch.max(1);
        let mut cursors = vec![0usize; num_users];

        let user_batches = self.base_strategy.batches(num_users, rng);
        let mut out = Vec::with_capacity(user_batches.len());

        for user_batch in user_batches {
            let mut batch_rows = Vec::with_capacity(user_batch.len());
            for user_idx in user_batch {
                let u = user_idx as usize;
                let examples = &owned[u];
                if examples.is_empty() {
                    batch_rows.push(vec![-1; per_user]);
                    continue;
                }

                let mut row = Vec::with_capacity(per_user);
                for _ in 0..per_user {
                    let cursor = cursors[u] % examples.len();
                    row.push(examples[cursor]);
                    cursors[u] = cursors[u].wrapping_add(1);
                }
                batch_rows.push(row);
            }
            out.push(batch_rows);
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_split_and_pad() {
        let indices: Vec<Index> = (0..10).collect();
        let batches = split_and_pad_global_batch(&indices, 4, None);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[2].len(), 4);
        assert_eq!(batches[2][2], -1);
    }

    #[test]
    fn test_split_and_pad_matrix() {
        let indices: Vec<Batch> = vec![vec![0, 1], vec![2, 3], vec![4, 5]];
        let batches = split_and_pad_global_batch_matrix(&indices, 2, None);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[1].len(), 2);
        assert_eq!(batches[1][1], vec![-1, -1]);
    }

    #[test]
    fn test_split_and_pad_matrix_microbatch_reorder() {
        let indices: Vec<Batch> = vec![vec![0], vec![1], vec![2], vec![3]];
        let batches = split_and_pad_global_batch_matrix(&indices, 4, Some(2));
        assert_eq!(batches.len(), 1);
        let order: Vec<Index> = batches[0].iter().map(|row| row[0]).collect();
        assert_eq!(order, vec![0, 2, 1, 3]);
    }

    #[test]
    fn test_cyclic_poisson_runs() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let strat = CyclicPoissonSampling {
            sampling_prob: 0.25,
            iterations: 8,
            truncated_batch_size: None,
            cycle_length: 2,
            partition_type: PartitionType::EqualSplit,
        };
        let batches = strat.batches(100, &mut rng);
        assert_eq!(batches.len(), 8);
    }

    #[test]
    fn test_user_selection_shape() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let base = CyclicPoissonSampling {
            sampling_prob: 1.0,
            iterations: 3,
            truncated_batch_size: Some(2),
            cycle_length: 1,
            partition_type: PartitionType::EqualSplit,
        };
        let user_ids = vec![0, 0, 1, 1, 2, 2];
        let strat = UserSelectionStrategy {
            base_strategy: base,
            examples_per_user_per_batch: 2,
            shuffle_per_user: false,
        };
        let batches = strat.user_batches(&user_ids, &mut rng);
        assert_eq!(batches.len(), 3);
        assert!(!batches[0].is_empty());
        assert_eq!(batches[0][0].len(), 2);
    }
}
