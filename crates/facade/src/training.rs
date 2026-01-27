//! Framework-agnostic training helpers for DP-SGD-style pipelines.

use rand::Rng;

use jax_privacy_core::batch_selection::BatchSelectionStrategy;
use jax_privacy_core::sampling::Index;
use jax_privacy_core::{compute_early_stopping_order, DpSgdAggregator, DpSgdState, PyTree, Tensor};

/// Dataset supporting indexed access.
pub trait IndexedDataset {
    /// Item type returned by the dataset.
    type Item: Clone;

    /// Number of items in the dataset.
    fn len(&self) -> usize;

    /// Fetch an item by index.
    fn get(&self, index: usize) -> Self::Item;
}

impl<T: Clone> IndexedDataset for Vec<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.len()
    }

    fn get(&self, index: usize) -> Self::Item {
        self[index].clone()
    }
}

/// Batch data plus padding mask.
#[derive(Clone, Debug)]
pub struct BatchData<T> {
    /// Batch items (including padding items).
    pub items: Vec<T>,
    /// Whether each position is padding.
    pub is_padding: Vec<bool>,
}

impl<T: Clone> BatchData<T> {
    /// Reorder items/masks for microbatch early stopping.
    pub fn reorder_for_microbatch(&self, microbatch_size: Option<usize>) -> Self {
        if microbatch_size.is_none() {
            return self.clone();
        }
        let order = compute_early_stopping_order(self.items.len(), microbatch_size);
        let items = order.iter().map(|&i| self.items[i].clone()).collect();
        let is_padding = order.iter().map(|&i| self.is_padding[i]).collect();
        Self { items, is_padding }
    }

    /// Split into microbatches (after reordering for early stopping).
    pub fn split_into_microbatches(&self, microbatch_size: usize) -> Vec<Self> {
        assert!(microbatch_size > 0, "microbatch_size must be positive");
        assert!(
            self.items.len() % microbatch_size == 0,
            "batch size must be divisible by microbatch_size"
        );

        let reordered = self.reorder_for_microbatch(Some(microbatch_size));
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < reordered.items.len() {
            let end = start + microbatch_size;
            out.push(Self {
                items: reordered.items[start..end].to_vec(),
                is_padding: reordered.is_padding[start..end].to_vec(),
            });
            start = end;
        }
        out
    }
}

/// Reorder indices to align padding for microbatching.
pub fn reorder_indices_for_microbatch(
    indices: &[Index],
    microbatch_size: Option<usize>,
) -> Vec<Index> {
    if microbatch_size.is_none() {
        return indices.to_vec();
    }
    let order = compute_early_stopping_order(indices.len(), microbatch_size);
    order.into_iter().map(|i| indices[i]).collect()
}

/// Gather a batch from an indexed dataset, inserting padding items for -1s.
pub fn gather_batch_with_padding<D, F>(
    dataset: &D,
    indices: &[Index],
    make_padding: F,
) -> BatchData<D::Item>
where
    D: IndexedDataset,
    F: Fn() -> D::Item,
{
    let mut items = Vec::with_capacity(indices.len());
    let mut is_padding = Vec::with_capacity(indices.len());

    for &idx in indices {
        if idx < 0 {
            items.push(make_padding());
            is_padding.push(true);
        } else {
            let u = idx as usize;
            assert!(u < dataset.len(), "index out of bounds");
            items.push(dataset.get(u));
            is_padding.push(false);
        }
    }

    BatchData { items, is_padding }
}

/// Compute a DP-SGD aggregate from a batch of indices and a per-example grad fn.
pub fn dpsgd_aggregate_batch<D, G, F, P>(
    dataset: &D,
    indices: &[Index],
    make_padding: P,
    grad_fn: F,
    aggregator: &DpSgdAggregator,
    state: DpSgdState,
) -> (G, DpSgdState, Option<Vec<f64>>)
where
    D: IndexedDataset,
    G: PyTree<Leaf = Tensor>,
    F: Fn(&D::Item) -> G,
    P: Fn() -> D::Item,
{
    let batch = gather_batch_with_padding(dataset, indices, make_padding);
    let per_example = batch
        .items
        .iter()
        .map(|item| grad_fn(item))
        .collect::<Vec<_>>();
    let (noisy, state, norms) = aggregator.aggregate(&per_example, Some(&batch.is_padding), state);
    (noisy, state, norms)
}

/// Simple DP-SGD runner that produces noisy gradients for each batch.
#[derive(Clone, Debug)]
pub struct DpSgdRunner<S> {
    /// Batch selection strategy.
    pub batch_strategy: S,
    /// DP-SGD aggregator.
    pub aggregator: DpSgdAggregator,
}

impl<S> DpSgdRunner<S>
where
    S: BatchSelectionStrategy,
{
    /// Execute the batch strategy over the dataset and return noisy gradients.
    pub fn run<D, G, F, P, R>(
        &self,
        dataset: &D,
        rng: &mut R,
        make_padding: P,
        grad_fn: F,
    ) -> Vec<G>
    where
        D: IndexedDataset,
        G: PyTree<Leaf = Tensor>,
        F: Fn(&D::Item) -> G,
        P: Fn() -> D::Item,
        R: Rng + ?Sized,
    {
        let batches = self.batch_strategy.batches(dataset.len(), rng);
        let mut state = self.aggregator.init_state();
        let mut outputs = Vec::new();

        for batch in batches {
            if batch.is_empty() {
                continue;
            }
            let (noisy, new_state, _norms) = dpsgd_aggregate_batch(
                dataset,
                &batch,
                &make_padding,
                &grad_fn,
                &self.aggregator,
                state,
            );
            state = new_state;
            outputs.push(noisy);
        }
        outputs
    }

    /// Execute the batch strategy and return both outputs and the final state.
    pub fn run_with_state<D, G, F, P, R>(
        &self,
        dataset: &D,
        rng: &mut R,
        make_padding: P,
        grad_fn: F,
        mut state: DpSgdState,
    ) -> (Vec<G>, DpSgdState)
    where
        D: IndexedDataset,
        G: PyTree<Leaf = Tensor>,
        F: Fn(&D::Item) -> G,
        P: Fn() -> D::Item,
        R: Rng + ?Sized,
    {
        let batches = self.batch_strategy.batches(dataset.len(), rng);
        let mut outputs = Vec::new();

        for batch in batches {
            if batch.is_empty() {
                continue;
            }
            let (noisy, new_state, _norms) = dpsgd_aggregate_batch(
                dataset,
                &batch,
                &make_padding,
                &grad_fn,
                &self.aggregator,
                state,
            );
            state = new_state;
            outputs.push(noisy);
        }
        (outputs, state)
    }

    /// Execute the DP-SGD loop and call a user-provided update function.
    pub fn run_with_update<D, G, F, P, R, U>(
        &self,
        dataset: &D,
        rng: &mut R,
        make_padding: P,
        grad_fn: F,
        mut update_fn: U,
    ) -> DpSgdState
    where
        D: IndexedDataset,
        G: PyTree<Leaf = Tensor>,
        F: Fn(&D::Item) -> G,
        P: Fn() -> D::Item,
        R: Rng + ?Sized,
        U: FnMut(&G),
    {
        let batches = self.batch_strategy.batches(dataset.len(), rng);
        let mut state = self.aggregator.init_state();

        for batch in batches {
            if batch.is_empty() {
                continue;
            }
            let (noisy, new_state, _norms) = dpsgd_aggregate_batch(
                dataset,
                &batch,
                &make_padding,
                &grad_fn,
                &self.aggregator,
                state,
            );
            update_fn(&noisy);
            state = new_state;
        }
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jax_privacy_prng::JaxKey;

    fn make_tensor(v: &[f64]) -> Tensor {
        ndarray::Array1::from_vec(v.to_vec()).into_dyn()
    }

    #[test]
    fn gather_batch_inserts_padding() {
        let dataset = vec![1.0, 2.0, 3.0];
        let indices = vec![0, -1, 2];
        let batch = gather_batch_with_padding(&dataset, &indices, || 0.0);
        assert_eq!(batch.items, vec![1.0, 0.0, 3.0]);
        assert_eq!(batch.is_padding, vec![false, true, false]);
    }

    #[test]
    fn batch_reorder_for_microbatch_matches_ordering() {
        let batch = BatchData {
            items: vec![1, 2, 3, 4],
            is_padding: vec![false, false, true, true],
        };
        let reordered = batch.reorder_for_microbatch(Some(2));
        assert_eq!(reordered.items, vec![1, 3, 2, 4]);
        assert_eq!(reordered.is_padding, vec![false, true, false, true]);
    }

    #[test]
    fn split_into_microbatches_preserves_padding() {
        let batch = BatchData {
            items: vec![1, 2, 3, 4],
            is_padding: vec![false, false, true, true],
        };
        let micro = batch.split_into_microbatches(2);
        assert_eq!(micro.len(), 2);
        assert_eq!(micro[0].items, vec![1, 3]);
        assert_eq!(micro[0].is_padding, vec![false, true]);
        assert_eq!(micro[1].items, vec![2, 4]);
        assert_eq!(micro[1].is_padding, vec![false, true]);
    }

    #[test]
    fn dpsgd_aggregate_batch_skips_padding() {
        let dataset = vec![make_tensor(&[1.0, 0.0]), make_tensor(&[2.0, 2.0])];
        let indices = vec![0, -1, 1];
        let agg = DpSgdAggregator::new(10.0, false, 1.0, 0.0, true, None, JaxKey::new(7));
        let state = agg.init_state();
        let (out, _state, norms) = dpsgd_aggregate_batch(
            &dataset,
            &indices,
            || make_tensor(&[0.0, 0.0]),
            |item| item.clone(),
            &agg,
            state,
        );
        let expected = &dataset[0] + &dataset[1];
        assert_eq!(out, expected);
        let norms = norms.expect("norms");
        assert_eq!(norms.len(), indices.len());
        assert_eq!(norms[1], 0.0);
    }

    #[test]
    fn dpsgd_runner_produces_outputs() {
        let dataset = vec![make_tensor(&[1.0, 0.0]), make_tensor(&[2.0, 2.0])];
        let strategy = jax_privacy_core::CyclicPoissonSampling {
            sampling_prob: 1.0,
            iterations: 2,
            truncated_batch_size: None,
            cycle_length: 1,
            partition_type: jax_privacy_core::PartitionType::EqualSplit,
        };
        let agg = DpSgdAggregator::new(10.0, false, 1.0, 0.0, true, None, JaxKey::new(5));
        let runner = DpSgdRunner {
            batch_strategy: strategy,
            aggregator: agg,
        };
        let mut rng = JaxKey::new(1).to_rng();
        let outputs = runner.run(
            &dataset,
            &mut rng,
            || make_tensor(&[0.0, 0.0]),
            |x| x.clone(),
        );
        assert!(!outputs.is_empty());
        for out in outputs {
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn dpsgd_runner_update_hook_runs() {
        let dataset = vec![make_tensor(&[1.0, 0.0]), make_tensor(&[2.0, 2.0])];
        let strategy = jax_privacy_core::CyclicPoissonSampling {
            sampling_prob: 1.0,
            iterations: 1,
            truncated_batch_size: None,
            cycle_length: 1,
            partition_type: jax_privacy_core::PartitionType::EqualSplit,
        };
        let agg = DpSgdAggregator::new(10.0, false, 1.0, 0.0, true, None, JaxKey::new(3));
        let runner = DpSgdRunner {
            batch_strategy: strategy,
            aggregator: agg,
        };
        let mut rng = JaxKey::new(2).to_rng();
        let mut seen = 0usize;
        let _state = runner.run_with_update(
            &dataset,
            &mut rng,
            || make_tensor(&[0.0, 0.0]),
            |x| x.clone(),
            |_grad| {
                seen += 1;
            },
        );
        assert!(seen > 0);
    }

    #[test]
    fn dpsgd_runner_returns_state() {
        let dataset = vec![make_tensor(&[1.0, 0.0]), make_tensor(&[2.0, 2.0])];
        let strategy = jax_privacy_core::CyclicPoissonSampling {
            sampling_prob: 1.0,
            iterations: 1,
            truncated_batch_size: None,
            cycle_length: 1,
            partition_type: jax_privacy_core::PartitionType::EqualSplit,
        };
        let agg = DpSgdAggregator::new(10.0, false, 1.0, 0.0, true, None, JaxKey::new(3));
        let runner = DpSgdRunner {
            batch_strategy: strategy,
            aggregator: agg,
        };
        let mut rng = JaxKey::new(2).to_rng();
        let state = runner.aggregator.init_state();
        let (outputs, _state) = runner.run_with_state(
            &dataset,
            &mut rng,
            || make_tensor(&[0.0, 0.0]),
            |x| x.clone(),
            state,
        );
        assert!(!outputs.is_empty());
    }
}
