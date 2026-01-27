//! Gradient transformation interfaces and noise privatizers.

use jax_privacy_prng::JaxKey;

use crate::clipping::clip_and_aggregate_with_padding;
use crate::noise::add_gaussian_noise_tree;
use crate::pytree::PyTree;
use crate::tensor::Tensor;

/// A stateful transformation applied to gradients.
pub trait GradientTransform<T: PyTree<Leaf = Tensor>> {
    /// State maintained across updates.
    type State;

    /// Initialize state based on model structure.
    fn init(&self, model: &T) -> Self::State;

    /// Update a gradient and return the transformed gradient plus new state.
    fn update(&self, grad: &T, state: Self::State) -> (T, Self::State);
}

/// Stateless Gaussian noise privatizer with a deterministic key schedule.
#[derive(Clone, Debug)]
pub struct GaussianPrivatizer {
    /// Standard deviation of the added noise.
    pub stddev: f64,
    /// Root PRNG key.
    pub key: JaxKey,
}

/// State for the Gaussian privatizer.
#[derive(Clone, Debug)]
pub struct GaussianState {
    key: JaxKey,
    step: u64,
}

/// Configuration for DP-SGD-style aggregation.
#[derive(Clone, Debug)]
pub struct DpSgdAggregator {
    /// L2 clip norm for per-example gradients.
    pub l2_clip_norm: f64,
    /// Whether to rescale to unit norm after clipping.
    pub rescale_to_unit_norm: bool,
    /// Normalization divisor for the aggregated gradient.
    pub normalize_by: f64,
    /// Noise multiplier (sigma / sensitivity).
    pub noise_multiplier: f64,
    /// Gradient accumulation steps (noise is scaled by sqrt of this value).
    pub accumulation_steps: usize,
    /// Whether to sanitize NaNs/Infs during clipping.
    pub nan_safe: bool,
    /// Optional microbatch size for padding-aware aggregation.
    pub microbatch_size: Option<usize>,
    /// Internal Gaussian privatizer.
    privatizer: GaussianPrivatizer,
}

/// State for DP-SGD aggregation.
#[derive(Clone, Debug)]
pub struct DpSgdState {
    noise: GaussianState,
}

impl DpSgdAggregator {
    /// Create a new DP-SGD aggregator.
    pub fn new(
        l2_clip_norm: f64,
        rescale_to_unit_norm: bool,
        normalize_by: f64,
        noise_multiplier: f64,
        nan_safe: bool,
        microbatch_size: Option<usize>,
        key: JaxKey,
    ) -> Self {
        let denom = normalize_by.max(1e-12);
        let stddev = noise_multiplier * l2_clip_norm / denom;
        let privatizer = GaussianPrivatizer::new(stddev, key);
        Self {
            l2_clip_norm,
            rescale_to_unit_norm,
            normalize_by,
            noise_multiplier,
            accumulation_steps: 1,
            nan_safe,
            microbatch_size,
            privatizer,
        }
    }

    /// Adjust the aggregator for gradient accumulation.
    pub fn with_accumulation_steps(mut self, steps: usize) -> Self {
        let steps = steps.max(1);
        self.accumulation_steps = steps;
        let denom = self.normalize_by.max(1e-12) * (steps as f64).sqrt();
        let stddev = self.noise_multiplier * self.l2_clip_norm / denom;
        self.privatizer.stddev = stddev;
        self
    }

    /// Initialize aggregation state.
    pub fn init_state(&self) -> DpSgdState {
        DpSgdState {
            noise: GaussianState {
                key: self.privatizer.key,
                step: 0,
            },
        }
    }

    /// Aggregate per-example gradients with clipping and noise.
    pub fn aggregate<T: PyTree<Leaf = Tensor>>(
        &self,
        per_example_grads: &[T],
        is_padding_example: Option<&[bool]>,
        state: DpSgdState,
    ) -> (T, DpSgdState, Option<Vec<f64>>) {
        let (clipped, norms) = clip_and_aggregate_with_padding(
            per_example_grads,
            self.l2_clip_norm,
            self.rescale_to_unit_norm,
            self.normalize_by,
            self.nan_safe,
            true,
            is_padding_example,
            self.microbatch_size,
        );

        let (noisy, noise_state) = self.privatizer.update(&clipped, state.noise);
        (noisy, DpSgdState { noise: noise_state }, norms)
    }
}

impl GaussianPrivatizer {
    /// Create a new Gaussian privatizer from a stddev and key.
    pub fn new(stddev: f64, key: JaxKey) -> Self {
        Self { stddev, key }
    }
}

impl<T> GradientTransform<T> for GaussianPrivatizer
where
    T: PyTree<Leaf = Tensor>,
{
    type State = GaussianState;

    fn init(&self, _model: &T) -> Self::State {
        GaussianState {
            key: self.key,
            step: 0,
        }
    }

    fn update(&self, grad: &T, mut state: Self::State) -> (T, Self::State) {
        let subkey = state.key.fold_in(state.step);
        let mut rng = subkey.to_rng();
        let noisy = add_gaussian_noise_tree(grad, self.stddev, &mut rng);
        state.step = state.step.saturating_add(1);
        (noisy, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gaussian_privatizer_deterministic() {
        let grad = array![0.0, 0.0, 0.0].into_dyn();
        let priv1 = GaussianPrivatizer::new(1.0, JaxKey::new(42));
        let state1 = priv1.init(&grad);
        let (noisy1, _) = priv1.update(&grad, state1);

        let priv2 = GaussianPrivatizer::new(1.0, JaxKey::new(42));
        let state2 = priv2.init(&grad);
        let (noisy2, _) = priv2.update(&grad, state2);

        assert_eq!(noisy1, noisy2);
    }

    #[test]
    fn dpsgd_aggregator_skips_padding() {
        let a = array![1.0, 0.0].into_dyn();
        let b = array![0.0, 1.0].into_dyn();
        let grads = vec![a.clone(), b.clone()];
        let padding = vec![false, true];

        let agg = DpSgdAggregator::new(10.0, false, 1.0, 0.0, true, None, JaxKey::new(1));
        let state = agg.init_state();
        let (noisy, _, norms) = agg.aggregate(&grads, Some(&padding), state);
        assert_eq!(noisy, a);
        let norms = norms.expect("norms");
        assert_eq!(norms[1], 0.0);
    }

    #[test]
    fn dpsgd_aggregator_accumulation_scales_noise() {
        let agg = DpSgdAggregator::new(2.0, false, 4.0, 1.5, true, None, JaxKey::new(0))
            .with_accumulation_steps(4);
        let expected = 1.5 * 2.0 / (4.0 * (4.0f64).sqrt());
        assert!((agg.privatizer.stddev - expected).abs() < 1e-10);
    }
}
