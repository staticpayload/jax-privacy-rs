//! Noise mechanisms for differential privacy.

use rand::Rng;
use rand_distr::{Distribution, Exp, Normal};

use crate::pytree::{map_leaves, PyTree};
use crate::tensor::{Scalar, Tensor};

/// Add Gaussian noise to a tensor in-place.
pub fn add_gaussian_noise<R: Rng>(tensor: &mut Tensor, sigma: f64, rng: &mut R) {
    if !sigma.is_finite() || sigma <= 0.0 || tensor.is_empty() {
        return;
    }

    let dist = match Normal::new(0.0, sigma) {
        Ok(d) => d,
        Err(_) => return,
    };

    tensor.mapv_inplace(|x| {
        if x.is_finite() {
            x + (dist.sample(rng) as Scalar)
        } else {
            x
        }
    });
}

/// Add Laplace noise to a tensor in-place.
pub fn add_laplace_noise<R: Rng>(tensor: &mut Tensor, scale: f64, rng: &mut R) {
    if !scale.is_finite() || scale <= 0.0 || tensor.is_empty() {
        return;
    }

    // Laplace noise can be sampled as the difference of two exponentials.
    let lambda = 1.0 / scale;
    let dist = match Exp::new(lambda) {
        Ok(d) => d,
        Err(_) => return,
    };

    tensor.mapv_inplace(|x| {
        if x.is_finite() {
            let n = dist.sample(rng) - dist.sample(rng);
            x + (n as Scalar)
        } else {
            x
        }
    });
}

/// Add i.i.d. Gaussian noise across a PyTree.
pub fn add_gaussian_noise_tree<R: Rng, T: PyTree<Leaf = Tensor>>(
    tree: &T,
    sigma: f64,
    rng: &mut R,
) -> T {
    map_leaves(tree, |leaf| {
        let mut out = leaf.clone();
        add_gaussian_noise(&mut out, sigma, rng);
        out
    })
}

/// Gaussian mechanism with sensitivity calibration.
#[derive(Clone, Debug)]
pub struct GaussianMechanism {
    /// Noise multiplier (sigma = noise_mult * sensitivity).
    pub noise_mult: f64,
    /// L2 sensitivity bound.
    pub sensitivity: f64,
}

impl GaussianMechanism {
    /// Create a new Gaussian mechanism.
    pub fn new(noise_mult: f64, sensitivity: f64) -> Self {
        Self {
            noise_mult,
            sensitivity,
        }
    }

    /// Get the noise standard deviation.
    pub fn sigma(&self) -> f64 {
        self.noise_mult * self.sensitivity
    }

    /// Add noise to a tensor.
    pub fn apply<R: Rng>(&self, tensor: &mut Tensor, rng: &mut R) {
        add_gaussian_noise(tensor, self.sigma(), rng);
    }
}

/// Laplace mechanism with sensitivity calibration.
#[derive(Clone, Debug)]
pub struct LaplaceMechanism {
    /// Epsilon parameter.
    pub epsilon: f64,
    /// L1 sensitivity bound.
    pub sensitivity: f64,
}

impl LaplaceMechanism {
    /// Create a new Laplace mechanism.
    pub fn new(epsilon: f64, sensitivity: f64) -> Self {
        Self {
            epsilon,
            sensitivity,
        }
    }

    /// Get the scale parameter.
    pub fn scale(&self) -> f64 {
        self.sensitivity / self.epsilon
    }

    /// Add noise to a tensor.
    pub fn apply<R: Rng>(&self, tensor: &mut Tensor, rng: &mut R) {
        add_laplace_noise(tensor, self.scale(), rng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_gaussian_deterministic() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut t1 = Array1::zeros(100).into_dyn();
        add_gaussian_noise(&mut t1, 1.0, &mut rng);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut t2 = Array1::zeros(100).into_dyn();
        add_gaussian_noise(&mut t2, 1.0, &mut rng);

        assert_eq!(t1, t2);
    }

    #[test]
    fn test_gaussian_statistics() {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let mut t = Array1::zeros(10_000).into_dyn();
        add_gaussian_noise(&mut t, 1.0, &mut rng);

        let mean: f64 = t.iter().map(|&x| x as f64).sum::<f64>() / t.len() as f64;
        let var: f64 = t.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / t.len() as f64;

        assert!(mean.abs() < 0.1);
        assert!((var - 1.0).abs() < 0.1);
    }
}
