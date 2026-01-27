//! Lightweight optimization helpers for matrix-factorization strategies.
//!
//! The original JAX implementation relies on Optax + L-BFGS. To keep this
//! crate dependency-light and portable, we provide a simple finite-difference
//! gradient descent routine that is sufficient for small strategy searches.

/// Callback information emitted on each optimization step.
#[derive(Clone, Debug)]
pub struct CallbackArgs<'a> {
    /// Zero-based optimization step.
    pub step: usize,
    /// Loss value at the current step.
    pub loss: f64,
    /// Current parameter vector.
    pub params: &'a [f64],
}

fn sanitize_step_size(step_size: f64) -> f64 {
    if step_size.is_finite() && step_size > 0.0 {
        step_size
    } else {
        0.1
    }
}

/// Finite-difference gradient descent for scalar losses.
///
/// This mirrors the shape of the Python helper but intentionally trades off
/// sophistication for minimal dependencies and deterministic behavior.
pub fn optimize(
    loss_fn: impl Fn(&[f64]) -> f64,
    params: Vec<f64>,
    max_steps: usize,
    step_size: f64,
    callback: impl FnMut(CallbackArgs<'_>) -> bool,
) -> Vec<f64> {
    optimize_projected(loss_fn, params, max_steps, step_size, |_| {}, callback)
}

/// Finite-difference gradient descent with a projection step.
///
/// The projection is applied after each parameter update. This is useful for
/// constrained optimizations (e.g., non-negativity + normalization).
pub fn optimize_projected(
    loss_fn: impl Fn(&[f64]) -> f64,
    mut params: Vec<f64>,
    max_steps: usize,
    step_size: f64,
    mut project: impl FnMut(&mut [f64]),
    mut callback: impl FnMut(CallbackArgs<'_>) -> bool,
) -> Vec<f64> {
    let step_size = sanitize_step_size(step_size);

    project(&mut params);

    for step in 0..max_steps {
        let grad = finite_difference_grad(&loss_fn, &params, 1e-5);
        for (p, g) in params.iter_mut().zip(grad.iter()) {
            *p -= step_size * g;
        }
        project(&mut params);

        let loss = loss_fn(&params);
        if callback(CallbackArgs {
            step,
            loss,
            params: &params,
        }) {
            break;
        }
    }
    params
}

fn finite_difference_grad(loss_fn: &impl Fn(&[f64]) -> f64, params: &[f64], eps: f64) -> Vec<f64> {
    let base = loss_fn(params);
    let mut grad = Vec::with_capacity(params.len());
    for i in 0..params.len() {
        let mut perturbed = params.to_vec();
        perturbed[i] += eps;
        let loss_i = loss_fn(&perturbed);
        grad.push((loss_i - base) / eps);
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimize_reduces_quadratic() {
        let loss = |p: &[f64]| (p[0] - 2.0).powi(2) + (p[1] + 1.0).powi(2);
        let out = optimize(loss, vec![10.0, -10.0], 50, 0.2, |_| false);
        assert!(loss(&out) < loss(&[10.0, -10.0]));
    }

    #[test]
    fn projection_is_applied() {
        let loss = |p: &[f64]| p[0].powi(2);
        let project = |p: &mut [f64]| {
            if p[0].is_sign_negative() {
                p[0] = 0.0;
            }
        };
        let out = optimize_projected(loss, vec![-1.0], 5, 0.5, project, |_| false);
        assert!(out[0] >= 0.0);
    }
}
