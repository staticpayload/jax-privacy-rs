//! Statistical auditing helpers inspired by the Python reference implementation.

use statrs::distribution::{Binomial, ContinuousCDF, DiscreteCDF, Normal};

fn logistic(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_positive() { 1.0 } else { 0.0 };
    }
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn binomial_sf(k: i64, n: u64, p: f64) -> f64 {
    if k < 0 {
        return 1.0;
    }
    if k as u64 >= n {
        return 0.0;
    }
    let p = p.clamp(0.0, 1.0);
    let binom = Binomial::new(p, n).unwrap_or_else(|e| panic!("invalid binomial params: {e}"));
    1.0 - binom.cdf(k as u64)
}

fn gaussian_delta_for_sigma(sigma: f64, eps: f64, normal: &Normal) -> f64 {
    if !sigma.is_finite() || sigma <= 0.0 || !eps.is_finite() {
        return 1.0;
    }
    let term1 = normal.cdf(-eps * sigma + 1.0 / (2.0 * sigma));
    let term2 = eps.exp() * normal.cdf(-eps * sigma - 1.0 / (2.0 * sigma));
    (term1 - term2).clamp(0.0, 1.0)
}

/// Approximate Gaussian mechanism sigma for the given (eps, delta).
pub fn sigma_for_gaussian_eps_delta(eps: f64, delta: f64) -> f64 {
    if !eps.is_finite() || eps <= 0.0 {
        return 0.0;
    }
    if !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
        return 0.0;
    }
    let normal = Normal::new(0.0, 1.0).expect("normal");

    let mut lo = 1e-6;
    let mut hi = 1.0;
    let mut delta_hi = gaussian_delta_for_sigma(hi, eps, &normal);
    let mut expansions = 0usize;
    while delta_hi > delta && hi < 1e6 && expansions < 80 {
        lo = hi;
        hi *= 2.0;
        delta_hi = gaussian_delta_for_sigma(hi, eps, &normal);
        expansions += 1;
    }

    if delta_hi > delta {
        return hi;
    }

    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        let d = gaussian_delta_for_sigma(mid, eps, &normal);
        if d > delta {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() <= 1e-6 {
            break;
        }
    }

    hi
}

/// Gaussian DP blow-up inverse helper.
pub struct GaussianDpBlowUpInverse {
    sigma: f64,
    normal: Normal,
}

impl GaussianDpBlowUpInverse {
    pub fn new(eps: f64, delta: f64) -> Self {
        let sigma = sigma_for_gaussian_eps_delta(eps, delta);
        let normal = Normal::new(0.0, 1.0).expect("normal");
        Self { sigma, normal }
    }

    pub fn apply(&self, x: f64) -> f64 {
        if self.sigma == 0.0 {
            return if (x - 1.0).abs() < 1e-12 { 1.0 } else { 0.0 };
        }
        let x = x.clamp(0.0, 1.0);
        let z = self.normal.inverse_cdf(x);
        self.normal.cdf(z - 1.0 / self.sigma)
    }
}

/// Compute the p-value for a one-shot canary audit.
///
/// This mirrors the structure of `_one_run_p_value` in the Python source.
pub fn one_run_p_value(m: usize, n_guess: usize, n_correct: usize, eps: f64, delta: f64) -> f64 {
    if n_guess == 0 || n_correct == 0 {
        return 1.0;
    }
    let n_correct = n_correct.min(n_guess);
    let q = logistic(eps).clamp(0.0, 1.0);
    let beta = binomial_sf(n_correct as i64 - 1, n_guess as u64, q);

    if delta <= 0.0 {
        return beta.clamp(0.0, 1.0);
    }

    let mut alpha = 0.0f64;
    for i in 1..=n_correct {
        let k = n_correct as i64 - i as i64 - 1;
        let cum = (binomial_sf(k, n_guess as u64, q) - beta).max(0.0);
        alpha = alpha.max(cum / i as f64);
    }

    let adjusted = beta + alpha * delta * 2.0 * m as f64;
    adjusted.clamp(0.0, 1.0)
}

/// Compute the largest epsilon falsified by a one-shot audit.
///
/// This uses a monotone bisection search in place of Brent's method.
pub fn epsilon_one_run(
    eps_lo: f64,
    m: usize,
    n_guess: usize,
    n_correct: usize,
    significance: f64,
    delta: f64,
) -> f64 {
    let eps_lo = eps_lo.max(0.0);
    let significance = significance.clamp(1e-12, 1.0 - 1e-12);

    if one_run_p_value(m, n_guess, n_correct, eps_lo, 0.0) > significance {
        return eps_lo;
    }

    let objective = |eps: f64| significance - one_run_p_value(m, n_guess, n_correct, eps, delta);
    if objective(eps_lo) <= 0.0 {
        return eps_lo;
    }

    let mut lo = eps_lo;
    let mut hi = eps_lo.max(1.0);
    if hi == lo {
        hi = lo + 1.0;
    }

    let mut hi_obj = objective(hi);
    let mut expansions = 0usize;
    while hi_obj > 0.0 && hi < 1e6 && expansions < 60 {
        lo = hi;
        hi *= 2.0;
        hi_obj = objective(hi);
        expansions += 1;
    }

    if hi_obj > 0.0 {
        return lo;
    }

    for _ in 0..80 {
        if (hi - lo).abs() <= 1e-6 {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if objective(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Compute epsilon bound using the f-DP audit variant.
pub fn epsilon_one_run_fdp(
    eps_lo: f64,
    m: usize,
    n_guess: usize,
    n_correct: usize,
    significance: f64,
    delta: f64,
) -> f64 {
    let eps_lo = eps_lo.max(0.0);
    let significance = significance.clamp(1e-12, 1.0 - 1e-12);
    let m = m.max(1) as f64;
    let n_guess = n_guess.max(1) as f64;
    let n_correct = n_correct.min(n_guess as usize) as f64;

    let audit_objective = |eps: f64| -> f64 {
        let blow_up = GaussianDpBlowUpInverse::new(eps, delta);
        let mut r = significance * n_correct / m;
        let mut h = significance * (n_guess - n_correct) / m;

        let mut i = n_correct as i64 - 1;
        while i >= 0 {
            let h_new = h.max(blow_up.apply(r));
            if (h_new - h).abs() < 1e-12 {
                break;
            }
            let denom = n_guess - i as f64;
            if denom <= 0.0 {
                break;
            }
            let r_new = (r + (i as f64 / denom) * (h_new - h)).min(1.0);
            h = h_new;
            r = r_new;
            i -= 1;
        }

        r + h - n_guess / m
    };

    if audit_objective(eps_lo) <= 0.0 {
        return eps_lo;
    }

    let mut lo = eps_lo;
    let mut hi = eps_lo.max(1.0);
    if hi == lo {
        hi = lo + 1.0;
    }

    let mut hi_obj = audit_objective(hi);
    let mut expansions = 0usize;
    while hi_obj > 0.0 && hi < 1e6 && expansions < 60 {
        lo = hi;
        hi *= 2.0;
        hi_obj = audit_objective(hi);
        expansions += 1;
    }

    if hi_obj > 0.0 {
        return lo;
    }

    for _ in 0..80 {
        if (hi - lo).abs() <= 1e-6 {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if audit_objective(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    lo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_increases_p_value() {
        let p0 = one_run_p_value(100, 50, 10, 1.0, 0.0);
        let p1 = one_run_p_value(100, 50, 10, 1.0, 1e-6);
        assert!(p1 >= p0);
    }

    #[test]
    fn epsilon_increases_with_more_correct_guesses() {
        let base = epsilon_one_run(0.1, 200, 100, 10, 0.05, 1e-6);
        let stronger = epsilon_one_run(0.1, 200, 100, 25, 0.05, 1e-6);
        assert!(stronger >= base);
    }

    #[test]
    fn returns_eps_lo_when_not_rejectable() {
        let eps = epsilon_one_run(0.5, 50, 50, 1, 1e-6, 0.0);
        assert!((eps - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sigma_for_gaussian_is_positive() {
        let sigma = sigma_for_gaussian_eps_delta(1.0, 1e-6);
        assert!(sigma.is_finite());
        assert!(sigma > 0.0);
    }

    #[test]
    fn epsilon_one_run_fdp_runs() {
        let eps = epsilon_one_run_fdp(0.1, 100, 40, 10, 0.05, 1e-6);
        assert!(eps.is_finite());
        assert!(eps >= 0.1);
    }
}
