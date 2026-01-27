//! Rényi Differential Privacy accounting.

/// RDP accountant for privacy composition.
#[derive(Clone, Debug)]
pub struct RdpAccountant {
    orders: Vec<f64>,
    rdp: Vec<f64>,
    log_factorials: Vec<f64>,
}

impl Default for RdpAccountant {
    fn default() -> Self {
        Self::new()
    }
}

impl RdpAccountant {
    /// Create an accountant with default Rényi orders.
    pub fn new() -> Self {
        Self::with_orders(default_orders())
    }

    /// Create an accountant with custom orders.
    pub fn with_orders(orders: Vec<f64>) -> Self {
        let mut orders: Vec<f64> = orders
            .into_iter()
            .filter(|&a| a.is_finite() && a > 1.0)
            .collect();
        orders.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        orders.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        if orders.is_empty() {
            orders = default_orders();
        }

        let max_alpha = max_order_ceil(&orders);
        let log_factorials = precompute_log_factorials(max_alpha);
        let rdp = vec![0.0; orders.len()];
        Self {
            orders,
            rdp,
            log_factorials,
        }
    }

    /// Record one DP-SGD step.
    pub fn step(&mut self, noise_mult: f64, q: f64) {
        if !noise_mult.is_finite() || noise_mult <= 0.0 {
            self.invalidate();
            return;
        }
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            self.invalidate();
            return;
        }
        if q == 0.0 {
            return;
        }

        for (i, &alpha) in self.orders.iter().enumerate() {
            let rdp_step = rdp_gaussian_subsampled(alpha, noise_mult, q, &self.log_factorials);
            if rdp_step.is_finite() {
                self.rdp[i] += rdp_step;
            } else {
                self.rdp[i] = f64::INFINITY;
            }
        }
    }

    /// Record multiple identical steps.
    pub fn steps(&mut self, noise_mult: f64, q: f64, n: usize) {
        if n == 0 {
            return;
        }
        if !noise_mult.is_finite() || noise_mult <= 0.0 {
            self.invalidate();
            return;
        }
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            self.invalidate();
            return;
        }
        if q == 0.0 {
            return;
        }

        let n_f = n as f64;
        for (i, &alpha) in self.orders.iter().enumerate() {
            let rdp_step = rdp_gaussian_subsampled(alpha, noise_mult, q, &self.log_factorials);
            if rdp_step.is_finite() {
                self.rdp[i] += rdp_step * n_f;
            } else {
                self.rdp[i] = f64::INFINITY;
            }
        }
    }

    /// Convert to (epsilon, delta)-DP.
    pub fn epsilon(&self, delta: f64) -> f64 {
        if !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
            return f64::INFINITY;
        }

        let log_delta_inv = (1.0 / delta).ln();
        let mut best = f64::INFINITY;

        for (&alpha, &rdp) in self.orders.iter().zip(self.rdp.iter()) {
            if !rdp.is_finite() {
                continue;
            }
            let eps = rdp + log_delta_inv / (alpha - 1.0);
            if eps < best {
                best = eps;
            }
        }

        best
    }

    /// Reset the accountant.
    pub fn reset(&mut self) {
        self.rdp.fill(0.0);
    }

    fn invalidate(&mut self) {
        self.rdp.fill(f64::INFINITY);
    }
}

/// Default Rényi orders mirroring the Python implementation.
///
/// Python uses:
/// - `linspace(1.01, 8, num=50)`
/// - `arange(8, 64)`
/// - `linspace(65, 512, num=10, dtype=int)`
pub fn default_orders() -> Vec<f64> {
    let mut orders = Vec::new();

    orders.extend(linspace(1.01, 8.0, 50));
    for a in 8..64 {
        orders.push(a as f64);
    }
    for a in linspace(65.0, 512.0, 10) {
        orders.push(a.round());
    }

    orders
}

fn linspace(start: f64, end: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let step = (end - start) / (num as f64 - 1.0);
    (0..num).map(|i| start + step * i as f64).collect()
}

fn max_order_ceil(orders: &[f64]) -> usize {
    orders
        .iter()
        .copied()
        .filter(|&a| a.is_finite() && a > 1.0)
        .map(|a| a.ceil() as usize)
        .max()
        .unwrap_or(0)
}

fn precompute_log_factorials(max_alpha: usize) -> Vec<f64> {
    let mut log_fact = vec![0.0; max_alpha.saturating_add(1)];
    for i in 1..log_fact.len() {
        log_fact[i] = log_fact[i - 1] + (i as f64).ln();
    }
    log_fact
}

/// RDP of subsampled Gaussian mechanism.
fn rdp_gaussian_subsampled(alpha: f64, sigma: f64, q: f64, log_fact: &[f64]) -> f64 {
    if alpha <= 1.0 || !alpha.is_finite() {
        return 0.0;
    }
    if !sigma.is_finite() || sigma <= 0.0 {
        return f64::INFINITY;
    }
    if q == 0.0 {
        return 0.0;
    }
    if q == 1.0 {
        return alpha / (2.0 * sigma * sigma);
    }

    if is_integer(alpha) {
        let a = alpha.round() as usize;
        let log_a = log_a_term(q, sigma, a, log_fact);
        return log_a / (alpha - 1.0);
    }

    let floor = alpha.floor();
    let ceil = floor + 1.0;
    let rdp_floor = rdp_gaussian_subsampled(floor, sigma, q, log_fact);
    let rdp_ceil = rdp_gaussian_subsampled(ceil, sigma, q, log_fact);
    let t = alpha - floor;
    rdp_floor * (1.0 - t) + rdp_ceil * t
}

fn is_integer(x: f64) -> bool {
    (x - x.round()).abs() < 1e-12
}

/// Compute log(A_alpha) for integer alpha.
fn log_a_term(q: f64, sigma: f64, alpha: usize, log_fact: &[f64]) -> f64 {
    let log_q = q.ln();
    let log_1mq = (1.0 - q).ln();
    let sigma_sq = sigma * sigma;

    assert!(
        alpha < log_fact.len(),
        "precomputed log factorials must cover alpha"
    );

    let mut log_sum = f64::NEG_INFINITY;

    for j in 0..=alpha {
        let j_f = j as f64;
        let alpha_f = alpha as f64;

        let log_binom = log_fact[alpha] - log_fact[j] - log_fact[alpha - j];
        let log_prob = j_f * log_q + (alpha_f - j_f) * log_1mq;
        let log_exp = j_f * (j_f - 1.0) / (2.0 * sigma_sq);

        let log_term = log_binom + log_prob + log_exp;
        log_sum = log_add_exp(log_sum, log_term);
    }

    log_sum
}

/// Numerically stable log(exp(a) + exp(b)).
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_step() {
        let mut acc = RdpAccountant::new();
        acc.step(1.0, 0.01);
        let eps = acc.epsilon(1e-5);
        assert!(eps.is_finite());
        assert!(eps > 0.0);
    }

    #[test]
    fn test_epsilon_increases() {
        let mut acc = RdpAccountant::new();
        acc.step(1.0, 0.01);
        let eps1 = acc.epsilon(1e-5);
        acc.step(1.0, 0.01);
        let eps2 = acc.epsilon(1e-5);
        assert!(eps2 > eps1);
    }

    #[test]
    fn default_orders_cover_large_alphas() {
        let orders = default_orders();
        let max = orders
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        assert!(max >= 512.0);
        assert!(orders.iter().any(|&a| (a - 1.01).abs() < 1e-9));
    }

    #[test]
    fn steps_matches_repeated_step() {
        let mut a = RdpAccountant::new();
        let mut b = RdpAccountant::new();

        for _ in 0..5 {
            a.step(1.3, 0.02);
        }
        b.steps(1.3, 0.02, 5);

        let eps_a = a.epsilon(1e-6);
        let eps_b = b.epsilon(1e-6);
        assert!((eps_a - eps_b).abs() < 1e-9);
    }
}
