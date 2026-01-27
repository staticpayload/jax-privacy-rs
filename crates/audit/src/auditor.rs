//! Membership-inference style privacy auditing.

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Result of a membership inference audit.
#[derive(Clone, Debug)]
pub struct AuditResult {
    /// Empirical epsilon lower bound.
    pub epsilon_lower: f64,
    /// True positive rate.
    pub tpr: f64,
    /// False positive rate.
    pub fpr: f64,
    /// Area under the ROC curve.
    pub auc: f64,
    /// Number of trials.
    pub num_trials: usize,
}

impl AuditResult {
    /// Check if the empirical epsilon is within an expected bound.
    pub fn within_bound(&self, expected_epsilon: f64, margin: f64) -> bool {
        self.epsilon_lower <= expected_epsilon + margin
    }
}

/// Membership inference auditor.
#[derive(Clone, Debug)]
pub struct Auditor {
    /// Number of shadow models to train (conceptual placeholder).
    pub num_shadows: usize,
    /// Confidence level for bounds (conceptual placeholder).
    pub confidence: f64,
}

impl Default for Auditor {
    fn default() -> Self {
        Self {
            num_shadows: 100,
            confidence: 0.95,
        }
    }
}

impl Auditor {
    /// Create an auditor with custom parameters.
    pub fn new(num_shadows: usize, confidence: f64) -> Self {
        Self {
            num_shadows: num_shadows.max(10),
            confidence: confidence.clamp(0.5, 0.9999),
        }
    }

    /// Compute an epsilon lower bound from TPR/FPR.
    pub fn compute_epsilon_lower(&self, tpr: f64, fpr: f64) -> f64 {
        if fpr <= 0.0 || tpr <= 0.0 {
            return 0.0;
        }
        if fpr >= 1.0 || tpr >= 1.0 {
            return f64::INFINITY;
        }

        let eps1 = (tpr / fpr).ln();
        let eps2 = ((1.0 - fpr) / (1.0 - tpr)).ln();
        eps1.max(eps2).max(0.0)
    }

    /// Run audit with precomputed scores and a fixed threshold.
    pub fn audit_scores(
        &self,
        member_scores: &[f64],
        non_member_scores: &[f64],
        threshold: f64,
    ) -> AuditResult {
        let tp = member_scores.iter().filter(|&&s| s < threshold).count();
        let fp = non_member_scores.iter().filter(|&&s| s < threshold).count();

        let tpr = if member_scores.is_empty() {
            0.0
        } else {
            tp as f64 / member_scores.len() as f64
        };
        let fpr = if non_member_scores.is_empty() {
            0.0
        } else {
            fp as f64 / non_member_scores.len() as f64
        };

        let auc = compute_auc(member_scores, non_member_scores);
        let epsilon_lower = self.compute_epsilon_lower(tpr, fpr);

        AuditResult {
            epsilon_lower,
            tpr,
            fpr,
            auc,
            num_trials: member_scores.len() + non_member_scores.len(),
        }
    }

    /// Find the threshold that maximizes the epsilon estimate.
    pub fn find_optimal_threshold(
        &self,
        member_scores: &[f64],
        non_member_scores: &[f64],
    ) -> (f64, AuditResult) {
        let mut all_scores: Vec<f64> = member_scores
            .iter()
            .chain(non_member_scores.iter())
            .copied()
            .collect();
        all_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        all_scores.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        let mut best_threshold = f64::NEG_INFINITY;
        let mut best_result = self.audit_scores(member_scores, non_member_scores, best_threshold);

        for &threshold in &all_scores {
            let result = self.audit_scores(member_scores, non_member_scores, threshold);
            if result.epsilon_lower > best_result.epsilon_lower {
                best_result = result;
                best_threshold = threshold;
            }
        }

        (best_threshold, best_result)
    }
}

/// Compute area under ROC curve.
fn compute_auc(member_scores: &[f64], non_member_scores: &[f64]) -> f64 {
    if member_scores.is_empty() || non_member_scores.is_empty() {
        return 0.5;
    }

    let mut u = 0.0;
    for &m in member_scores {
        for &n in non_member_scores {
            if m < n {
                u += 1.0;
            } else if (m - n).abs() < 1e-12 {
                u += 0.5;
            }
        }
    }

    u / (member_scores.len() * non_member_scores.len()) as f64
}

/// Generate synthetic audit data for testing.
pub fn synthetic_audit_data<R: Rng>(
    rng: &mut R,
    epsilon: f64,
    num_samples: usize,
) -> (Vec<f64>, Vec<f64>) {
    let member_dist = Normal::new(0.0, 1.0).expect("valid normal");
    let non_member_dist = Normal::new(epsilon.min(5.0), 1.0).expect("valid normal");

    let members: Vec<f64> = (0..num_samples).map(|_| member_dist.sample(rng)).collect();
    let non_members: Vec<f64> = (0..num_samples)
        .map(|_| non_member_dist.sample(rng))
        .collect();

    (members, non_members)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_auc_random() {
        let members = vec![0.4, 0.5, 0.6, 0.45, 0.55];
        let non_members = vec![0.4, 0.5, 0.6, 0.45, 0.55];
        let auc = compute_auc(&members, &non_members);
        assert!((auc - 0.5).abs() < 0.2);
    }

    #[test]
    fn test_synthetic_audit() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (members, non_members) = synthetic_audit_data(&mut rng, 2.0, 1000);
        let auditor = Auditor::new(100, 0.95);
        let (_, result) = auditor.find_optimal_threshold(&members, &non_members);
        assert!(result.epsilon_lower > 0.5);
        assert!(result.auc > 0.6);
    }
}
