use jax_privacy::{CanaryScoreAuditor, ThresholdStrategy};

fn main() {
    let held_in = vec![0.05, 0.10, 0.12, 0.20, 0.22, 0.25, 0.30, 0.35];
    let held_out = vec![0.55, 0.60, 0.62, 0.70, 0.75, 0.80, 0.85, 0.90];
    let auditor = CanaryScoreAuditor::new(held_in, held_out);

    let bonf = auditor.epsilon_clopper_pearson(0.05, 1e-6, true, ThresholdStrategy::Bonferroni);
    println!("Bonferroni: eps>= {:.4}", bonf);

    let split = auditor.epsilon_clopper_pearson(
        0.05,
        1e-6,
        true,
        ThresholdStrategy::Split {
            threshold_estimation_frac: 0.5,
            seed: Some(42),
        },
    );
    println!("Split:       eps>= {:.4}", split);
}
