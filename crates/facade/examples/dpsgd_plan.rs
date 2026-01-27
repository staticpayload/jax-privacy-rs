use jax_privacy::core::{BatchSelectionStrategy, GradientTransform};
use jax_privacy::experimental::DpsgdExecutionPlanConfig;
use jax_privacy::JaxKey;

fn main() {
    let cfg = DpsgdExecutionPlanConfig {
        iterations: 10,
        epsilon: None,
        delta: 1e-6,
        noise_multiplier: Some(1.1),
        sampling_prob: 0.4,
        num_samples: 512,
        batch_size: 64,
        l2_clip_norm: 1.0,
        rescale_to_unit_norm: false,
        normalize_by: None,
        truncated_batch_size: None,
        partition_type: jax_privacy::core::PartitionType::Independent,
        neighboring_relation: jax_privacy::core::NeighboringRelation::ReplaceSpecial,
        noise_seed: 0,
        noise_strategy: jax_privacy::NoiseStrategy::Default,
    };

    let plan = cfg.build();
    println!(
        "DP-SGD epsilon at delta={}: {:.3}",
        cfg.delta,
        plan.epsilon(cfg.delta)
    );

    let dataset = vec![
        ndarray::array![1.0, 0.0].into_dyn(),
        ndarray::array![0.0, 1.0].into_dyn(),
        ndarray::array![0.5, 0.5].into_dyn(),
        ndarray::array![-0.5, 0.5].into_dyn(),
    ];

    let mut rng = JaxKey::new(7).to_rng();
    let batches = plan
        .batch_selection_strategy
        .batches(dataset.len(), &mut rng);
    let batch = batches
        .into_iter()
        .find(|b| !b.is_empty())
        .unwrap_or_default();
    if batch.is_empty() {
        println!("No samples drawn; try increasing sampling_prob.");
        return;
    }

    let examples = batch
        .into_iter()
        .map(|i| dataset[i as usize].clone())
        .collect::<Vec<_>>();

    let clipped = plan.clipped_aggregate(examples);
    let state = plan.noise_addition_transform.init(&clipped);
    let (noisy, _) = plan.noise_addition_transform.update(&clipped, state);
    println!(
        "Noisy gradient L2 norm: {:.4}",
        jax_privacy::core::l2_norm(&noisy)
    );
}
