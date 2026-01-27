use jax_privacy::core::{BatchSelectionStrategy, GradientTransform};
use jax_privacy::experimental::DpsgdExecutionPlanConfig;
use jax_privacy::JaxKey;

fn make_tensor(v: &[f64]) -> ndarray::ArrayD<f64> {
    ndarray::Array1::from_vec(v.to_vec()).into_dyn()
}

#[test]
fn dpsgd_plan_step_shapes_and_finiteness() {
    let cfg = DpsgdExecutionPlanConfig {
        iterations: 4,
        epsilon: None,
        delta: 1e-6,
        noise_multiplier: Some(1.0),
        sampling_prob: 0.6,
        num_samples: 128,
        batch_size: 16,
        l2_clip_norm: 1.0,
        rescale_to_unit_norm: false,
        normalize_by: None,
        truncated_batch_size: Some(8),
        partition_type: jax_privacy::core::PartitionType::EqualSplit,
        neighboring_relation: jax_privacy::core::NeighboringRelation::ReplaceSpecial,
        noise_seed: 123,
        noise_strategy: jax_privacy::NoiseStrategy::Default,
    };

    let plan = cfg.build();

    let dataset = vec![
        make_tensor(&[1.0, 0.0, 0.0]),
        make_tensor(&[0.0, 1.0, 0.0]),
        make_tensor(&[0.0, 0.0, 1.0]),
        make_tensor(&[1.0, 1.0, 1.0]),
    ];

    let mut rng = JaxKey::new(7).to_rng();
    let batches = plan
        .batch_selection_strategy
        .batches(dataset.len(), &mut rng);

    let first = batches
        .into_iter()
        .find(|b| !b.is_empty())
        .unwrap_or_default();
    if first.is_empty() {
        return;
    }

    let examples = first
        .into_iter()
        .map(|i| dataset[i as usize].clone())
        .collect::<Vec<_>>();

    let clipped = plan.clipped_aggregate(examples);
    let state = plan.noise_addition_transform.init(&clipped);
    let (noisy, _) = plan.noise_addition_transform.update(&clipped, state);

    assert_eq!(noisy.shape(), clipped.shape());
    assert!(noisy.iter().all(|v| v.is_finite()));
}

#[test]
fn dpsgd_plan_truncated_with_epsilon_calibrates() {
    let cfg = DpsgdExecutionPlanConfig {
        iterations: 32,
        epsilon: Some(5.0),
        delta: 1e-6,
        noise_multiplier: None,
        sampling_prob: 0.4,
        num_samples: 10_000,
        batch_size: 512,
        l2_clip_norm: 1.0,
        rescale_to_unit_norm: false,
        normalize_by: None,
        truncated_batch_size: Some(128),
        partition_type: jax_privacy::core::PartitionType::EqualSplit,
        neighboring_relation: jax_privacy::core::NeighboringRelation::ReplaceSpecial,
        noise_seed: 999,
        noise_strategy: jax_privacy::NoiseStrategy::Default,
    };

    let plan = cfg.build();
    let eps = plan.epsilon(cfg.delta);
    assert!(eps.is_finite());
    assert!(eps <= cfg.epsilon.unwrap() + 1e-6);
}
