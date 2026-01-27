use jax_privacy::core::BatchSelectionStrategy;
use jax_privacy::experimental::BandMfExecutionPlanConfig;
use jax_privacy::JaxKey;

fn make_tensor(v: &[f64]) -> ndarray::ArrayD<f64> {
    ndarray::Array1::from_vec(v.to_vec()).into_dyn()
}

#[test]
fn execution_plan_end_to_end_step() {
    let cfg = BandMfExecutionPlanConfig {
        iterations: 3,
        num_bands: 2,
        epsilon: None,
        delta: 1e-6,
        noise_multiplier: Some(1.1),
        sampling_prob: 0.5,
        num_samples: 128,
        batch_size: 16,
        l2_clip_norm: 1.0,
        rescale_to_unit_norm: false,
        normalize_by: None,
        truncated_batch_size: None,
        partition_type: jax_privacy::core::PartitionType::EqualSplit,
        neighboring_relation: jax_privacy::core::NeighboringRelation::ReplaceSpecial,
        strategy_optimization_steps: 25,
        noise_seed: 42,
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
    assert_eq!(batches.len(), cfg.iterations);

    let mut state = None;
    for batch in batches {
        if batch.is_empty() {
            continue;
        }
        let examples = batch
            .into_iter()
            .map(|i| dataset[i as usize].clone())
            .collect::<Vec<_>>();

        let clipped = plan.clipped_aggregate(examples.clone());
        let init_state = state.take().unwrap_or_else(|| plan.init_state(&clipped));
        let (noisy, new_state) = plan.step(examples, init_state);
        state = Some(new_state);

        assert_eq!(noisy.shape(), clipped.shape());
        assert!(noisy.iter().all(|v| v.is_finite()));
    }
}
