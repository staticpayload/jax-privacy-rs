use jax_privacy::core::{CyclicPoissonSampling, DpSgdAggregator, PartitionType, Tensor};
use jax_privacy::{DpSgdRunner, JaxKey};

fn make_tensor(v: &[f64]) -> Tensor {
    ndarray::Array1::from_vec(v.to_vec()).into_dyn()
}

fn main() {
    let dataset = vec![make_tensor(&[1.0, 0.0]), make_tensor(&[0.0, 1.0])];
    let strategy = CyclicPoissonSampling {
        sampling_prob: 1.0,
        iterations: 2,
        truncated_batch_size: None,
        cycle_length: 1,
        partition_type: PartitionType::EqualSplit,
    };
    let aggregator = DpSgdAggregator::new(1.0, false, 1.0, 0.5, true, None, JaxKey::new(0));
    let runner = DpSgdRunner {
        batch_strategy: strategy,
        aggregator,
    };

    let mut rng = JaxKey::new(123).to_rng();
    let outputs = runner.run(
        &dataset,
        &mut rng,
        || make_tensor(&[0.0, 0.0]),
        |x: &Tensor| x.clone(),
    );
    println!("Produced {} noisy gradients.", outputs.len());
}
