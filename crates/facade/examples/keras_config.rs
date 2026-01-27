use jax_privacy::{DpKerasConfig, JaxKey};

fn main() {
    let cfg = DpKerasConfig {
        epsilon: 1000.0,
        delta: 1e-5,
        clipping_norm: 1.0,
        batch_size: 8,
        gradient_accumulation_steps: 4,
        train_steps: 10,
        train_size: 80,
        noise_multiplier: Some(50.0),
        rescale_to_unit_norm: true,
        microbatch_size: None,
        seed: None,
    };

    let agg = cfg.build_aggregator(JaxKey::new(0)).expect("aggregator");
    println!("Effective batch size: {}", cfg.effective_batch_size());
    println!("Accumulation steps: {}", agg.accumulation_steps);
}
