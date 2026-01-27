use jax_privacy::core::GradientTransform;
use jax_privacy::{gaussian_privatizer, JaxKey, NoiseStrategy};

fn main() {
    let grad = ndarray::array![0.0, 0.0, 0.0].into_dyn();
    let privatizer = gaussian_privatizer(1.0, JaxKey::new(42), NoiseStrategy::Default);
    let state = privatizer.init(&grad);
    let (noisy, _) = privatizer.update(&grad, state);

    println!("Noisy grad norm: {:.4}", jax_privacy::core::l2_norm(&noisy));
}
