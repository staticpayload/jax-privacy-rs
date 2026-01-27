use jax_privacy::accounting::dpsgd_event;

fn main() {
    let event = dpsgd_event(3.0, 512, 0.1);
    let epsilon = event.epsilon(1e-6);
    println!("DP-SGD epsilon at delta=1e-6: {epsilon:.3}");
}
