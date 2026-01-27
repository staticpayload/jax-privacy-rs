use jax_privacy::{calibrate_dp_mechanism, dpsgd_event};

fn main() {
    let steps = 256u64;
    let q = 0.01;
    let delta = 1e-6;
    let target_eps = 3.0;

    let sigma = calibrate_dp_mechanism(
        |s| dpsgd_event(s, steps, q),
        target_eps,
        delta,
        0.1,
        4.0,
        1e-3,
        80,
    )
    .expect("calibrated sigma");

    let eps = dpsgd_event(sigma, steps, q).epsilon(delta);
    println!("calibrated sigma: {:.4}", sigma);
    println!("epsilon @ delta={delta}: {:.4}", eps);
}
