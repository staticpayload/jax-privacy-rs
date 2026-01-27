use jax_privacy::PldAccountant;

fn main() {
    let mut acct = PldAccountant::new();
    acct.steps(1.0, 0.1, 100);
    let eps = acct.epsilon(1e-6);
    println!("PLD-style epsilon (RDP-backed) at delta=1e-6: {eps:.3}");
}
