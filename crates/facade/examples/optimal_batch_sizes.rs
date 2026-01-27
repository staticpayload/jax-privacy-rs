use jax_privacy::accounting::optimal_physical_batch_sizes;

fn main() {
    let batch_sizes = vec![16, 17, 31, 32, 64, 65];
    let compiled = optimal_physical_batch_sizes(&batch_sizes, 2);
    println!("Suggested compiled batch sizes: {:?}", compiled);
}
