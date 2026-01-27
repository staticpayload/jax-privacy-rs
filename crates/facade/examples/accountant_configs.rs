use jax_privacy::{DpAccountantConfig, PldAccountantConfig, RdpAccountantConfig};

fn main() {
    let rdp_cfg = RdpAccountantConfig::default();
    let mut rdp = rdp_cfg.create_accountant();
    rdp.step(1.2, 0.01);
    println!("RDP epsilon @1e-6: {:.4}", rdp.epsilon(1e-6));

    let pld_cfg = PldAccountantConfig::default();
    let mut pld = pld_cfg.create_accountant();
    pld.step(1.2, 0.01);
    println!("PLD epsilon @1e-6: {:.4}", pld.epsilon(1e-6));
}
