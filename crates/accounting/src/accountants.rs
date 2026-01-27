//! Accountant configuration types mirroring the Python surface area.

use jax_privacy_core::clipping::NeighboringRelation;

use crate::{pld::PldAccountant, rdp::default_orders, RdpAccountant};

/// Configuration for constructing a privacy accountant with a fresh state.
pub trait DpAccountantConfig {
    /// Accountant type created by this config.
    type Accountant;

    /// Create a new accountant instance.
    fn create_accountant(&self) -> Self::Accountant;
}

/// Configuration for the RDP accountant.
#[derive(Clone, Debug)]
pub struct RdpAccountantConfig {
    /// Rényi orders to evaluate.
    pub orders: Vec<f64>,
}

impl Default for RdpAccountantConfig {
    fn default() -> Self {
        Self {
            orders: default_orders(),
        }
    }
}

impl RdpAccountantConfig {
    /// Create a config with explicit orders.
    pub fn new(orders: Vec<f64>) -> Self {
        Self { orders }
    }
}

impl DpAccountantConfig for RdpAccountantConfig {
    type Accountant = RdpAccountant;

    fn create_accountant(&self) -> Self::Accountant {
        RdpAccountant::with_orders(self.orders.clone())
    }
}

/// Configuration for the PLD accountant.
///
/// The Rust PLD accountant currently reuses the RDP implementation under the
/// hood. The discretization interval is retained for forward compatibility.
#[derive(Clone, Copy, Debug)]
pub struct PldAccountantConfig {
    /// Discretization interval for PLD values.
    pub value_discretization_interval: f64,
    /// Neighboring relation to analyze.
    pub neighboring_relation: NeighboringRelation,
}

impl Default for PldAccountantConfig {
    fn default() -> Self {
        Self {
            value_discretization_interval: 1e-4,
            neighboring_relation: NeighboringRelation::AddOrRemoveOne,
        }
    }
}

impl PldAccountantConfig {
    /// Create a config with an explicit discretization interval.
    pub fn new(value_discretization_interval: f64) -> Self {
        Self {
            value_discretization_interval,
            neighboring_relation: NeighboringRelation::AddOrRemoveOne,
        }
    }

    /// Set the neighboring relation for the PLD accountant.
    pub fn with_neighboring_relation(mut self, neighboring_relation: NeighboringRelation) -> Self {
        self.neighboring_relation = neighboring_relation;
        self
    }
}

impl DpAccountantConfig for PldAccountantConfig {
    type Accountant = PldAccountant;

    fn create_accountant(&self) -> Self::Accountant {
        PldAccountant::with_params(
            self.neighboring_relation,
            self.value_discretization_interval,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rdp_config_creates_finite_accountant() {
        let cfg = RdpAccountantConfig::default();
        let mut acc = cfg.create_accountant();
        acc.step(1.0, 0.01);
        assert!(acc.epsilon(1e-5).is_finite());
    }

    #[test]
    fn pld_config_creates_finite_accountant() {
        let cfg = PldAccountantConfig::default();
        let mut acc = cfg.create_accountant();
        acc.step(1.0, 0.01);
        assert!(acc.epsilon(1e-5).is_finite());
    }
}
