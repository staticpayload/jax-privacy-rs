//! Empirical privacy auditing utilities.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod auditor;
mod canary;
mod stats;

pub use auditor::{synthetic_audit_data, AuditResult, Auditor};
pub use canary::{CanaryAuditResult, CanaryScoreAuditor, ThresholdStrategy};
pub use stats::{
    epsilon_one_run, epsilon_one_run_fdp, one_run_p_value, sigma_for_gaussian_eps_delta,
};

/// Common imports for auditing.
pub mod prelude {
    pub use crate::{
        epsilon_one_run, epsilon_one_run_fdp, one_run_p_value, sigma_for_gaussian_eps_delta,
        synthetic_audit_data, AuditResult, Auditor, CanaryAuditResult, CanaryScoreAuditor,
        ThresholdStrategy,
    };
}
