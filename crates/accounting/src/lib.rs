//! Privacy accounting for differentially private training.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod accountants;
pub mod analysis;
pub mod calibrate;
pub mod compilation;
pub mod event;
pub mod params;
pub mod pld;
pub mod rdp;

pub use accountants::{DpAccountantConfig, PldAccountantConfig, RdpAccountantConfig};
pub use analysis::{
    interleave_nm_and_bs, CachedExperimentAccountant, DpTrainingAccountant,
    DpsgdTrainingAccountant, DpsgdTrainingUserLevelAccountant, SingleReleaseTrainingAccountant,
};
pub use calibrate::{
    calibrate_batch_size, calibrate_dp_mechanism, calibrate_dp_mechanism_pld,
    calibrate_noise_multiplier, calibrate_num_updates,
};
pub use compilation::optimal_physical_batch_sizes;
pub use event::{
    amplified_bandmf_event, dpsgd_event, truncated_amplified_bandmf_event, truncated_dpsgd_event,
    DpEvent,
};
pub use params::{BatchingScaleSchedule, DpParams, Sampler, Schedule};
pub use pld::PldAccountant;
pub use rdp::RdpAccountant;

/// Common imports for privacy accounting.
pub mod prelude {
    pub use crate::{
        amplified_bandmf_event, calibrate_batch_size, calibrate_dp_mechanism,
        calibrate_dp_mechanism_pld, calibrate_noise_multiplier, calibrate_num_updates, dpsgd_event,
        interleave_nm_and_bs, optimal_physical_batch_sizes, truncated_amplified_bandmf_event,
        truncated_dpsgd_event, BatchingScaleSchedule, CachedExperimentAccountant,
        DpAccountantConfig, DpEvent, DpParams, DpTrainingAccountant, DpsgdTrainingAccountant,
        DpsgdTrainingUserLevelAccountant, PldAccountant, PldAccountantConfig, RdpAccountant,
        RdpAccountantConfig, Sampler, Schedule, SingleReleaseTrainingAccountant,
    };
}
