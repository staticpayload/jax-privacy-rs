//! Error types for differential privacy operations.

/// Errors that can occur during DP operations.
#[derive(Debug, thiserror::Error)]
pub enum DpError {
    /// Privacy budget has been exhausted.
    #[error("privacy budget exhausted (eps={eps:.4}, delta={delta:.2e})")]
    PrivacyBudgetExhausted {
        /// Current epsilon value.
        eps: f64,
        /// Current delta value.
        delta: f64,
    },

    /// Invalid parameter provided.
    #[error("invalid parameter: {msg}")]
    InvalidParameters {
        /// Human-readable error description.
        msg: String,
    },

    /// Numerical computation error.
    #[error("numerical error: {msg}")]
    NumericalError {
        /// Human-readable error description.
        msg: String,
    },

    /// Configuration error.
    #[error("configuration error: {msg}")]
    ConfigError {
        /// Human-readable error description.
        msg: String,
    },

    /// Unsupported feature or configuration.
    #[error("unsupported feature: {msg}")]
    UnsupportedFeature {
        /// Human-readable error description.
        msg: String,
    },
}

/// Result type for DP operations.
pub type Result<T> = std::result::Result<T, DpError>;

impl DpError {
    /// Create an invalid parameter error.
    pub fn invalid<S: Into<String>>(msg: S) -> Self {
        Self::InvalidParameters { msg: msg.into() }
    }

    /// Create a numerical error.
    pub fn numerical<S: Into<String>>(msg: S) -> Self {
        Self::NumericalError { msg: msg.into() }
    }

    /// Create a configuration error.
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::ConfigError { msg: msg.into() }
    }

    /// Create an unsupported feature error.
    pub fn unsupported<S: Into<String>>(msg: S) -> Self {
        Self::UnsupportedFeature { msg: msg.into() }
    }
}
