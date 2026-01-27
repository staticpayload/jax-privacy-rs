//! Streaming matrix factorization utilities.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod banded;
pub mod buffered_toeplitz;
pub mod checks;
pub mod dense;
pub mod noise;
pub mod optimization;
pub mod sensitivity;
pub mod streaming;
pub mod test_utils;
pub mod toeplitz;

pub use banded::{
    last_error as banded_last_error, max_error as banded_max_error,
    mean_error as banded_mean_error,
    minsep_sensitivity_squared as banded_minsep_sensitivity_squared, optimize as banded_optimize,
    per_query_error as banded_per_query_error, ColumnNormalizedBanded, DenseState,
    DenseStreamingMatrix,
};
pub use buffered_toeplitz::{
    blt_pair_from_theta_pair, get_init_blt, iteration_error as blt_iteration_error,
    limit_max_error as blt_limit_max_error, limit_max_loss as blt_limit_max_loss,
    max_error as blt_max_error, max_error_blt, max_loss as blt_max_loss, max_loss_blt,
    mean_error_blt, min_buf_decay_gap, minsep_sensitivity_squared_blt, optimize as blt_optimize,
    optimize_loss, sensitivity_squared_blt, sensitivity_squared_blt_limit, BufferedToeplitz,
    LossFn, Parameterization, StreamingMatrixBuilder as BufferedStreamingMatrixBuilder,
};
pub use checks::{
    assert_lower_triangular, column_norms, is_lower_toeplitz, is_lower_triangular,
    normalize_columns,
};
pub use dense::{
    column_normalize, inverse_lower_triangular, optimize as dense_optimize, toeplitz_dense,
};
pub use noise::{
    gaussian_privatizer, matrix_factorization_privatizer, matrix_factorization_privatizer_dense,
    MatrixFactorizationPrivatizer, NoiseStrategy,
};
pub use optimization::{optimize, optimize_projected, CallbackArgs};
pub use sensitivity::{
    banded_lower_triangular_mask, banded_symmetric_mask, fixed_epoch_sensitivity,
    fixed_epoch_sensitivity_for_x, get_min_sep_sensitivity_upper_bound,
    get_min_sep_sensitivity_upper_bound_for_x, get_sensitivity_banded,
    get_sensitivity_banded_for_x, max_participation_for_linear_fn, minsep_true_max_participations,
    single_participation_sensitivity,
};
pub use streaming::{multiply_streaming_matrices, Diagonal, Identity, PrefixSum, StreamingMatrix};
pub use toeplitz::{
    inverse_as_streaming_matrix, inverse_coef, materialize_lower_triangular, max_error, max_loss,
    mean_error, mean_loss, minsep_sensitivity_squared, multiply, optimal_max_error_noising_coefs,
    optimal_max_error_strategy_coefs, optimize_banded_toeplitz, pad_coefs_to_n, per_query_error,
    sensitivity_squared, solve_banded, Toeplitz, ToeplitzInverseStream,
};

/// Common imports for matrix-factorization components.
pub mod prelude {
    pub use crate::{
        assert_lower_triangular, banded_last_error, banded_lower_triangular_mask, banded_max_error,
        banded_mean_error, banded_minsep_sensitivity_squared, banded_optimize,
        banded_per_query_error, banded_symmetric_mask, blt_iteration_error, blt_limit_max_error,
        blt_limit_max_loss, blt_max_error, blt_max_loss, blt_optimize, blt_pair_from_theta_pair,
        column_normalize, column_norms, dense_optimize, fixed_epoch_sensitivity,
        fixed_epoch_sensitivity_for_x, gaussian_privatizer, get_init_blt,
        get_min_sep_sensitivity_upper_bound, get_min_sep_sensitivity_upper_bound_for_x,
        get_sensitivity_banded, get_sensitivity_banded_for_x, inverse_as_streaming_matrix,
        inverse_coef, inverse_lower_triangular, is_lower_toeplitz, is_lower_triangular,
        materialize_lower_triangular, matrix_factorization_privatizer,
        matrix_factorization_privatizer_dense, max_error, max_error_blt, max_loss, max_loss_blt,
        max_participation_for_linear_fn, mean_error, mean_error_blt, mean_loss, min_buf_decay_gap,
        minsep_sensitivity_squared, minsep_sensitivity_squared_blt, minsep_true_max_participations,
        multiply, multiply_streaming_matrices, optimal_max_error_noising_coefs,
        optimal_max_error_strategy_coefs, optimize, optimize_banded_toeplitz, optimize_loss,
        optimize_projected, pad_coefs_to_n, per_query_error, sensitivity_squared,
        sensitivity_squared_blt, sensitivity_squared_blt_limit, single_participation_sensitivity,
        solve_banded, toeplitz_dense, BufferedStreamingMatrixBuilder, BufferedToeplitz,
        CallbackArgs, ColumnNormalizedBanded, DenseState, DenseStreamingMatrix, Diagonal, Identity,
        LossFn, MatrixFactorizationPrivatizer, NoiseStrategy, Parameterization, PrefixSum,
        StreamingMatrix, Toeplitz, ToeplitzInverseStream,
    };
}
