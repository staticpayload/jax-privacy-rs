# Core Library

The core crate (`jax-privacy-core`) provides DP primitives and traits.

## Traits

- `PyTree`: flatten/unflatten nested parameter structures.
- `Clipper`: clipping strategies with reports.
- `NoiseMechanism`: Gaussian/Laplace mechanisms over tensors.
- `BatchSelector`: sampling plans (Poisson, fixed-size, cyclic).
- `GradientTransform`: composable per-step transforms.

## Modules

- `clipping`: global norm clipping, bounded-sensitivity utilities.
- `noise`: Gaussian and Laplace noise addition.
- `batch_selection`: Poisson, cyclic Poisson, and balls-in-bins.
- `sharding`: helpers for shard-aware reshaping and padding.

## Examples

See `crates/facade/examples` for end-to-end usage (DP-SGD, accounting,
calibration, auditing).
