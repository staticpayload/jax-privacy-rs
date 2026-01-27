# Installation

This crate is not yet published on crates.io. Use a local path or Git source.

## Local path

```
cargo add jax-privacy --path crates/facade
```

## Git dependency

```
[dependencies]
jax-privacy = { git = "<your-repo-url>", package = "jax-privacy" }
```

## Optional features

- `f32`: switch core computations to f32.
- `serde`: enable serialization.
- `burn`, `candle`, `tch`: enable framework adapters.

Example:

```
[dependencies]
jax-privacy = { git = "<your-repo-url>", package = "jax-privacy", features = ["serde", "candle"] }
```
