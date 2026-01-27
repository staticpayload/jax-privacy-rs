# Library Design

This Rust port follows the architectural decisions documented in `AGENTS.md`:

- **Framework-agnostic core**: keep DP logic independent of any ML framework.
- **PyTree abstraction**: trait + derive macro for structured tensors.
- **Stateless PRNG**: JAX-compatible key splitting and fold-in semantics.
- **CPU-first**: GPU support is opt-in via adapter features.
- **Precision default**: f64 for numerical stability, optional f32 feature.

The goal is to mirror the JAX Privacy algorithms while presenting Rust-native
APIs with explicit error handling and memory ownership rules.
