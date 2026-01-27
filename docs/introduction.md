# Introduction

JAX-Privacy (Rust) is a production-oriented rewrite of the JAX Privacy library.
It is intended for teams building DP training pipelines in Rust and who want:

- A composable DP core that is not tied to a single ML framework.
- Accounting tools with repeatable numerical behavior.
- A clear mapping to the original JAX algorithms and APIs.

The library centers on a few core abstractions:

- `PyTree` for nested parameter structures.
- `Clipper` and `NoiseMechanism` for DP-SGD building blocks.
- `PrivacyAccountant` for tracking budget usage.

You can start by reviewing `docs/core_library.md` and the examples in
`crates/facade/examples`.
