# Overview

JAX-Privacy (Rust) provides differentially-private (DP) machine learning
building blocks that mirror the structure of the original JAX Privacy library.
The goal is to enable Rust ML stacks to compose DP training pipelines with
clear accounting and reproducible results.

## What this repo includes

- Framework-agnostic DP primitives (clipping, noise, sampling).
- Privacy accounting for DP-SGD and related mechanisms (RDP + PLD).
- Streaming matrix factorization utilities for correlated noise.
- Auditing utilities for membership inference and canary tests.
- A trait-based PyTree abstraction for structured tensors.
- Optional adapters for Burn, Candle, and tch-rs.

## What this repo does not include

- Full Python/Keras training pipelines.
- JAX/Flax-specific APIs or experiments.
- GPU kernels beyond what the adapters provide.

For paper reproduction scripts, refer to the original JAX Privacy repository.
