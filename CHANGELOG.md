# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-01-27

Initial Rust port with core parity:
- Core DP primitives (clipping, aggregation, Gaussian/Laplace noise).
- RDP + PLD accounting with calibration helpers.
- Streaming matrix factorization utilities.
- Auditing utilities for membership inference and canary scoring.
- JAX-compatible stateless PRNG.
- PyTree trait + derive macro.
- Adapter scaffolding for Burn, Candle, and tch-rs.
- Examples and integration tests.
