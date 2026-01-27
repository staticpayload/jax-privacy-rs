# Framework Adapters

Adapters provide integrations with Rust ML frameworks while keeping the core
DP logic framework-agnostic.

## Supported adapters

- **Burn** (`feature = "burn"`)
- **Candle** (`feature = "candle"`)
- **tch-rs** (`feature = "tch"`)

Each adapter exposes convenience wrappers for clipping and noise application
on framework-native tensors. Refer to the adapter crate docs for details.
