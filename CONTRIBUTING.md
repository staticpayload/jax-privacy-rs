# How to Contribute

We welcome bug fixes, documentation improvements, and focused feature additions.
Please keep changes scoped and aligned with the architecture in `AGENTS.md`.

## Coordination & Claiming Issues

To avoid duplicate effort and ensure your work can be merged:

- **Check for existing PRs and issues** before starting.
- **Claim the issue** by commenting on it.
- **Wait for assignment** before large changes.
- **Stale assignments** may be cleared after 14 days of inactivity.

## Contributor License Agreement

Contributions must be accompanied by a Contributor License Agreement (CLA). If this
repository is hosted under a Google-managed organization, sign the CLA at:
<https://cla.developers.google.com/>.

If the hosting organization changes, follow the CLA instructions in the repo.

## Code Reviews

All changes require review via GitHub pull requests.

## Style Guide

This repository follows idiomatic Rust patterns and the public API rules below:

1. **Public APIs** must include doc comments and examples where practical.
2. **Cross-module helpers** should include short doc comments and clear names.
3. **Private helpers** should be prefixed with `_` and documented only when needed.
4. **Unsafe code** should be avoided; if required, justify with comments and tests.

## Linting & Testing

Before submitting a PR, run:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features
cargo test --workspace
```

For example builds:

```bash
cargo build --workspace --examples
```
