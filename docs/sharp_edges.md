# Sharp Edges

This project is a fast-moving port. Known sharp edges include:

- Adapter APIs are intentionally minimal and may evolve.
- PLD accounting is discretized and may diverge slightly from the Python
  reference under extreme parameter settings.
- Correlated-noise matrix factorization utilities are tuned for correctness,
  not peak performance.
- No GPU kernels are provided outside adapter integrations.
