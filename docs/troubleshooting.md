# Troubleshooting

## Build failures on macOS

Ensure Xcode licenses are accepted:

```
xcodebuild -license
```

## Mismatched epsilon values

- Confirm the sampling method and neighboring relation match the Python
  reference.
- Check that the discretization interval and truncation parameters match
  your intended PLD settings.
- Use `f64` unless you explicitly enable the `f32` feature.

## Slow accounting

- PLD composition is heavier than RDP. Prefer RDP when approximate accounting
  is sufficient.
- Reduce PLD discretization interval only if needed for precision.
