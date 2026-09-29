# Changelog

## Unreleased

- Refuse `aperture_tag` (coupled infinite baffle) with any `metal_native_assembly_mode` other than `"corrected"` (`"optimized"`, `"parity"`, `"reference"`): those modes omit the singular aperture integrals at real k and are wrong at resonance (`"optimized"`: about 0.9 dB and 17 degrees on a 100 mm channel). `SolveConfig` raises `ValueError`; the native helper fails with a clear message. Use `"corrected"` (the default).
- Correct a native-helper comment that called `complex_k` the default coupled-IB formulation; the default is `standard`.

## 0.2.0 — 2026-09-28

- Report `impedance` as the pressure average on the lowest source tag with a nonzero weight (previously the lowest listed tag, even at zero weight); all-zero weights keep the lowest tag. The observation frame still uses the lowest listed tag.
- Breaking change (minor before 1.0): remove the CircSym solver, meridian API, CPU kernels, and native Metal ring kernels. Full-3D Metal remains the supported solve path.
- Remove `SolveConfig.circsym_baffle_z`, `SolveConfig.circsym_aperture_tag`, and `SolveConfig.should_continue`; passing any of these now raises `TypeError`.
- Keep coupled infinite-baffle and pulsating-sphere validation against analytic references.
