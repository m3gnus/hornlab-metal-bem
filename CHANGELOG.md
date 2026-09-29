# Changelog

## Unreleased

- Fix intermittent NaN in the native helper on coupled infinite-baffle solves with a large aperture (about 1,500 aperture triangles). Accelerate's legacy LAPACK entry points (`cgesv_`, `zgesv_`, and the rest of the `__CLPK_*` interface) returned NaN when called concurrently from the helper's 6-way solve pipeline alongside `zgesv`/`cgemm`; the helper then aborted without a message. The helper now uses Accelerate's current LAPACK/BLAS (`ACCELERATE_NEW_LAPACK`, the `$NEWLAPACK` entry points): no failures in thousands of stress solves, results within float32 rounding of the legacy entry points, speed unchanged.
- Compatibility change: the native helper's macOS deployment target is now 13.3 (was 13.0), the first release with Accelerate's current LAPACK. Rebuild the helper (`swift build -c release`).
- The native helper now refuses to write a non-finite result: it fails with an error naming the case, frequency and quantity (`non-finite dense solution ...` or `non-finite value in native result at ...`) instead of aborting with an anonymous Objective-C exception. `metal-bem` raises it as a `RuntimeError`.
- The helper also scans its binary float32 outputs (surface pressure, field, batch field) and the standalone Neumann batch for non-finite values, with the same case/frequency error. `HORNLAB_METAL_BEM_NATIVE_TEST_INJECT_NAN=<case>:<surface|field>` plants a NaN for tests; it is off by default.
- Package the helper's Swift sources with a `*.swift` glob (wheels and sdists previously shipped only `main.swift`), so an installed copy can be rebuilt; the runtime stale-helper warning now compares against every Swift source and `Package.swift`, like `tests/native_helper_guard.py`.
- Add `scripts/schur_race_stress/` (manual stress reproduction of the legacy LAPACK failure; not run in CI).
- Add tests that the built helper links no legacy LAPACK symbol and that a large-aperture coupled-IB batch solves identically at solve concurrency 6 and 1.
- Fix `dense_solve_implementation="gmres"` refusing float32-exact answers next to an interior resonance (`info=-999`, e.g. a closed 816-triangle channel at 1700 Hz). Acceptance now also admits a normwise backward error at the float32 direct-LU level; a genuinely unconverged solve still fails, now with its iteration count, residual and backward error in the message.
- Refuse `aperture_tag` (coupled infinite baffle) with any `metal_native_assembly_mode` other than `"corrected"` (`"optimized"`, `"parity"`, `"reference"`): those modes omit the singular aperture integrals at real k and are wrong at resonance (`"optimized"`: about 0.9 dB and 17 degrees on a 100 mm channel). `SolveConfig` raises `ValueError`; the native helper fails with a clear message. Use `"corrected"` (the default).
- Add `SolveConfig.source_axes`, an optional per-source piston axis `{tag: (x, y, z)}`. Each axial source moves along its own axis with per-face scale `n_hat . axis`, no area-weighted sign vote, no symmetry projection and no dependence on the observation frame; an axis outside the symmetry subspace of a reduced solve raises `ValueError`. `source_axes=None` (default) leaves every existing solve unchanged. `solve_multi_source` honours it and requires an axis for every axial tag.
- Axial source motion with a degenerate or non-finite resolved axis now raises `ValueError` instead of silently running as uniform normal.
- Correct a native-helper comment that called `complex_k` the default coupled-IB formulation; the default is `standard`.

## 0.2.0 — 2026-09-28

- Report `impedance` as the pressure average on the lowest source tag with a nonzero weight (previously the lowest listed tag, even at zero weight); all-zero weights keep the lowest tag. The observation frame still uses the lowest listed tag.
- Breaking change (minor before 1.0): remove the CircSym solver, meridian API, CPU kernels, and native Metal ring kernels. Full-3D Metal remains the supported solve path.
- Remove `SolveConfig.circsym_baffle_z`, `SolveConfig.circsym_aperture_tag`, and `SolveConfig.should_continue`; passing any of these now raises `TypeError`.
- Keep coupled infinite-baffle and pulsating-sphere validation against analytic references.
