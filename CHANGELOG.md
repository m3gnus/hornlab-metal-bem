# Changelog

## Unreleased

- Fix intermittent NaN in the native helper on coupled infinite-baffle solves with a large aperture (about 1,500 aperture triangles). Accelerate's legacy LAPACK entry points (`cgesv_`, `zgesv_`, and the rest of the `__CLPK_*` interface) returned NaN when called concurrently from the helper's 6-way solve pipeline alongside `zgesv`/`cgemm`; the helper then aborted without a message. The helper now uses Accelerate's current LAPACK/BLAS (`ACCELERATE_NEW_LAPACK`, the `$NEWLAPACK` entry points): no failures in thousands of stress solves, results within 5e-6 relative of the legacy entry points, speed unchanged.
- Compatibility change: the native helper's macOS deployment target is now 13.3 (was 13.0), the first release with Accelerate's current LAPACK. Rebuild the helper (`swift build -c release`).
- The native helper now refuses to write a non-finite result: it fails with an error naming the case, frequency and quantity (`non-finite dense solution ...` or `non-finite value in native result at ...`) instead of aborting with an anonymous Objective-C exception. `metal-bem` raises it as a `RuntimeError`.
- Add tests that the built helper links no legacy LAPACK symbol and that a large-aperture coupled-IB batch solves identically at solve concurrency 6 and 1.
- Refuse `aperture_tag` (coupled infinite baffle) with any `metal_native_assembly_mode` other than `"corrected"` (`"optimized"`, `"parity"`, `"reference"`): those modes omit the singular aperture integrals at real k and are wrong at resonance (`"optimized"`: about 0.9 dB and 17 degrees on a 100 mm channel). `SolveConfig` raises `ValueError`; the native helper fails with a clear message. Use `"corrected"` (the default).
- Correct a native-helper comment that called `complex_k` the default coupled-IB formulation; the default is `standard`.

## 0.2.0 — 2026-09-28

- Report `impedance` as the pressure average on the lowest source tag with a nonzero weight (previously the lowest listed tag, even at zero weight); all-zero weights keep the lowest tag. The observation frame still uses the lowest listed tag.
- Breaking change (minor before 1.0): remove the CircSym solver, meridian API, CPU kernels, and native Metal ring kernels. Full-3D Metal remains the supported solve path.
- Remove `SolveConfig.circsym_baffle_z`, `SolveConfig.circsym_aperture_tag`, and `SolveConfig.should_continue`; passing any of these now raises `TypeError`.
- Keep coupled infinite-baffle and pulsating-sphere validation against analytic references.
