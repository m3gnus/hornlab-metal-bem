# ASRO2 Quarter Native Solve-Speed Follow-up

Date: 2026-06-02

Scope: corrected ASRO2 WG quarter mesh at
`runs/canonical-validation/260602-asro2-quarter-directivity-matched-resolution-backplate-sector/wg_hornlab_quarter_snapped.msh`.
All observations here keep corrected native assembly as the default path.

## Benchmarked Changes

- Per-kernel native Metal threadgroup overrides were added for matrix, RHS,
  Duffy, and field kernels. The global
  `HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP` / `SolveConfig.metal_native_threads_per_group`
  value remains the fallback.
- The resident solve-field path now returns native impedance and source
  surface-pressure averages, so the Python sweep no longer needs to read full
  surface pressure just for those scalar reductions.
- Resident field output can now use shared batch real/imag arrays instead of
  per-frequency observation-pressure files.
- The resident batch path reuses a single uploaded observation-point buffer
  when all frequencies share the same observation descriptor.
- An opt-in dense solve variant,
  `HORNLAB_SOLVER_METAL_NATIVE_DENSE_SOLVE_IMPL=cgetrf_cgetrs`, was added for
  comparison with the default `cgesv`.

## ASRO2 Results

Per-kernel threadgroup sweep output:
`runs/canonical-validation/260602-asro2-quarter-per-kernel-threadgroup-tuning/summary.json`

Latest run after native reductions, batched field output, and observation-buffer
reuse:

- Matrix sweep: 32/64/128 threads per group.
- Duffy sweep: 64/128/256/448 threads per group.
- Field sweep: 32/64/128 threads per group.
- RHS: global fallback, 64 threads per group.
- Pressure and directivity parity against corrected baseline: exact in the
  stored summary (`pressure_relative_l2=0`, max directivity delta `0 dB`).
- Impedance parity against corrected baseline: max relative L2 below `8.24e-9`.

Measured per-run differences are small and noisy. Global 64 remains the
practical default for this mesh; per-kernel knobs are useful for measurement
and future meshes, but this ASRO2 run does not justify changing defaults.

Dense solve variant output:
`runs/canonical-validation/260602-asro2-quarter-dense-solve-variants/summary.json`

- `cgetrf_cgetrs` matched pressure, directivity, and impedance numerically.
- In the recorded run, dense solve time was slightly slower than `cgesv`
  (`0.7783 s` vs `0.7758 s`), so `cgesv` remains the default.

Native scalar reduction validation against the previous Python computation on
representative frequencies found max absolute delta about `6.12e-10` for both
impedance and source surface-pressure average.

## Block-Staged Assembly

The block-staged triangle-pair prototype remains non-default.

The current prototype is numerically close but much slower because it computes
triangle-pair blocks on the GPU and then performs the matrix/RHS scatter
reduction on the CPU. For this ASRO2 mesh, that reduction cost dominates and
removes the benefit of staging pair blocks.

A useful follow-up would need a GPU-side reduction strategy that avoids this
host reduction. The obvious direction is a GPU scatter/reduction plan over the
precomputed Duffy slots and matrix/RHS destinations, but that introduces heavy
write contention into dense matrix entries and would need careful atomic or
multi-pass reduction design to preserve accuracy and determinism. That is not a
small incremental change from the current prototype, so no block-staged default
change is recommended for this mesh.
