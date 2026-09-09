# Axisymmetric qualification prototype

Date: 2026-09-09. Base: `e7e32d0`. Status: **speed qualified; overall
qualification failed. Keep Axisymmetric out of the UI and AUTO routing until the
physics gate passes.**

## Scope and decision rule

The saved circular R-OSSE comparison uses 53 meridian segments, a 1,222-triangle
`yz+xz` quarter surface, 40 logarithmic frequencies from 100 Hz to 20 kHz, and
37-angle horizontal and vertical cuts at 2 m. Mesh generation is excluded from
both timed arms. One cold run per arm is excluded, and subsequent paired runs
alternate order.

Qualification requires all of the following:

- Axisymmetric/quarter-3D warm-median ratio below 0.5.
- Axisymmetric warm median at or below 0.5 seconds.
- Metal Axisymmetric output matching the compact CPU implementation.
- Axisymmetric versus quarter-3D errors below 2% complex-pressure relative L2,
  0.5 dB directivity above -40 dB, and 5 degrees RMS phase above the amplitude
  floor.

The harness requires every gate for overall approval. Passing speed alone never
enables the product path.

## Implemented positive results

- The portable CPU field default is now the compiled Numba implementation.
  NumPy remains selectable for diagnosis. On the CPU-only comparison, skipping
  analytically replaced near pairs reduced assembly by about 8--9% while
  preserving exact output; the measured one-thread sweep improved from 3.369 s
  to 3.158 s and the eight-thread sweep from 0.524 s to 0.499 s.
- Near-pair geometry is stored compactly and expanded only for the active block,
  avoiding the previous large dense geometry allocation.
- Metal assembly and field evaluation batch the complete frequency sweep in one
  helper invocation. Geometry, quadrature data, pipelines, and output buffers
  are reused across frequencies within that invocation. There is no persistence
  between helper invocations.
- The candidate 32-point azimuth floor agrees with the unchanged 64-point
  Axisymmetric reference on this fixture. Orders above the floor remain
  frequency-dependent.
- The benchmark records release/debug helper flavor, runs a helper smoke test,
  reports per-frequency errors, normalizes source velocity for equal physical
  volume velocity, and includes compact-CPU parity in overall qualification.
- Native cancellation now drains helper output while polling, the batched Metal
  path rejects work counts outside its 32-bit kernel range, and regressions cover
  forced batch sweeps, baffled and unbaffled kernels, complex wavenumbers, and
  variable quadrature orders.

## Speed result

The consolidated release-helper run measured:

| Axisymmetric | Quarter 3-D | Ratio | 2x gate | 0.5 s gate |
|---:|---:|---:|---:|---:|
| 0.202 s | 0.925 s | 0.218 | PASS | PASS |

Axisymmetric is about 4.58x faster than quarter-domain full 3-D for this matched
Apple Metal workload. A separate seven-pair run measured 0.329 s versus 0.971 s
(0.339x), so the exact wall time is sensitive to host load; both measurements
pass the product speed gates.

This acceleration is specific to the Axisymmetric formulation. A full-3D SIMD
prototype was also tested and rejected: it was 2.05x slower on the representative
2,272-triangle case, so no full-3D code from that experiment was retained.

The CPU improvements are useful independently of Metal and preserve the earlier
advantage observed on slower GPU-less systems, but CPU hardware still needs its
own qualification measurements before automatic routing is changed.

## Physics blocker

Before the independent review's source-area correction, Axisymmetric versus the
quarter mesh differed by **3.88%** complex-pressure relative L2, **14.73 dB**
directivity above -40 dB, and **29.31 degrees** RMS phase. The Claude Fable
reviewer was explicitly given those figures and told that speed is no longer the
remaining blocker.

The reviewer found that the polygonal quarter source has 2.69% less area than the
meridian source. Scaling quarter-source velocity by 1.027608 to compare equal
volume velocity improved the 100 Hz pressure error from 2.62% to 1.03%. Across
the full 100 Hz--20 kHz sweep, however, the normalized pressure error is 5.21%;
directivity remains 14.73 dB and phase remains 29.31 degrees. This confirms that
source normalization is necessary but does not explain the high-frequency
discrepancy.

The coarse meshes are not converged enough to decide which solver is the better
high-frequency reference. The next physics work is an equal-volume-velocity
meridian/quarter refinement ladder with per-frequency error reporting,
self-convergence for both formulations, and an analytic low-frequency anchor.
Until that work passes the numerical gate, overall qualification remains FAIL.

## Independent review

Claude Fable found no core numerical bug in the new compact CPU or batched Metal
paths. Its actionable findings were implemented: portable Windows backend
selection, Numba product default, equal-volume source normalization,
per-frequency metrics, release-helper provenance, sweep-level Metal regression,
pipe-safe cancellation, and the Metal work-count guard.

The review also identified larger future optimizations that were not positive
prototype results yet: cross-invocation persistent Metal sessions, streaming
large remainder batches instead of materializing every remainder, and reusing
the active mask across azimuth samples. They remain research items and are not
used to claim qualification.

## Reproduction

```bash
PYTHONPATH=.:../hornlab-waveguide-mesher \
python scripts/bench_axisymmetric_vs_quarter.py \
  --config path/to/rosse-config.json \
  --quarter-mesh path/to/rosse-quarter.msh \
  --axisym-backend metal --repeats 5 --azimuth-min 32 --json
```

Decision: retain the qualified fast implementations and the refusal harness,
but do not restore Axisymmetric to the UI or AUTO routing until the physics gate
passes on converged matched inputs.
