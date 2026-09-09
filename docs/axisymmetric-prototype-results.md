# Axisymmetric qualification prototype

Date: 2026-09-09. Status: **positive geometry and qualification-infrastructure
work retained; Axisymmetric remains disabled in the UI and AUTO routing.**

## Scope and decision rule

The current circular R-OSSE comparison uses a regenerated `yz+xz` quarter
surface with 52 meridian segments and 1,208 triangles. Geometry validation uses
the same configuration's 8x-refined generating curve: maximum distance is
0.0606 mm and p95 is 0.00641 mm. Coarse-panel-to-quarter distance remains a
scale-aware discretization diagnostic, not a topology verdict; the shared flat
mouth closure is checked separately and strictly.

Mesh generation is excluded from timed arms. One cold run per arm is excluded,
and subsequent paired runs alternate order.

Qualification requires all of the following:

- Axisymmetric/quarter-3D warm-median ratio below 0.5.
- Axisymmetric warm median at or below 0.5 seconds.
- Metal Axisymmetric output matching the compact CPU implementation.
- Axisymmetric versus quarter-3D errors below 2% complex-pressure relative L2,
  0.5 dB directivity above -40 dB, and 5 degrees RMS phase above the amplitude
  floor, both in aggregate and at every requested frequency. The per-frequency
  directivity gate uses a fixed reference mask; the historical intersection-mask
  metric remains in the report for continuity.

The harness requires every gate for overall approval. Passing speed alone never
enables the product path.

## Implemented positive results

- The mesher and CircSym builder now share the same flat freestanding mouth
  closure. The quarter mesh's angular circle geometry was also corrected. These
  fixes remove the prior closure-contract disagreement without treating coarse
  curve chords as a physics error.
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
  reports per-frequency errors, significant-field/null-region phase diagnostics,
  equal-volume source normalization, geometry provenance, and compact-CPU
  parity. It now also supports exact-reflected, independent-full, and ordered
  full-mesh-ladder controls without changing the activation gate. Fixed-geometry
  meridian subdivision supports 53/106/212/424-style convergence ladders without
  changing the represented profile.
- The paired real-k+CHIEF and complex-k shift-ladder experiment is diagnostic
  only. Thin-wall CHIEF points triggered near-boundary warnings and degraded the
  result; complex-k shifts 0.001 and 0.005 did not improve it.
- Native cancellation now drains helper output while polling, the batched Metal
  path rejects work counts outside its 32-bit kernel range, and regressions cover
  forced batch sweeps, baffled and unbaffled kernels, complex wavenumbers, and
  variable quadrature orders.

## Speed result and current candidates

The consolidated release-helper run measured:

| Axisymmetric | Quarter 3-D | Ratio | 2x gate | 0.5 s gate |
|---:|---:|---:|---:|---:|
| 0.202 s | 0.925 s | 0.218 | PASS | PASS |

Axisymmetric is about 4.58x faster than quarter-domain full 3-D for this matched
Apple Metal workload. A separate seven-pair run measured 0.329 s versus 0.971 s
(0.339x), so the exact wall time is sensitive to host load; both measurements
pass the product speed gates.

The targeted full-sweep candidates below were measured after the geometry work.
They remain speed-positive candidates, not high-frequency qualified physics
results because the quarter reference is unresolved there.

| Meridian segments | Axisym/quarter ratio | Axisym median |
|---:|---:|---:|
| 102 | 0.310 | 0.281 s |
| 126 | 0.337 | 0.305 s |
| 135 | 0.362 | 0.327 s |
| 148 | 0.391 | 0.354 s |

This acceleration is specific to the Axisymmetric formulation. A full-3D SIMD
prototype was also tested and rejected: it was 2.05x slower on the representative
2,272-triangle case, so no full-3D code from that experiment was retained.

The CPU improvements are useful independently of Metal and preserve the earlier
advantage observed on slower GPU-less systems, but CPU hardware still needs its
own qualification measurements before automatic routing is changed.

## Physics research and current limit

The original request was motivated by the historical coarse-mesh discrepancy:
**3.88%** complex-pressure relative L2, **14.73 dB** directivity above -40 dB,
and **29.31 degrees** RMS phase. Those figures remain historical evidence, not
the current decision metric after geometry and resolution work.

The reviewer found that the polygonal quarter source has 2.69% less area than the
meridian source. Scaling quarter-source velocity by 1.027608 to compare equal
volume velocity improved the 100 Hz pressure error from 2.62% to 1.03%. Across
the full 100 Hz--20 kHz sweep, however, the normalized pressure error is 5.21%;
directivity remains 14.73 dB and phase remains 29.31 degrees. This confirms that
source normalization is necessary but does not explain the high-frequency
discrepancy.

The finalized quarter ladder has 1,208 / 2,284 / 4,512 triangles. It is credible
only roughly through 1 / 1.5 / 2 kHz respectively. Across 100--2,000 Hz, the
fsqrt2-versus-f2 aggregate is 0.232% pressure, 0.135 dB directivity, and 0.224
degrees phase. At the comparable Axisymmetric rung, Axisym208-versus-f2 is
0.634% pressure, 0.640 dB directivity, and 0.439 degrees phase: directivity
barely exceeds the 0.5 dB gate.

Quarter near-quadrature experiments were rejected for negligible accuracy
benefit at their added cost. They are not used to explain or suppress the
remaining comparison uncertainty.

The high-frequency comparison remains unqualified because a converged quarter
reference has not yet been established or shown feasible. Therefore the current
decision is **not** that the Axisymmetric speed path fails. The infrastructure
and positive geometry/harness changes are retained, while Axisymmetric UI/AUTO
routing remains disabled pending a converged high-frequency reference.

### Unrestricted full-3D controls at 20 kHz

An exact four-image reflection of the 1,208-triangle quarter mesh was solved as
an ordinary unrestricted 4,832-triangle full-domain P1 system. Across five
frequencies from 100 Hz to 20 kHz, quarter-versus-full agreement remained far
inside the numerical budgets. At 20 kHz it was 0.000678% complex-pressure L2,
0.00327 dB fixed-reference-mask directivity, and 0.00792 degrees phase RMS. This
clears native symmetry, seam handling, and equal-volume source normalization as
material causes of the discrepancy.

Separately generated full meshes were then tested as an ordered 4,830 / 9,124 /
17,974-triangle ladder. The corresponding P1 systems have 2,417 / 4,564 / 8,989
unknowns; the finest primary complex128 matrix is about 1.20 GiB before solver
workspace. At 20 kHz, the middle full rung still differs from the finest full
rung by 2.42% pressure, 1.45 dB directivity, and 29.88 degrees phase. The full
ladder therefore fails convergence and the finest rung cannot be called ground
truth.

The matching finest quarter and independently remeshed full meshes nevertheless
agree to 0.292% pressure, 0.391 dB directivity, and 2.58 degrees phase at 20 kHz.
Against that same full result, Axisymmetric differs by 6.55% pressure, 9.05 dB
directivity, and 70.59 degrees phase. This is useful corroboration that the
Axisymmetric upper-band difference is real on the tested discretizations, but it
does not supersede the failed full-mesh convergence check. The finest full mesh's
worst edge is 21.69 mm, only 0.79 elements per 17.15 mm wavelength at 20 kHz.

### Geometry and azimuth convergence

The earlier 208/416/832 ladder subdivides the original straight meridian chords.
It tests panel/integration convergence on frozen faceted geometry, not convergence
to the configured R-OSSE profile. A separate regenerated-geometry experiment at
20 kHz changed by 6.00% pressure, 18.93 dB directivity, and 66.60 degrees phase
between 102 and 396 meridian segments. The 396-to-791 change fell to 0.331%,
2.225 dB, and 12.44 degrees, respectively. Geometry/panelization is therefore a
dominant upper-band error source, and even the 791-segment rung remains outside
the directivity and phase budgets.

The actual azimuth rule selects 235 samples at 20 kHz, so the old floor-32 versus
floor-64 check did not exercise upper-band quadrature. On regenerated 102-segment
geometry, forcing 256, 512, and 1,024 samples changed pressure by at most
1.04e-12 relative and directivity by at most 4.45e-10 dB between adjacent tested
rungs. The 256 floor is converged for this case; 512 and 1,024 only add cost.

### Cyclic full-3D reference prototype

A validation-only cyclic P1 `m=0` reduction now revolves a meridian into an exact
rotational mesh, sums full-matrix column orbits into representative rows, and
reconstructs the ordinary full-3D pressure. Against actually assembled corrected
full matrices at 20 kHz, reduced/full solution relative L2 was 2.17e-5 to 2.74e-5
for 8--32 sectors. This proves the discrete reduction contract independently of
the current midpoint-collocation Axisymmetric formulation.

The prototype still materializes the full dense matrix. Scaling it into a useful
upper-band reference requires one orbit-aware native operation covering both the
regular kernel and the singular/adjacent Duffy corrections. Implementing only
representative regular rows would silently omit corrections and is intentionally
not retained as a solve route. The detailed contract is in
`docs/cyclic-m0-reference-prototype.md`.

### Axisymmetric self-convergence

A fixed-geometry 40-frequency ladder from 100 Hz to 20 kHz was run at 208, 416,
and 832 meridian segments. The 416-versus-832 aggregate is 0.102% pressure,
0.862 dB directivity, and 2.655 degrees official phase RMS at the unchanged
1e-4 reference-amplitude floor. At 20 kHz, the corresponding values are 0.395%
pressure, 0.131 dB directivity, and 13.808 degrees official phase RMS; at the
tighter 1e-2 floor the same 20 kHz phase diagnostic is 0.429 degrees.

Pressure stabilizes and directivity is much closer, but the 0.862 dB aggregate
directivity result still exceeds the 0.5 dB gate. Deep-null phase sensitivity
also leaves the unchanged official phase gate unresolved. The 1e-2 result is
evidence that the added phase diagnostics are useful, not authority to weaken or
replace the 1e-4 qualification gate. The 832-segment sweep took about 3.03 s
and is a reference rung, not a product-speed candidate.

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

Decision: retain the fast implementations, CPU route, geometry fixes, and
refusal harness. Do not restore Axisymmetric to the UI or AUTO routing until a
converged high-frequency matched reference supports the physics gate.
