# Axisymmetric feasibility prototype

Date: 2026-09-09. Base: `e7e32d0`. Status: **qualification failed; do not
restore Axisymmetric to the UI or AUTO routing.**

## Scope and decision rule

This is the bounded two-round experiment from the 0.3.3 plan. It uses the saved
plain circular R-OSSE inputs: 53 meridian segments, a 1,222-triangle `yz+xz`
quarter surface, 40 logarithmic frequencies from 100 Hz to 20 kHz, and 37-angle
horizontal and vertical cuts at 2 m. Mesh generation is excluded from both timed
arms.

The hard product question is whether Axisymmetric is materially faster than the
realistic quarter-domain full-3D solve. The prior plan makes "materially" concrete
as at least 2x faster (ratio below 0.5) and at most 0.5 seconds for this sweep. A
result within timing noise of quarter is a failure.

## Prototype rounds

1. A Numba field kernel preserves the complete S/H ring quadrature but performs
   the target/source/line/azimuth reduction in compiled loops. It avoids the
   large temporary complex arrays created once per line node by the NumPy path.
   The production default remains NumPy; the prototype is explicitly selected
   with `HORNLAB_CIRCSYM_CPU_FIELD_BACKEND=numba` or `--cpu-field numba`.
2. An experimental azimuth-order floor tests 32 points instead of the existing
   conservative 64-point floor. Frequency-dependent orders above the floor are
   unchanged. This is an experiment, not an adaptive production rule.

The matched benchmark alternates solver order after one excluded warm-up per
arm. Full-3D uses the resident corrected Metal `yz+xz` path. Axisymmetric uses
the existing compiled CPU assembly plus the candidate CPU field kernel. The
existing one-shot CircSym Metal path was separately measured and was slower,
because it repeatedly launches and provisions the helper.

## Results

Warm wall-clock medians on the same Apple M-series host:

| Axisymmetric implementation | Pairs | Axisymmetric | Quarter 3-D | Ratio | Result |
|---|---:|---:|---:|---:|---|
| NumPy field, 64-point floor | 5 | 2.008 s | 0.957 s | 2.099 | Slower than quarter |
| Numba field, 64-point floor | 3 | 0.956 s | 0.973 s | 0.983 | Timing-noise tie |
| Numba field, 32-point floor | 9 | 0.919 s | 0.952 s | 0.965 | Timing-noise tie |

The compiled field kernel is a real local improvement: on the matched 64-point
case it reduces the field stage from about 1.34 s to about 0.28 s and total wall
time by roughly 2.1x. It does not create the required product advantage. The
best nine-pair candidate is only 3.5% faster than quarter, misses the 2x gate by
about 1.93x, and misses the 0.5-second target by about 0.42 s.

The best candidate's excluded cold run was 1.254 s; the corresponding quarter
warm-up was 1.007 s. Its remaining warm Axisymmetric cost was approximately
0.64 s assembly, 0.24 s field, and 0.007 s dense solve. Dense-solver work is not
a useful optimization target.

## Numerical evidence and limitation

- Numba versus NumPy ordinary field matrices agrees within `3e-13` relative in
  direct unit comparisons, for free-space and image-plane cases.
- The 32-point candidate versus the unchanged 64-point full Axisymmetric sweep
  differs by `7.66e-8` pressure relative L2, `4.93e-5 dB` maximum directivity,
  and `1.92e-5` degrees RMS phase above the amplitude floor on this fixture.
- Axisymmetric versus the current small quarter mesh over the complete 20 kHz
  band differs by 3.88% complex-pressure relative L2 and 2.29% magnitude L2;
  the directivity error is 14.73 dB even where both patterns exceed -40 dB,
  and RMS phase error above the amplitude floor is 29.31 degrees. These fail
  the prototype's explicit 2%, 0.5 dB, and 5-degree numerical gates. The prior
  convergence work already warns that this quarter mesh is not an absolute
  high-frequency truth reference. Therefore this run does not establish
  equal-error physics across the full band. That uncertainty cannot rescue a
  failed speed gate.

Existing analytic and CircSym parity tolerances remain unchanged. The complete
repository suite passes: 581 passed and 89 skipped.

## Independent alternatives from the Astra review

The independent architecture review found that the largest missing transport
improvement is a resident or frequency-batched CircSym helper. Other ranked
options are an error-controlled per-pair azimuth rule, SIMD-group reduction
inside each GPU ring pair, reuse of near-pair geometry, and eventually a
higher-order meridian discretization. The compiled CPU field option ranked
first and produced the measured improvement above.

A resident helper is not a credible automatic rescue by itself: the best
candidate already spends only about 0.24 s in field work, while CPU assembly is
the dominant 0.64 s stage. Meeting 0.5 s would require a further measured
assembly reduction as well as retaining all numerical gates. The plan permits
only two focused feasibility rounds, so those ideas remain research proposals
rather than grounds to re-enable the product path.

The full plan also called for a BEAT Metal comparison, independent meridian and
quarter-mesh convergence, raw load capture, and packaged-runtime identity. This
prototype stops before claiming that full qualification because the candidate
already failed the hard Metal-quarter speed gate. Running a slower comparison
or additional physics work cannot turn a timing-noise tie into the required 2x
advantage. The benchmark nevertheless records numerical thresholds and will
only report overall PASS when both speed targets and the pressure/directivity/
phase gates pass.

## Reproduction

```bash
PYTHONPATH=.:../hornlab-waveguide-mesher \
python scripts/bench_axisymmetric_vs_quarter.py \
  --config path/to/rosse-config.json \
  --quarter-mesh path/to/rosse-quarter.msh \
  --repeats 9 --cpu-field numba --azimuth-min 32 --json
```

Decision: keep explicit experimental backend access and saved-result reading,
but leave Axisymmetric hidden and excluded from AUTO. Retirement or preservation
of the research implementation is a separate follow-on decision.
