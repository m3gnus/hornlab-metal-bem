# hornlab-metal-bem

Apple Metal accelerated acoustic BEM solver for HornLab waveguide and
loudspeaker surface meshes.

This repository packages the fastest currently validated HornLab solver path:

- corrected native resident Metal dense assembly
- Accelerate `cgesv` dense solve
- native impedance and source surface-pressure reductions
- batched native field output
- resident observation buffer reuse
- stable default Metal dispatch at 64 threads per threadgroup

Use the `hornlab_metal_bem` namespace for all new integrations.

## Status

This is an Apple Silicon/macOS solver backend. It is intended to replace or
augment local acoustic BEM backends on Apple Silicon, not NVIDIA CUDA backends
on Windows.

The solver uses a NumPy-only mesh/grid/function-space loader and does not
depend on `bempp-cl`. There is no OpenCL/Bempp fallback path in this package.

## Quick Start

Run a Bempp-free native Metal solve:

```python
from hornlab_metal_bem import native_config, solve

config = native_config()
result = solve("waveguide.msh", config)

print(result.frequencies_hz.shape)
print(result.directivity_db.shape)
print(result.impedance.shape)
```

Recent ASRO2 corrected-quarter benchmark (HornLab, Apple M-series):

- 40 frequencies
- 3 planes x 37 angles (996-dof `yz+xz` quarter mesh)
- about 2 s end-to-end `solve()` wall time — GPU-assembly-bound, with
  pipelined GPU assembly overlapping a concurrent dense-solve worker pool
  (`HORNLAB_METAL_BEM_NATIVE_SOLVE_CONCURRENCY`, default 6)
- corrected assembly matches the subdivided-quadrature reference to `< 1e-4`
  relative L2 (matrix and RHS), and the `yz+xz` quarter matches the
  full-domain solve

## Inputs

`solve(mesh, config=None)` accepts either:

- a path to a Gmsh `.msh` triangle surface mesh
- a `LoadedMesh` returned by `load_mesh()`

Mesh requirements:

- coordinates are metres unless `mesh_scale` is set
- mesh cells must contain triangles
- triangle cells must have physical-group tags
- triangle winding must be outward for exterior-domain canonical meshes; coupled
  infinite-baffle meshes with an aperture tag use the interior-domain contract
  and carry negative signed volume, with mouth-aperture normals pointing `-Z`
- physical tag `1` is the rigid-wall convention
- source/radiator tags must match `config.velocity_sources`
- the default source tag is `2`

The solver infers the observation frame from the source-tag element normals and
the mesh mouth. For enclosed or unusual geometry, pass `frame_override`.

`native_symmetry_plane` means a mirror-reduced half/quarter mesh: the inferred
frame axis and origin are projected onto the requested symmetry plane(s) so
reduced solves report the same frame as the full model. Callers that use a
symmetry plane as a rigid-baffle image method around a full mesh must pass
`frame_override` instead.

## Configuration

Use `native_config(**overrides)` to create a supported Metal configuration.

Common fields:

- `freq_min_hz`, `freq_max_hz`, `freq_count`, `freq_spacing`
- `velocity_sources`, mapping physical tag to source weight
- `velocity_source_callback`, for frequency-dependent complex source weights
- `velocity_mode`, either `VelocityMode.ACCELERATION` or `VelocityMode.VELOCITY`
- `source_motion`, either `SourceMotion.NORMAL` (default; uniform normal
  velocity, a breathing cap) or `SourceMotion.AXIAL` (rigid piston along the
  source axis, `v_n = weight * (n_hat . axis)` — the realistic wavefront for a
  dome/cone/diaphragm; a flat disc reduces exactly to `NORMAL`; one tag covering
  both front/back faces of a thin diaphragm gives the dipole path because axial
  preserves the opposite per-face signs)
- `source_velocity_profiles`, optional per-tag overrides for `source_motion`:
  `NormalProfile`, `AxialProfile`, `TaperProfile(kind="raised_cosine"|"linear",
  start=0.7)`, `AnnularProfile(r_inner, r_outer)`, plus `PerFaceProfile(weights)`
  and `CallableProfile(callback)` hooks for explicit/modal/measured maps
- `observation`, an `ObservationConfig`
- `mesh_scale`
- `air_density`
- `native_symmetry_plane`, one of `None`, `"yz"`, `"xz"`, `"xy"`, or `"yz+xz"`
- `return_surface_pressure`, opt-in full solved P1 surface pressure output
- `progress_callback`
- `on_frequency_result`, for streaming/early stop; entries include complex
  observation pressure

The native Metal package supports standard Neumann solves by default. It also
exposes experimental opt-in `formulation="complex_k"` and
`impedance_sources={tag: beta}` Robin admittance support. It does not expose
legacy OpenCL/Bempp fallback configuration or Burton-Miller as user-facing
features.

## CircSym Axisymmetric Solver

`solve_circsym()` and `solve_circsym_frequencies()` run the axisymmetric
`m=0` DP0 meridian solver for circular bodies of revolution. Use it only for
circular/axisymmetric geometries; non-round cross sections, morphing, and
enclosures are outside the current validity envelope. Infinite-baffle CircSym
is supported for circular waveguides via `SolveConfig.circsym_aperture_tag`,
which names the flush `z=0` aperture segments and switches the solve to the
exact coupled path (interior meridian BEM + analytic Rayleigh half-space
aperture coupling). Prefer the full 3D coupled solve (`aperture_tag`) for
production directivity; the CircSym IB path is for fast axisymmetric
impedance/validation sweeps and requires a flush aperture at the global
`z=0` baffle plane.

With the wavelength-scaled meridian budget, CircSym is intended for waveguide
sweeps up to roughly 30-40 kHz when the meridian resolves the requested band.
The default `complex_k` formulation avoids closed-surface irregular
frequencies, but the complex shift adds an `O(shift)` bias to surface
impedance, about 0.03 dB for the default shift. Observation fields are still
evaluated with the real acoustic wavenumber.

The returned `impedance` is the area-weighted average pressure on the driven
source cap per unit drive. It is not a throat-plane radiation impedance and is
not normalized by `rho*c`.

Use `solve_frequencies(mesh, frequencies_hz, config=None)` when frequency order
comes from the caller instead of a generated sweep.

Use `solve_multi_source(mesh, sources, config=None, frequencies_hz=None)` when
several velocity sources share one mesh (e.g. HF/MF/LF drive bases or aperture
radiation-matrix columns). Each entry of `sources` is a `velocity_sources`
dict; the helper assembles and factors each frequency's operator ONCE and
back-substitutes one right-hand side per source (multi-RHS), so N sources cost
roughly one solve plus N-1 cheap RHS/field passes. It returns one
`SolveResult` per source, matching sequential `solve()` calls to float32
tolerance, and records `surface_pressure_avg` on the union of all source tags
in every result (zero-velocity tags are legal listeners). Multi-source rides
the default pipelined corrected/optimized assembly path; when
`on_frequency_result` is set, each callback entry includes a `source_results`
list with one per-source log entry. The reference/parity debug modes stay
single-source.

## Observation Points

`ObservationConfig` builds polar observation arcs by default:

```python
from hornlab_metal_bem import ObservationConfig, native_config

config = native_config(
    observation=ObservationConfig(
        planes=["horizontal", "vertical"],
        distance_m=2.0,
        angle_min_deg=0.0,
        angle_max_deg=180.0,
        angle_count=37,
        origin="mouth",
    )
)
```

Allowed plane names are `"horizontal"`, `"vertical"`, and `"diagonal"`.

For exact observation coordinates, set `custom_points` to a mapping of plane
name to an `(N, 3)` array in metres. All requested planes must be present and
must have the same point count.

## Outputs

`solve()` and `solve_frequencies()` return `SolveResult`.

Key result fields:

- `frequencies_hz`: `(F,)` solved frequencies in Hz
- `pressure_complex`: `(F, P, N)` complex pressure at observation points
- `directivity_db`: `(F, P, N)` directivity normalized so the on-axis angle is `0 dB`
- `spl_norm_db`: alias for `directivity_db`
- `impedance`: `(F,)` area-weighted average complex surface pressure on the
  impedance source tag, in pascals per unit drive (not divided by drive
  velocity and not normalized to `rho*c`)
- `observation_angles_deg`: `(N,)` polar angles in degrees
- `observation_points`: `(P, N, 3)` observation coordinates in metres
- `observation_planes`: plane names matching axis `P`
- `surface_pressure_avg`: source-tag keyed average surface pressure arrays
- `surface_pressure_complex`: optional `(F, n_p1_dofs)` solved surface pressure
  when `return_surface_pressure=True`
- `native_diagnostics`: per-frequency native implementation, LAPACK, Duffy,
  Metal dispatch, symmetry, and resident batch metadata
- `timings` and `solver_log`: backend timing and diagnostic metadata

`directivity_db` is not absolute SPL. Use `pressure_complex` for absolute
complex pressure and derive SPL explicitly when needed.

## Install For Development

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/build_metal_native_release.py --require-metallib
python -m pytest tests/test_config.py tests/test_hornlab_metal_bem_namespace.py tests/test_metal_native.py -q
```

## Maintainer Docs

For implementation details, see:

- [Architecture](docs/architecture.md)
- [Native IPC contract](docs/native-ipc.md)

## Boundary Lab Backend

The package exposes a Boundary Lab solver backend/session implementation and
translates Boundary Lab `SolveRequest` and `SimulationConfig` objects into the
native Metal solve configuration.

Backend id: `hornlab_metal`.

```python
from hornlab_metal_bem.boundary_lab import create_backend

backend = create_backend()
session = backend.create_session(solve_request)
for result in session.solve_stream():
    ...
```
