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

The full-3D native backend is Apple Silicon/macOS only. The axisymmetric
meridian (`solve_circsym*`) formulation is portable: it runs on macOS, Windows,
and Linux through compiled CPU kernels, and opportunistically uses Metal on
Apple Silicon. It does not route Windows users through the full-3D Metal helper.

The solver uses a NumPy-only mesh/grid/function-space loader and does not
depend on `bempp-cl`. There is no OpenCL/Bempp fallback path in this package.
On Apple Silicon, the normal package build requires Swift from the Xcode
command-line tools and compiles a release helper into the installed package,
avoiding a slow first-use source compilation.

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

For a reproducible axisymmetric benchmark, including the exact backend and
quadrature order used, run:

```bash
python scripts/bench_circsym.py --json
python scripts/bench_circsym.py --fixture infinite-baffle --backend cpu --json
```

The default fixture is a closed free-standing conical horn swept from 400 Hz to
16 kHz. `--target-edge-mm`, `--frequencies`, `--angles`, and `--repeat` expose
the workload without hiding it behind a machine-specific preset. CI separately
checks free-standing and coupled infinite-baffle numerical goldens on macOS,
Windows, and Linux.

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

Signed-volume winding validation is applied to closed two-manifold meshes. For
supported open meshes (bare horns and mirror-reduced domains), signed volume is
origin-dependent, so loading does not flip or reject winding from that scalar;
the source-normal, symmetry/open-edge, and native geometry contracts remain the
relevant checks.

A Gmsh surface physical group named `mouth_aperture` is the canonical coupled
infinite-baffle declaration. `load_mesh()` carries that declaration into the
public solve APIs, which automatically resolve `SolveConfig.aperture_tag` and
reject explicit conflicts. A bare numeric tag `12` is not sufficient by itself:
external meshes may use that number for unrelated boundaries, so automatic
coupled-IB routing requires the physical name (or an explicit `aperture_tag`).

The solver treats usable source-tag element normals as the authoritative forward
direction and locates the mesh mouth along that axis. For external meshes whose
source winding is not authoritative, unusual multi-source geometry, or an exact
caller-owned reference frame, pass `frame_override`.

`native_symmetry_plane` means a mirror-reduced half/quarter mesh: the inferred
frame axis and origin are projected onto the requested symmetry plane(s) so
reduced solves report the same frame as the full model. Callers that use a
symmetry plane as a rigid-baffle image method around a full mesh must pass
`frame_override` instead.

## Rigid Half Space

`ground_plane` puts one rigid, infinite, perfectly reflecting boundary in the
domain and solves the open half space above it: `"xy"` is a floor at Z=0,
`"yz"` a wall at X=0, `"xz"` a wall at Y=0. The mesh is the **complete**
radiating body and must lie at or above zero on that axis.

It shares the image kernel with `native_symmetry_plane` but not its contract,
and the difference is what the mesh means:

| | `native_symmetry_plane` | `ground_plane` |
|---|---|---|
| the mesh is | half/quarter of a mirror-symmetric body | the whole body |
| must reach the plane | yes, its rim lies on the cut | no, it may float clear |
| a face lying in the plane | rejected | rejected |
| the image is | part of the real radiator | fictitious |
| radiated surface power | multiplied by the copy count | counted once |
| observation frame | projected onto the plane | left where it is |

So a cabinet flown above a stage, or standing on a floor with a gap under it,
is a `ground_plane` solve and is not expressible as a symmetry plane at all —
the reduced-domain validator requires a vertex on the plane. A cabinet resting
flat with its contact face deleted (rim on the plane) is expressible either
way; `native_symmetry_plane="xy"` gives the correct field for it but reports
twice the radiated surface power, because that mode believes the mirrored body
is a second real radiator.

`ground_plane_min_clearance_m` optionally demands a gap. Contact along an edge
or vertex is otherwise allowed, and the existing Duffy correction covers the
coincident and adjacent real-vs-image element pairs it produces. For a body
very close to but not touching the plane, the real-vs-image pairs become
near-singular; the opt-in near-quadrature correction
(`HORNLAB_METAL_BEM_NATIVE_NEAR_QUADRATURE=auto`, default off) covers those.
Measured on a 100 mm sphere with a 57.7 mm max edge at 2 kHz, enabling it moved
the far field by 0.012 dB at 2 mm clearance (0.03 of an element edge) and by
0.0001 dB at 20 mm, and moved the surface-pressure impedance by 2.9e-4 and
5.6e-6 relative. It is worth enabling for near-field and impedance work, not
for polar sweeps.

`ground_plane` does not currently compose with `native_symmetry_plane` (the
native session carries one image-plane set) or with the coupled
infinite-baffle `aperture_tag`; both combinations refuse in `SolveConfig`.

## Several Bodies In One Domain

`combine_bodies()` places multiple closed bodies into a single exterior domain,
where they couple through the kernel like any other elements — a boundary
integral operator pairs elements, not connected components.

```python
from hornlab_metal_bem import BodyPlacement, combine_bodies, rotation_matrix

scene = combine_bodies([
    BodyPlacement(top, name="top"),
    BodyPlacement(
        sub,
        translation_m=(0.0, 0.0, -0.9),
        rotation=rotation_matrix([0.0, 0.0, 1.0], 12.0),
        tag_map={1: 11, 2: 12},          # keep this body's tags distinct
        name="sub",
    ),
])
result = solve(scene.mesh, native_config(
    ground_plane="xy", velocity_sources={2: 1.0, 12: 1.0}))
```

`tag_map` is what keeps `velocity_sources` able to address one body rather than
both: two cabinets that each arrive tagged `{1: rigid, 2: throat}` would
otherwise merge into one indistinguishable pair of tags. A collision between
bodies is an error rather than a silent merge. Improper (reflecting) rotations
are refused because they invert outward winding, which the loader cannot detect
once several bodies are summed into one signed volume. Vertices are never
merged between bodies, so touching cabinets stay two closed surfaces.

`CombinedMesh` carries `body_ids` (per triangle), `tag_maps`, `body_names`, and
`tags_for(body)` back out.

## Configuration

Use `native_config(**overrides)` to create a supported Metal configuration.

Common fields:

- `freq_min_hz`, `freq_max_hz`, `freq_count`, `freq_spacing`
- `velocity_sources`, mapping physical tag to source weight
- `velocity_source_callback`, for frequency-dependent complex source weights.
  Its returned tags must be a subset of `velocity_sources`; declare every
  potentially driven tag there up front (a zero static weight is fine).
  Omitting a declared tag leaves it undriven at that frequency. Unlike
  `impedance_source_callback`, it cannot introduce tags because velocity tags
  establish the source geometry, observation frame, native session, pressure
  averages, and impedance reference before the frequency loop.
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
- `ground_plane`, one of `None` (default), `"xy"`, `"yz"`, or `"xz"` — a
  rigid half-space boundary the complete mesh stands next to, with
  `ground_plane_min_clearance_m` as an optional minimum gap
- `return_surface_pressure`, opt-in full solved P1 surface pressure output
- `return_surface_traces`, opt-in P1 pressure plus total DP0 Neumann traces for
  post-solve exterior-field evaluation
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
impedance/validation sweeps. It requires one closed interior-channel meridian
entirely behind the baffle, with a contiguous mouth-to-axis aperture at global
`z=0` whose normals point `-Z`. Complex-k regularization and Robin/admittance
walls are supported; CHIEF points are rejected until their coupled augmented
constraints are implemented. Generated observation arcs honor `origin="mouth"`
or `origin="throat"`.

The legacy `circsym_baffle_z` image kernel is intentionally limited to a
coplanar flat Rayleigh sheet (such as a baffled piston). It is not a recessed
horn model: use `circsym_aperture_tag` for a flush-mounted waveguide. Bare,
zero-thickness open meridians are rejected because the current one-trace BIE is
a closed-surface formulation; finite-thickness freestanding meridians that
close on the symmetry axis remain supported.

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
  (always populated, including CircSym coupled-IB solves)
- `surface_pressure_complex`: optional `(F, n_p1_dofs)` solved surface pressure
  when `return_surface_pressure=True` or `return_surface_traces=True`
- `surface_neumann_complex`: optional `(F, n_dp0_dofs)` total `dp/dn`, including
  the Robin correction, when `return_surface_traces=True`
- `native_diagnostics`: per-frequency native implementation, LAPACK, Duffy,
  Metal dispatch, symmetry, and resident batch metadata
- `timings` and `solver_log`: backend timing and diagnostic metadata

`directivity_db` is not absolute SPL. Use `pressure_complex` for absolute
complex pressure and derive SPL explicitly when needed.

Retained traces use the solver's `e^{-i omega t}` phase convention. Evaluate
one frequency at arbitrary `(N, 3)` exterior points with
`evaluate_exterior_from_traces(mesh, frequency_hz, k_real, pressure_p1,
neumann_dp0, points_xyz, symmetry_plane=...)`. The mesh and symmetry must match
the solve. Coupled infinite-baffle and CircSym trace evaluation are not part of
this full-3D Phase 0 API.

## Install For Development

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/build_metal_native_release.py --require-metallib
python -m pytest tests/test_config.py tests/test_hornlab_metal_bem_namespace.py tests/test_metal_native.py -q
```

Cross-repository parity tests import their dependencies from the active Python
environment and never reach into sibling checkouts. Install the public
packages when running that coverage:

```bash
python -m pip install \
  "hornlab-waveguide-mesher @ git+https://github.com/m3gnus/hornlab-waveguide-mesher.git" \
  "hornlab-bempp-bem @ git+https://github.com/m3gnus/hornlab-bempp-bem.git"
```

Those tests skip cleanly when their optional dependencies are not installed.

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
