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
- triangle winding must be outward for canonical meshes
- physical tag `1` is the rigid-wall convention
- source/radiator tags must match `config.velocity_sources`
- the default source tag is `2`

The solver infers the observation frame from the source-tag element normals and
the mesh mouth. For enclosed or unusual geometry, pass `frame_override`.

## Configuration

Use `native_config(**overrides)` to create a supported Metal configuration.

Common fields:

- `freq_min_hz`, `freq_max_hz`, `freq_count`, `freq_spacing`
- `velocity_sources`, mapping physical tag to source weight
- `velocity_source_callback`, for frequency-dependent complex source weights
- `velocity_mode`, either `VelocityMode.ACCELERATION` or `VelocityMode.VELOCITY`
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

Use `solve_frequencies(mesh, frequencies_hz, config=None)` when frequency order
comes from the caller instead of a generated sweep.

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
swift build -c release --package-path hornlab_metal_bem/metal/native_helper
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
