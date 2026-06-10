# Architecture

This package exposes a Bempp-free acoustic BEM solve path backed by the
package-owned native Swift/Metal helper. The Python package is responsible for
loading meshes, constructing observation coordinates, translating user or
Boundary Lab configuration, and normalizing results. The native helper owns the
resident dense assembly, dense solve, exterior field evaluation, and native
pressure reductions.

## Module Responsibilities

- `hornlab_metal_bem.__init__` is the public package surface. It exports
  `native_config()`, `solve()`, `solve_frequencies()`, mesh loading, config,
  observation, and result types. `native_config()` forces the supported native
  assembly mode and does not expose legacy OpenCL/Bempp options.
- `config.py` defines `SolveConfig`, `ObservationConfig`, velocity-source
  modes, native symmetry options, callback hooks, mesh scale, air density, and
  native threadgroup overrides.
- `mesh.py` loads Gmsh triangle meshes into NumPy-only `PureGrid` and
  `PureFunctionSpace` objects. The native path consumes these Bempp-shaped
  arrays without importing `bempp-cl`.
- `observation.py` infers the acoustic frame from source-tag normals and builds
  polar or caller-supplied observation points.
- `sweep.py` is the production solve coordinator. It builds the frequency grid,
  observation buffers, function spaces, native geometry buffers, Neumann data,
  invokes the resident native session, then assembles `SolveResult`.
- `bie.py` contains Python BIE helper math used around the native path, such as
  source Neumann coefficients and fallback pressure reductions when the helper
  writes full surface pressure instead of reductions.
- `result.py` contains `SolveResult` and `MeshInfo`, including the public array
  shapes returned by `solve()` and `solve_frequencies()`.
- `backends.py` performs production runtime availability checks for the native
  Metal backend.
- `boundary_lab.py` adapts Boundary Lab requests and config objects onto the
  package API.
- `metal/geometry.py`, `metal/session.py`, `metal/native.py`, and the Swift
  files under `metal/` implement the native helper split described below.
- `validation/native_symmetry.py` contains native symmetry validation helpers
  used by the test suite.

## Public API Flow

`solve(mesh, config=None)` and `solve_frequencies(mesh, frequencies_hz,
config=None)` share the same execution flow:

1. Resolve configuration. If omitted, `native_config()` creates a `SolveConfig`
   with native corrected assembly mode.
2. Resolve the mesh. A path is loaded by `load_mesh(..., scale=config.mesh_scale)`;
   an existing `LoadedMesh` is used directly.
3. Resolve the observation frame. `frame_override` is respected first;
   otherwise `infer_frame()` uses the lowest configured velocity-source tag as
   the source tag.
4. Route to the native path. `should_route_native_metal()` validates the narrow
   supported native symmetry options.
5. `run_sweep_native_metal()` builds observation points, P1 and DP0 function
   spaces, Metal geometry buffers, frequency and wavenumber arrays, and DP0
   Neumann rows.
6. `MetalNativeStandardSession.create_session()` writes a session manifest and
   geometry binaries, validates native runtime availability, and launches the
   resident helper operations.
7. The production path uses
   `assemble_solve_evaluate_standard_neumann_batch()` for resident assembly,
   Accelerate dense solve, exterior field evaluation, impedance, and source
   surface-pressure reductions. When `return_surface_pressure=True`, it also
   requests solved P1 surface-pressure output from the helper.
8. Python reads little-endian float32 real/imag result arrays, reshapes them to
   observation planes and angles, computes normalized directivity, accumulates
   timing/log metadata and native diagnostics, and returns `SolveResult`.

When `on_frequency_result` is unset, `sweep.py` sends the full frequency batch
and may request a single batched field output. When `on_frequency_result` is
set, it still sends one full-sweep batch but passes `on_case_result` down to
the session, which opts the helper into streamed per-case results: the helper
writes one `case-XXXX.json` manifest (atomically) as each frequency completes,
Python tails those files to fire `on_frequency_result` while later frequencies
are still solving, and an early-stop request terminates the helper process —
every finished case's outputs are already on disk. Streaming callback entries
include normalized directivity and complex observation pressure for the solved
frequency. Helpers that predate streamed case results degrade gracefully: the
batch runs to completion and callbacks fire from the final batch manifest.

"Resident" refers to buffer and pipeline reuse across the cases of a single
helper invocation: each operation manifest launches one helper process. Both
the batch and streaming paths therefore pay Metal device/pipeline setup and
geometry upload once per sweep.

Inside one `assemble_solve_evaluate_standard_neumann_batch` invocation, the
helper additionally overlaps GPU and CPU work: case i+1's assembly command
buffers are committed before case i's Accelerate dense solve runs, with
even/odd cases writing to double-buffered assembly outputs. Batch wall time
then approaches max(GPU assembly, CPU solve) instead of their sum.

## Boundary Lab Adapter

`boundary_lab.py` exposes backend id `hornlab_metal`.

`create_backend(**default_overrides)` returns a `BoundaryLabBackend` with
capabilities that match the local native backend: streaming, cancellation,
impedance, and symmetry are supported; spherical sampling, Burton-Miller,
remote assets, flat target normalization, and parallel workers are not exposed.

`BoundaryLabSession` accepts either a Boundary Lab `SolveRequest`-like object or
a raw `SimulationConfig`-like object. The adapter deliberately accepts dicts and
ordinary objects and ignores unknown fields so Boundary Lab can evolve
independently. It translates:

- frequency range or explicit `frequencies_hz`
- observation planes, distance, angle bounds/count, origin
- `velocity_sources`, `source_tag`, `driver_tag`, or radiator/channel drive
  objects
- radiator level, polarity, delay, and analog HPF/LPF crossover response
- mesh scale, air density, and symmetry mode

`BoundaryLabSession.solve()` delegates to `solve()` or `solve_frequencies()`.
`solve_stream()` launches a worker thread, uses `on_frequency_result` to publish
per-frequency `FrequencyResult` objects, attaches complex observation pressure
and native diagnostics when the Boundary Lab result type permits extension, and
stops when cancellation is requested.

## Native Helper Split

The native implementation is split so the Python boundary remains explicit and
testable:

- `metal/geometry.py` converts grid and function-space metadata into validated
  NumPy buffers. It enforces zero-based indices, P1/DP0 DOF counts, float32 and
  int32 buffer shapes, triangle areas/normals, and reduced-domain native
  symmetry rules.
- `metal/session.py` owns the language-neutral IPC schema. It defines payload
  dataclasses, binary array descriptors, manifest writers/readers, and Python
  contract validation.
- `metal/native.py` discovers Swift/helper availability, creates temporary
  session work directories, writes operation manifests and binary inputs, calls
  the helper, reads result manifests, and cleans temporary artifacts unless
  asked to keep them.
- `metal/HornlabMetalBemNative.swift` is the Swift script entrypoint used as a
  fallback. It runs the SwiftPM helper package.
- `metal/native_helper/Sources/HornlabMetalBemNative/main.swift` is the helper
  implementation. It validates manifests, runs Metal kernels, calls Accelerate
  dense solve, writes binary outputs, and returns result manifests.

The regular dense-assembly kernel is selected by
`HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL`. The default `pair_atomic`
runs one thread per triangle pair, computes the pair's 36 kernel evaluations
once, and scatters 3x3 blocks into A with atomic float adds; atomic ordering
makes it nondeterministic at float32 rounding level between runs. `entrywise`
(one thread per matrix entry, bit-reproducible, ~2x slower assembly) and
`block_staged` (pair blocks staged through a large intermediate buffer) remain
selectable for A/B comparison and deterministic debugging.

Runtime discovery prefers an explicit helper executable, then
`HORNLAB_METAL_BEM_NATIVE`, then compiled SwiftPM binaries under
`metal/native_helper/.build/{release,debug}/HornlabMetalBemNative`. If no helper
binary is present, it falls back to Swift via `HORNLAB_METAL_BEM_SWIFT` or
`swift` on `PATH` and the script entrypoint.
