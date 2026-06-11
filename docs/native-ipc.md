# Native IPC Contract

The native helper contract is defined on the Python side in
`hornlab_metal_bem/metal/session.py` and validated again by the Swift helper in
`hornlab_metal_bem/metal/native_helper/Sources/HornlabMetalBemNative/main.swift`.
It is a file-based contract: JSON manifests reference raw little-endian binary
arrays by relative path.

## Schema

All current manifests use:

- `schema`: `hornlab.metal.standard.v1`
- `index_base`: `0`
- `matrix_layout`: `row_major_c` for dense matrix payloads
- formulation: `standard_neumann`
- trial/test basis: `P1`
- source basis: `DP0`
- precision: currently written as `complex64`

The schema is intentionally narrow. It covers standard Neumann dense BEM
assembly, dense solve, and exterior field evaluation for package-owned native
Metal execution. The production combined solve operation also accepts
experimental per-case extensions for complex-k assembly and Robin admittance;
these keep the same schema and are ignored by older helpers that do not read
them.

## Manifest and Binary Rules

Each binary array is described by a `BinaryArrayDescriptor`:

- `path` is non-empty, POSIX-style, relative to the session work directory, and
  must not contain `..`.
- `shape` is a non-empty list of positive dimensions.
- `dtype` is `float32` or `int32`.
- `byte_order` is `little`.
- `order` is `C`.

Complex arrays are not stored as interleaved complex values. They are split
into paired `real_f32` and `imag_f32` float32 descriptors with matching shapes.
Python writes arrays with `np.ascontiguousarray(...).tofile()`, and reads result
pairs with little-endian `<f4`.

The create-session geometry manifest contains:

- `mesh.vertices_f32`: shape `(3, n_vertices)`, float32
- `mesh.triangles_i32`: shape `(3, n_triangles)`, int32
- `mesh.physical_tags_i32`: shape `(n_triangles,)`, int32
- `mesh.p1_local2global_i32`: shape `(n_triangles, 3)`, int32
- `mesh.triangle_areas_f32`: shape `(n_triangles,)`, float32
- `mesh.triangle_normals_3xm_f32`: shape `(3, n_triangles)`, float32
- `space.p1_dof_count`: positive integer
- `space.dp0_dof_count`: must equal `n_triangles`
- `assembly_scope.symmetry_plane`: `null`, `yz`, `xz`, `xy`, or `yz+xz`

All triangle indices and DOF indices are zero-based at the Python/native
boundary.

## Key Operations

The helper executable accepts `--smoke` plus operation subcommands. Operation
commands use:

```text
<op> <session.json> <payload.json> <result.json>
```

except `validate_session`, which uses:

```text
validate_session <session.json> <result.json>
```

Supported operation payloads are:

- `create_session`: written as `session.json`; validates geometry, basis,
  quadrature, index base, matrix layout, and symmetry metadata.
- `assemble_standard_neumann`: single-frequency assembly that writes real/imag
  dense matrix and RHS arrays.
- `assemble_standard_neumann_batch`: multi-frequency assembly with one case per
  frequency.
- `assemble_solve_standard_neumann_batch`: multi-frequency assembly plus dense
  solve, writing solved P1 pressure arrays.
- `assemble_solve_evaluate_standard_neumann_batch`: production operation for
  resident assembly, dense solve, exterior field evaluation, optional surface
  pressure output, optional batched field output, impedance, and
  source-pressure reductions.
- `evaluate_standard_exterior`: single case exterior field evaluation from
  supplied P1 pressure and DP0 Neumann arrays.
- `evaluate_standard_exterior_batch`: batched exterior field evaluation.

Batch payloads contain a non-empty `cases` list. Cases include a stable
`case_id`, `frequency_hz`, `k_real_f32`, inputs, and output descriptors. The
combined assembly/solve/field operation may also include `batch_outputs` for a
single row-major `(case_count, n_obs)` field array pair.

`assemble_solve_evaluate_standard_neumann_batch` cases may additionally carry:

- `k_imag_f32`: optional non-negative assembly wavenumber imaginary part.
  When present with `k_real_f32`, assembly uses `k_real + i*k_imag`.
- `field_k_real_f32`: optional real wavenumber for exterior field evaluation.
  Complex-k solves set this to the physical real `k`.
- `impedance_sources`: optional object mapping stringified physical tags to
  `[real, imag]` normalized admittance beta values. The helper maps these tags
  onto DP0 triangles, adds the Robin LHS term `-i*k*V*diag(beta)*P1_to_DP0`,
  and reconstructs total Neumann data for field evaluation.

When `k_imag_f32` is nonzero or `impedance_sources` is present, the helper uses
the Swift reference assembly path for that case and disables pipelined Metal
assembly for the batch.

## Streamed Per-Case Results

`assemble_solve_evaluate_standard_neumann_batch` accepts an optional
`case_results_dir` payload key: a non-empty work-dir-relative POSIX path
without `..`. When present, the helper writes `case-0000.json`,
`case-0001.json`, … into that directory as each case completes, after the
case's binary outputs are on disk. Each streamed manifest is the case's entry
from the final result `cases` list plus a `case_index` key, and is written
atomically (temp file + rename) so a polling reader never observes partial
JSON. The final batch result reports `streamed_case_results: true`.

This is the streaming contract used by `sweep.py` when
`SolveConfig.on_frequency_result` is set: Python launches one helper process
for the whole sweep, tails the per-case manifests to fire the callback per
frequency, and terminates the helper for an early stop — results already
streamed stay valid. `case_results_dir` is rejected together with
`batch_outputs`, because the single batched field array is only written when
the whole batch completes. Helpers that predate this key ignore unknown
payload fields, so the Python side falls back to firing callbacks from the
final batch manifest after the process exits.

## Array Conventions

- Geometry vertices and triangle normals use `3 x N` orientation.
- Triangle connectivity uses shape `(3, n_triangles)`.
- P1 local-to-global mapping uses shape `(n_triangles, 3)`.
- DP0 Neumann rows passed from Python to native use shape
  `(frequency_count, dp0_dof_count)` before each case is split into real/imag
  vectors.
- Observation points are normalized by Python to shape `(3, n_obs)`. Callers may
  supply `(n_obs, 3)` to session methods; `_require_observation_points_3xn()`
  transposes and validates it before writing the IPC buffer.
- Per-case field outputs use shape `(n_obs,)` real/imag arrays.
- Batched field outputs use shape `(frequency_count, n_obs)` real/imag arrays
  and result cases carry `field_row_index` plus `field_batch_shape`.
- Dense assembly matrices are row-major C arrays with shape `(p1_dof_count,
  p1_dof_count)`.
- Surface pressure and impedance reductions are returned in result manifests
  when the helper provides them. Complex scalars use `[real, imag]`; complex
  maps use stringified integer tags mapped to `[real, imag]`.
- Experimental case diagnostics may include `complex_k`,
  `assembly_k_imag_f32`, `field_k_real_f32`, `robin_boundary`, and
  `field_uses_total_neumann`. Dense-solve diagnostics include
  `dense_solve_rcond` and, when finite, `dense_solve_condition_1norm`.

## Runtime and Binary Selection

`metal/native.py` discovers the helper in this order:

1. explicit `MetalNativeRuntimeConfig.helper_executable`
2. `HORNLAB_METAL_BEM_NATIVE`
3. compiled SwiftPM helper at
   `metal/native_helper/.build/release/HornlabMetalBemNative`
4. compiled SwiftPM helper at
   `metal/native_helper/.build/debug/HornlabMetalBemNative`
5. Swift script fallback through `HORNLAB_METAL_BEM_SWIFT` or `swift` on
   `PATH`, invoking `metal/HornlabMetalBemNative.swift`

Runtime availability also requires Apple Silicon macOS, packaged helper assets,
and, when requested, a successful `--smoke` run that creates a Metal device.

## Test Enforcement

Contract tests live in:

- `tests/test_metal_session.py`: schema constants, deterministic JSON shape,
  relative paths, little-endian C-contiguous binary buffers, descriptor
  validation, geometry/field shape rejection, and absence of scratch-path
  leakage.
- `tests/test_metal_native.py`: native discovery, helper/binary selection,
  smoke-test behavior, helper invocation arguments, session validation,
  operation manifest generation, result parsing, batch operations, symmetry
  handling, and native parity checks where the helper can run.
- `tests/test_metal_geometry.py`: geometry buffer validation, zero-based index
  enforcement, shape/dtype conventions, area/normal generation, and symmetry
  preconditions.
- `tests/test_native_symmetry_validation.py`: reduced-domain/native symmetry
  validation helpers.
- `tests/test_boundary_lab_backend.py`: Boundary Lab translation and backend
  session contract.

Maintainers changing `session.py`, `native.py`, the Swift helper, or geometry
buffer shapes should update these tests with the contract change in the same
patch.
