# Native resident solve plan

Date: 2026-06-02

Scope: `hornlab-solver` experimental native Metal standard-Neumann path only.
This note intentionally does not edit `main.swift` or `sweep.py`.

## Current boundary

- `hornlab_solver/metal/session.py` defines the language-neutral file contract
  for `create_session`, `assemble_standard_neumann`,
  `assemble_standard_neumann_batch`, `evaluate_standard_exterior`, and
  `evaluate_standard_exterior_batch`.
- `hornlab_solver/metal/native.py` wraps the Swift helper with
  `MetalNativeStandardSession`.
- `hornlab_solver/sweep.py` currently batches native assembly, reads dense
  `A_re/A_im/rhs_re/rhs_im` files back into Python, solves with SciPy or NumPy,
  then calls the native field batch.
- The Swift helper already has resident batch contexts for assembly and field,
  but no dense solve operation.

Measured context from existing artifacts:

- `runs/canonical-validation/260602-metal-native-40freq-field-batch-solve/summary.json`
  reports 40-frequency ASRO68 native batch timings:
  `assembly_s=134.10`, `dense_solve_s=60.04`, `directivity_s=3.75`,
  `total_s=199.66`.
- `docs/waveguide-generator/_public/research/bem-solver-research.md` records a
  later resident-batch ASRO68 40-frequency breakdown:
  `28.02 s assembly`, `57.66 s dense CPU solve`, `0.58 s field`,
  `89.32 s wall`.
- `runs/canonical-validation/260602-mlx-dense-solve-benchmark/summary.json`
  reports a 4554-DOF native-assembled system: SciPy solve `1.49 s`; MLX complex
  solve unavailable; MLX real-block CPU solve available but slower at `3.56 s`.

## Accelerate feasibility

Local Swift probes on this machine:

- `import Accelerate` compiles.
- Deprecated CLAPACK `cgesv_` works with `__CLPK_complex` and returns the
  expected 1x1 solution.
- Modern Accelerate LAPACK works when Swift is compiled with
  `-Xcc -DACCELERATE_NEW_LAPACK`; complex arrays are exposed as opaque pointers,
  so use an explicit interleaved `[Float]` buffer and
  `withUnsafeMutableBytes { OpaquePointer($0.baseAddress!) }`.

Package implication:

- Add `import Accelerate` in the helper.
- Prefer `Package.swift` settings:
  `swiftSettings: [.define("ACCELERATE_NEW_LAPACK")]`,
  `linkerSettings: [.linkedFramework("Accelerate")]`.
- If script fallback must support solve ops, update
  `_native_helper_command(...)` in `native.py` to pass
  `swift -Xcc -DACCELERATE_NEW_LAPACK main.swift ...`, or explicitly make
  solve ops require the compiled package binary.

## Ranked implementation plan

### 1. Add a native dense solve primitive

Files:

- `hornlab_solver/metal/native_helper/Package.swift`
- `hornlab_solver/metal/native_helper/Sources/HornlabSolverMetalNative/main.swift`
- tests in `hornlab-solver/tests/test_metal_native.py`

Swift API:

```swift
struct DenseSolveRun {
    let pressure: [Complex32]
    let implementation: String
    let seconds: Double
    let lapackInfo: Int32
}

func solveDenseAccelerateCgesv(
    aReRowMajor: [Float],
    aImRowMajor: [Float],
    rhsRe: [Float],
    rhsIm: [Float],
    n: Int
) throws -> DenseSolveRun
```

Implementation details:

- Convert separate row-major real/imag matrix arrays to one interleaved
  column-major `[Float]` buffer for LAPACK.
- Convert RHS to one interleaved `[Float]` vector.
- Call `cgesv_` with `nrhs=1`.
- Fail if `info != 0`; include `lapack_info` in result JSON.
- Return pressure as `[Complex32]`.

Validation:

- Tiny 1x1 and 2x2 Swift executable tests comparing against NumPy.
- Existing tiny native assembly test: solve assembled matrix in Swift and
  compare pressure to `np.linalg.solve` within `1e-5` relative L2.

### 2. Add `assemble_solve_standard_neumann_batch`

Files:

- `hornlab_solver/metal/session.py`
- `hornlab_solver/metal/native.py`
- `main.swift`
- `tests/test_metal_native.py`

Manifest:

```json
{
  "schema": "hornlab.metal.standard.v1",
  "op": "assemble_solve_standard_neumann_batch",
  "session_id": "...",
  "index_base": 0,
  "matrix_layout": "row_major_c",
  "cases": [
    {
      "case_id": "case-0000-100hz",
      "frequency_hz": 100.0,
      "k_real_f32": 1.831832,
      "neumann_dp0": {
        "real_f32": {"path": "...", "shape": [4554], "dtype": "float32", "byte_order": "little", "order": "C"},
        "imag_f32": {"path": "...", "shape": [4554], "dtype": "float32", "byte_order": "little", "order": "C"}
      },
      "outputs": {
        "pressure_real_f32": {"path": "...", "shape": [4554], "dtype": "float32", "byte_order": "little", "order": "C"},
        "pressure_imag_f32": {"path": "...", "shape": [4554], "dtype": "float32", "byte_order": "little", "order": "C"}
      }
    }
  ]
}
```

Result:

```json
{
  "schema": "hornlab.metal.standard.v1",
  "op": "assemble_solve_standard_neumann_batch_result",
  "implementation": "swift_native_resident_metal_assembly_accelerate_solve_batch",
  "session_id": "...",
  "case_count": 40,
  "assembly_seconds": 28.0,
  "dense_solve_seconds": 57.0,
  "wall_seconds": 86.0,
  "resident_reuse": {
    "geometry_buffers": true,
    "assembly_output_buffers": true,
    "lapack_work_buffers": true
  },
  "cases": [
    {
      "case_id": "case-0000-100hz",
      "frequency_hz": 100.0,
      "assembly_seconds": 0.70,
      "dense_solve_seconds": 1.45,
      "lapack_info": 0,
      "pressure_real_f32": "...",
      "pressure_imag_f32": "...",
      "shape": [4554]
    }
  ]
}
```

This removes dense matrix/RHS file writes and Python reads, but still returns
surface pressure files for existing Python impedance and field plumbing. It is
the smallest useful step and should be benchmarked before touching `sweep.py`.

### 3. Add full `solve_evaluate_standard_neumann_batch`

Files:

- Same helper/session/native files as step 2.
- `hornlab_solver/sweep.py` only after the helper op passes tests.

Manifest additions:

```json
{
  "op": "solve_evaluate_standard_neumann_batch",
  "batch_id": "all_observation_planes",
  "observation_points": {"path": "...", "shape": [3, 74], "dtype": "float32", "byte_order": "little", "order": "C"},
  "source_tags_i32": {"path": "...", "shape": [1], "dtype": "int32", "byte_order": "little", "order": "C"},
  "outputs": {
    "observation_pressure_real_f32": {"path": "...", "shape": [40, 74], "dtype": "float32", "byte_order": "little", "order": "C"},
    "observation_pressure_imag_f32": {"path": "...", "shape": [40, 74], "dtype": "float32", "byte_order": "little", "order": "C"},
    "surface_pressure_avg_real_f32": {"path": "...", "shape": [40, 1], "dtype": "float32", "byte_order": "little", "order": "C"},
    "surface_pressure_avg_imag_f32": {"path": "...", "shape": [40, 1], "dtype": "float32", "byte_order": "little", "order": "C"},
    "impedance_real_f32": {"path": "...", "shape": [40], "dtype": "float32", "byte_order": "little", "order": "C"},
    "impedance_imag_f32": {"path": "...", "shape": [40], "dtype": "float32", "byte_order": "little", "order": "C"}
  }
}
```

Native scalar formulas:

- Surface average by tag matches `compute_surface_pressure_avg`: average the
  three P1 vertex pressures per triangle, area-weight over triangles whose
  physical tag matches the requested tag.
- Impedance matches `_compute_impedance`: same area-weighted source pressure
  divided by total source area unless future velocity weights are supplied.

This is the architecture target. It writes no dense matrix/RHS files and does
not write/read surface pressure unless a debug output flag is requested.

### 4. Wire sweep routing

File:

- `hornlab_solver/sweep.py`

Change:

- In the no-callback native batch branch, call
  `session.solve_evaluate_standard_neumann_batch(...)`.
- Keep the callback branch on per-frequency ops until a streaming resident
  callback contract is designed.
- Preserve fallback behavior and leave default Bempp/OpenCL routing unchanged.

## Risks

- LAPACK expects column-major interleaved complex buffers. Current assembly
  arrays are separate real/imag row-major buffers, so the native solve path
  needs a deterministic transpose/interleave copy unless a solve-only assembly
  layout is added.
- Swift script fallback and compiled Swift package may diverge if only the
  package target gets `ACCELERATE_NEW_LAPACK`.
- `cgesv` should match SciPy/Accelerate in speed, not beat it materially. The
  win is avoiding matrix/RHS disk I/O and Python matrix reconstruction, not a
  better dense solver.
- Full resident evaluation must reproduce Python impedance and surface average
  exactly enough for existing `SolveResult` consumers.
- Standard Neumann matrices are not Hermitian or symmetric; do not use
  symmetric/Hermitian LAPACK routines.

## Recommendation

Implement step 1 and step 2 now if the Swift-helper owner can take the changes.
They are bounded, testable, and remove the largest avoidable file round trips
without changing production routing.

Do not jump directly to step 3 until step 2 has an ASRO 40-frequency benchmark
showing the matrix/RHS I/O removal actually moves wall time. If step 2 saves
only a small amount, the full resident operation is still cleaner architecture,
but it is a larger contract change and should be justified by sweep-level data.
