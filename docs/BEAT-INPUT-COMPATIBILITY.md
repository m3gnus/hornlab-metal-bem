# BEAT-compatible inputs and outputs

This package accepts BEAT's solve-request envelope for **exterior** problems
and answers in BEAT's result format, so an application that drives BEAT's
worker can use this solver for those jobs without changing its request or
result code. Interior FEM and coupled FEM-BEM jobs stay with BEAT.

Two entry points share one implementation:

- `solve_compiled(request, *, base_dir=".", cancel_cb=None)` returns a Python
  `CompiledResult`.
- `MetalEngineWorker` has the client surface of BEAT's `EngineWorker`
  (`worker_info`, `ensure_started`, `submit`, `terminate`) and yields BEAT
  worker protocol v1 events carrying `system_result` v2 records.

Neither is selected implicitly. Existing `solve`, `solve_frequencies`,
`SolveConfig` and Boundary Lab paths are unchanged. Install the package with
its `beat-contract` extra to supply BEAT's Python validator; the adapter loads
it only when called and never launches Julia.

```python
from hornlab_metal_bem import MetalEngineWorker

worker = MetalEngineWorker()            # relative mesh files resolve against cwd
worker.ensure_started()
info = worker.worker_info               # BEAT "ready" announcement
for event in worker.submit("request.json"):   # or an inline request dict
    if event["type"] == "result":
        record = event["result"]        # BEAT system_result v2
```

```python
from hornlab_metal_bem import solve_compiled

result = solve_compiled(request, base_dir="meshes")
pressure = result.pressure["far"]       # (frequency, excitation, point)
z = result.quantities["zrad"].values    # (frequency, receiver, source)
```

## Supported requests

The request uses `schema_version: 1` and compiled-system v1, v2 or v3, and must
pass BEAT's validator. The system has exactly one `unbounded_air` region, no
interfaces, no region losses and no volume groups. The region owns one or more
`bem_surface` meshes; they are combined in region order. A physical tag that
repeats across meshes is given a fresh solver tag, so tags need only be unique
within a mesh. Every triangle belongs to exactly one declared `rigid` or
`moving` boundary, and boundary parameters must be empty.

Mesh coordinates are scaled by `scale_to_m`, then translated by
`translation_m`; observation points are already global metres. File paths
resolve inside `base_dir`; packed `mesh_data` supports linear triangles only.
The combined surface must consist of closed, consistently outward-wound
solids, checked per connected shell. No normal repair or vertex merging is
applied.

### Components and ports

Every moving boundary belongs to exactly one component.

- `ideal_velocity_source` with `normal_velocity` ports, each prescribing
  1 m/s. Parameters: `motion_profile` (`uniform_normal` or, from v2,
  `rigid_translation` with `motion_axis`) and `boundary_motion_weights`
  (positive multipliers by boundary ID). Components may use different
  profiles. A rigid-translation face moves with `weight * (n . axis)`.
- `electrodynamic_transducer` (v3) with `voltage` ports, driven at
  `solver_options.transducer_reference_voltage_v` (default 2.83 V, applied as
  the phasor amplitude). Required: `re_ohm`, `le_h`, `bl_n_per_a`, `mmd_kg`,
  `cms_m_per_n`, `rms_n_s_per_m`, `motion_axis`. Optional:
  `boundary_motion_signs` (+/-1), `boundary_motion_weights`,
  `semi_inductance` (`re_prime_ohm`, `leb_h`, `le_h`, `ke_semi_h`, `rss_ohm`,
  `enabled`), `lumped_sealed_rear_chamber` (`volume_m3`, `projected_area_m2`,
  `enabled`), and the symmetry fields `surface_completion_factor`,
  `physical_driver_orbit_count`, `symmetry_role`, `fractional_symmetry_axes`
  with BEAT's consistency rules.

`passive_radiator` is refused, as in BEAT.

### Motion basis and the driver network

The solve uses one unit-motion column per transducer (compiled order, driven or
not), then one per requested ideal source (port order, once per component).
Unrequested ideal sources stay rigid. All columns share one assembly and
factorization per frequency (`solve_multi_source`).

From the P1 surface pressure of each column, the host forms in float64 the
generalized load `Z = F^T P`, where `F` holds nodal force coefficients
`c * v_f * A_f / 3` (`c` is the physical copy count for ideal rows and the
surface completion factor for transducer rows). With `Zm`, `Ze` and `Bl` per
transducer it solves BEAT's network

```text
(diag(Zm) + Z_DD) u_D - diag(Bl) i = -Z_DS u_S
                  diag(Bl) u_D + diag(Ze) i = V
```

Undriven coils are shorted (V = 0). Pressure, boundary pressure and boundary
Neumann data are the basis columns combined with the solved amplitudes.

### Images

| `symmetry` | Native solve | Physical copies |
|---|---|---|
| `off` | full model | 1 |
| `x` | `native_symmetry_plane="yz"` (X=0 mirror) | 2 |
| `xy` | `native_symmetry_plane="yz+xz"` (X=0 and Y=0) | 4 |
| `ground` | `ground_plane="xz"` (rigid Y=0) | 1 (the image is not a radiator) |

For `x`/`xy`, the representative may be cut by an image plane (a fractional
driver, `surface_completion_factor > 1`) or be a complete body off the plane
whose mirrors are further physical drivers (`physical_driver_orbit_count > 1`);
the adapter sets `native_symmetry_complete_images` so the native validator
accepts both. Vertices within `max(1e-9, 1e-6 * model size)` of an image plane
are snapped onto it and the mesh must lie in the positive fundamental domain.
An open edge is accepted only when both ends lie on an active image plane, and
no face may lie on one. Motion axes must lie in every active symmetry plane,
using BEAT's `1e-8` tolerance on normalized components. The native config owns
this validation; native callers keep the default `1e-9` tolerance.
`ground` requires the whole radiator at Y >= 0, refuses faces flat on Y=0 and
honours `ground_plane_min_clearance_m`. Images other than these are refused.

### Solver options

`bem_backend` must be `metal` and `precision` `float32` (both may be omitted).
The default worker uses the standard real-wavenumber formulation. A host that
needs the native complex-wavenumber formulation may explicitly construct
`MetalEngineWorker(native_formulation="complex_k", complex_k_shift=0.005)`
or pass those keyword arguments to `solve_compiled`. This is a host-side
extension, not a BEAT request option; result diagnostics record the effective
formulation and shift. Invalid formulations and shifts are refused.
For hosts that already display native provisional results, an optional
`native_progress_cb(index, frequency_hz, entry)` receives the native
unit-velocity callback as frequencies complete. This callback is a host-side
extension; it does not change the worker protocol's result-event ordering.
`phasor_convention` may be `exp(-i omega t)` (default) or `exp(+i omega t)`;
the native solve always runs in `exp(-i omega t)` and the outputs are
conjugated for the positive convention, which is exact for real drives.
`regular_quadrature_mode` must be `fixed`. `quadrature_order` and
`singular_order` are accepted only at BEAT's default of 4: the native
corrected assembly uses its own fixed rules regardless, so other orders are
refused rather than ignored. Frequencies may repeat; each distinct frequency is
solved once and results keep the request order.

## Outputs

| Quantity | Unit | Axes after frequency | Notes |
|---|---|---|---|
| `exterior_pressure` | Pa | excitation, observation | `options.points_m` required, untargeted |
| `bem_boundary_pressure` | Pa | excitation, bem_node | combined-mesh vertex order; unreferenced vertices are 0 |
| `bem_boundary_neumann` | Pa/m | excitation, bem_face | `i rho omega v_n` (exp(-i omega t)); rigid faces 0 |
| `radiation_impedance` | N*s/m | radiator | ideal self loads in compiled order; every ideal source must be excited |
| `radiation_impedance_matrix` | N*s/m | receiver_component, source_component | targets or all basis components in compiled order; BEAT metadata |
| `diaphragm_velocity` | m/s | excitation, transducer | transducer requests only |
| `voice_coil_current` | A | excitation, transducer | transducer requests only |

The matrix metadata carries `component_ids`, `kinds`, `row_weights`,
`surface_completion_factors`, `physical_driver_orbit_counts`, the signed
effective volume areas and their cancellation ratio/flags, `definition`,
`phasor_convention`, and per-frequency `reciprocity_max_rel` and
`passivity_min_eig`, all defined as in BEAT's compiled-system contract.

### `CompiledResult`

Every quantity array has a leading frequency axis in request order.
`pressure` maps each `exterior_pressure` output ID to (frequency, excitation,
point); `quantities` holds every output as a `CompiledQuantity` (`id`,
`quantity`, `unit`, `axes`, `values`, `metadata`, `target_id`).
`native_results` holds one native `SolveResult` per motion-basis column
(`motion_basis_component_ids`), always in exp(-i omega t).

Both direct results and worker records preserve
`diagnostics.engine_provenance` with `schema_version: 1`, following the
[BEAT Run Provenance contract](https://github.com/JWSound/BEAT_Engine/blob/main/docs/BEAT%20Run%20Provenance.md).
This subset records the implementation and package version, Python/NumPy and
machine/kernel identity, the executed native helper's content hash (or Swift
script and interpreter hashes), submitted solver options, effective native
numerical settings, and input mesh hashes sampled at solve entry. Packed
meshes hash their received JSON payload; file meshes hash the original file
bytes. Mesh transforms are retained alongside these hashes. Each frequency
record receives an independent copy.

At solve entry the engine record hashes all Python source files and records a
Git commit only when the installed distribution owns the imported package and
pip recorded a Git revision. A source checkout shadowing an installed wheel
does not inherit that wheel's version or revision. The helper hash identifies
the executable actually used; the Python source hash distinguishes installed
edits from the recorded commit. Dirty status and the helper's device identity
remain unknown. Dependency environment/library hashes and thread counts are
not collected. Local paths are omitted. Inputs must remain unchanged during a
solve.

### Worker events

`MetalEngineWorker.worker_info` is a BEAT `ready` announcement that passes
`beat_engine.beat_contract.worker.validate_worker_ready` and
`negotiate_submission`. It advertises request v1, compiled-system v1-v3,
result v2, `solve` only, `float32`, `exterior_bem`, both source profiles,
both component kinds, both phasor conventions, `radiation_impedance_matrix`
and marker-file cancellation, with `backends.metal.available` from the native
runtime probe. Protocol clients require `engine.name == "BEAT Engine"`; the
announcement also carries `implementation: {name: "hornlab-metal-bem"}`.

Each submission yields a `status` event, then one `result` event per
frequency and `completed` with `solved_count`. Records follow BEAT exactly:
`schema_version: 2`, `freq_hz` at solver precision, `excitation_port_ids`,
`quantities` in output order with `target_id`, `axes`, base64 little-endian
C-order `values` and `metadata`, and `diagnostics` with `precision`,
`bem_backend`, `symmetry`, `phasor_convention`, `formulation`, `timings` and,
for transducers, `transducer_reference_voltage_v` and the network residual.
Pressure and boundary traces use `complex64`; the matrix, diaphragm velocity
and coil current use `complex128`, as BEAT does.

`submit` returns a `MetalSubmission` whose solve runs on its own thread.
`close()` may be called from any thread, including while another thread is
blocked reading; it cancels the job and returns only after the job has
stopped, after which the submission publishes nothing. The native helper
notices cancellation at its next frequency checkpoint and is then killed, so
`close()` can wait up to one frequency's solve. `terminate()` closes the
active submission. Concurrent closers all wait for shutdown. One job runs at a
time; `submit` waits for the previous one to finish or be closed, and startup
waits until termination has finished clearing the announcement. Drain or close
every submission.

Differences from BEAT's worker: the job runs in-process; all results are
emitted after the whole sweep (a cancelled job reports `solved_count: 0`); the
`bem_field` operation is not offered; and request or solver errors become
`failed` events with code `worker_request_failed`.

## Cancellation

An optional `cancel_path` (absolute, or relative to `base_dir`) and
`cancel_cb` are checked before mesh I/O, before submission and after every
native frequency. Cancellation raises `InterruptedError` from
`solve_compiled` (a `cancelled` event from the worker); no incomplete basis is
returned. Mesh staging uses a temporary directory that is always removed.
Cancellation cannot interrupt a native matrix assembly already in progress.

## Scope and validation

This is input/output interoperability for BEAT's exterior physics, solved by
this package's own kernels. It does not claim numerical equivalence with
BEAT's Burton-Miller assembly: compare both engines on the qualification set
before treating them as interchangeable for a design.

Portable tests cover official-schema refusals, BEAT's negotiation and event
validators, submission lifecycle, symmetry and ground accounting, and the
driver network against closed-form single-driver solutions using a synthetic
linear BEM response. Native qualification
(`tests/test_compiled_native_qualification.py`, Apple Silicon) compares an
oscillating-sphere transducer with the analytic dipole load and field,
symmetry representatives (cut, whole-driver copies, cut plus copy) with the
explicitly mirrored full model, a coupled pair with a shorted undriven
driver, both phasor conventions, and colliding tags across meshes with one
merged mesh.
