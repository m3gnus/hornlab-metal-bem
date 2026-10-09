# Opt-in BEAT compiled inputs

`solve_compiled(request, *, base_dir=".", cancel_cb=None)` accepts the official
BEAT JSON solve-request envelope and translates its supported exterior subset
into this package's existing native solver. `compiled_capabilities()` describes
that subset without loading BEAT or probing hardware. Existing `solve`,
`solve_frequencies`, `SolveConfig`, capability defaults and Boundary Lab paths
are unchanged. This adapter is never selected implicitly.

Install this package with its `beat-contract` extra to supply the exact official
BEAT Python validator used by WG. Importing or using legacy APIs does not need
BEAT. The new API loads the validator only when called and never launches Julia.

```python
import json
from pathlib import Path
from hornlab_metal_bem import solve_compiled

request = json.loads(Path("request.json").read_text())
result = solve_compiled(request, base_dir=Path("meshes"))
pressure = result.pressure[request["outputs"][0]["id"]]
# pressure.shape == (frequency_count, excitation_count, point_count)
```

The input uses `schema_version: 1` and compiled-system v1 or v2. One
`bem_surface` mesh and one `unbounded_air` region are supported. Every triangle
must belong to exactly one declared rigid/moving boundary. Each moving boundary
belongs to exactly one `ideal_velocity_source`; a source can own multiple tags.
Ports are `normal_velocity` ports. Each requested port is independently driven
at **1 m/s** while all other prescribed sources are fixed at zero velocity.
The adapter does not apply an acceleration basis, peak/RMS conversion or LEM.

All sources must use one motion profile: uniform normal motion (empty
parameters or `motion_profile: "uniform_normal"`) or, in v2, `rigid_translation` with a
finite nonzero global `motion_axis` for every source. Axes are normalized once
and applied through the native explicit axis API without an observation-frame
sign vote. Mesh coordinates are scaled by `scale_to_m`, then translated by
`translation_m`. Observation points are already in the global metre frame;
they are neither rescaled nor translated. File paths resolve inside `base_dir`;
packed `mesh_data` supports linear triangles only. Meshes must be complete,
closed, consistently outward oriented solids. No normal repair is applied.

Only untargeted `exterior_pressure` outputs with a
nonempty `options.points_m` list of `[x,y,z]` coordinates are supported. Points,
frequencies, excitation IDs and output IDs retain the request order. Unequal
point counts across outputs are supported. Complex pressure remains in Pa per
unit velocity with `phasor_convention: "exp(-i omega t)"`. `CompiledResult` is a
Python pressure-basis result, **not** BEAT's worker protocol or typed result
wire format. `native_results` retain the old solver diagnostics and quantities.

Options are deliberately restricted. `symmetry` must be `off`, and
`regular_quadrature_mode` must be `fixed`. Unknown numerical options are errors;
metadata is advisory. Backend and precision options must agree with the package:

- Old Metal: `bem_backend: "metal"`, `precision: "float32"`. The native corrected
  assembly and its fixed quadrature rules are retained. Explicit BEAT
  `quadrature_order`/`singular_order` overrides are refused because this solver
  cannot honor them. Region sound speed and density are passed explicitly.
- BEM++: `bem_backend: "cpu"` selects the native Numba CPU assembly;
  `precision: "float32"`/`"float64"` selects single/double native precision.
  The region must request 343 m/s because this legacy solver fixes sound speed.
  Density is passed explicitly. Optional integer `quadrature_order` and
  `singular_order` (1–6) select **BEM++'s native orders**, default 4. These are
  backend numerical controls, not a promise of BEAT quadrature-rule equivalence.

Backend/precision may be omitted (Metal defaults float32, BEM++ float64).
Coupled transducers, Robin boundaries, bulk losses, FEM interfaces, ground or
symmetry images, mixed motion profiles and non-pressure output requests are
refused. This is input interoperability for the supported subset, not full
BEAT physics or numerical equivalence across engines.

An optional `cancel_path` is checked before mesh I/O/submission, between
excitations and after every native frequency. Relative markers use `base_dir`.
`cancel_cb`, when supplied, raises to cancel at the same checks. Cancellation
raises `InterruptedError` (or the caller's exception); no incomplete basis is
returned. Mesh staging uses a temporary directory that is removed on success,
cancellation or failure. Cancellation cannot interrupt a native matrix
assembly already in progress.

Validation includes official-schema refusal tests and numerical comparisons of
normal/axial, multiple-source requests against independently constructed direct
native calls. These comparisons establish adapter equivalence within a backend;
they do not qualify installed WG candidates or HBB retirement.
