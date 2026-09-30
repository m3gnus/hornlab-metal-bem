# BEM capabilities schema v1

<!-- This copy must match the counterpart in hornlab-metal-bem and hornlab-bempp-bem. -->

`capabilities()` in hornlab-metal-bem and hornlab-bempp-bem returns the shared
`"schema": "hornlab-bem-capabilities"`, `"schema_version": 1` contract.
Validate it against [capabilities-schema.json](capabilities-schema.json).
The top-level keys are `schema`, `schema_version`, `request_schema_version`,
`package`, `package_version`, `request_fields`, `conventions`, and `features`.
`package_version` is the installed distribution version, or `null` without
metadata. `request_fields` lists every `SolveConfig` constructor field.
`request_schema_version` versions that Python request API separately.

Every feature has the same six keys:

- `supported`: boolean package support, independent of host readiness.
- `request_fields`: constructor fields used to request it, including names
  absent from this provider when the feature is unsupported. Explicit frequency
  lists use an entry point instead, so their field list is empty.
- `values`: enumerable choices, or `null` for non-enumerable features. For
  explicit frequencies this lists the public entry point `solve_frequencies`.
- `requires`: prerequisites, with arrays of permitted request values or text
  describing geometric/per-source requirements. Formulation prerequisites
  apply to the feature as a whole.
- `refuses`: conditional restrictions, each with a stable `id`, the related
  `request_fields`, and a `reason` describing the rejected combination.
  A formulation restriction mentioning Burton-Miller applies only to that
  formulation. Tests cover every declared refusal at its actual guard.
- `composes_with`: explicitly verified compositions, each with a `feature` name
  and `values` listing permitted modes or request-field combinations. This
  records tested compositions; an empty list gives no composition guarantee.

Both providers expose exactly the feature names in the JSON Schema, including
`native_symmetry` (requested through `native_symmetry_plane`) and
`assembly_backend` (the provider's own assembly selector). Unsupported optional
features remain objects with `supported: false`; schema identity and version
must be checked before interpreting them. Reports are fresh JSON-compatible
snapshots and do not import numerical backends or probe devices.

Both packages' raw public output uses `e^{-iωt}` with outgoing `e^{+ikr}`:
`conventions.time_convention` is `"exp(-i*omega*t)"` and `outgoing_wave` is
`"exp(+i*k*r)"`. Consumer conversions do not change the package convention.

BEAT keeps its own provider schema (currently capability/request version 2,
with `engine`, `availability_probe`, `solve_modes`, and `backends`). It is
explicitly not this schema. WG must read BEAT through a separate named adapter,
`beat-provider-v2`; the shared BEM reader is `hornlab-bem-capabilities-v1`.
These names specify consumer dispatch boundaries, not new exported solver APIs
or a claim that a WG adapter has been installed by these packages.

The spec, JSON Schema, and shared contract test are mirrored files: update both
repositories together and keep their copies byte-identical.
