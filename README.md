# hornlab-metal-bem

Apple Metal accelerated acoustic BEM solver extracted from HornLab.

This repository packages the fastest currently validated HornLab solver path:

- corrected native resident Metal dense assembly
- Accelerate `cgesv` dense solve
- native impedance and source surface-pressure reductions
- batched native field output
- resident observation buffer reuse
- stable default Metal dispatch at 64 threads per threadgroup

The current public Python package still imports as `hornlab_solver` during the
first extraction pass. That keeps the proven HornLab implementation and tests
intact while this repository is prepared for standalone and downstream
application integration.

## Status

This is an Apple Silicon/macOS solver backend. It is intended to replace or
augment local acoustic BEM backends on Apple Silicon, not NVIDIA CUDA backends
on Windows.

Current caveat: the fast path still uses `bempp-cl` scaffolding for mesh/grid
and function-space construction. The expensive solve path is native Metal, but
the next packaging milestone is a pure mesh/function-space loader so `bempp-cl`
can become a validation-only optional dependency.

## Fastest Validated Path

Use:

```python
from hornlab_solver import SolveConfig, solve

config = SolveConfig(
    assembly_backend="metal",
    experimental_metal_backend=True,
    metal_backend_fallback="error",
    metal_native_assembly_mode="corrected",
)

result = solve("waveguide.msh", config)
```

Recent ASRO2 corrected-quarter benchmark from HornLab:

- 40 frequencies
- 3 planes x 37 angles
- about 5.14 s wall time
- pressure/directivity parity exact against the corrected baseline
- impedance relative L2 max about `8.24e-9`

## Install For Development

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
swift build -c release --package-path hornlab_solver/metal/native_helper
python -m pytest tests/test_config.py tests/test_metal_runtime.py tests/test_metal_native.py -q
```

## Integration Target

Downstream adapters should translate their application-level simulation config
into `SolveConfig` and call the package through a small backend/session layer.

Recommended backend id: `hornlab_metal`.

## Roadmap

1. Keep this extraction buildable with the current `hornlab_solver` import.
2. Add backend adapter package/modules where downstream applications need them.
3. Replace `bempp-cl` mesh/function-space scaffolding with a pure loader.
4. Move `bempp-cl` into validation extras only.
5. Rename the import namespace only after adapters and tests are stable.
