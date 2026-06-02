# hornlab-metal-bem

Apple Metal accelerated acoustic BEM solver extracted from HornLab.

This repository packages the fastest currently validated HornLab solver path:

- corrected native resident Metal dense assembly
- Accelerate `cgesv` dense solve
- native impedance and source surface-pressure reductions
- batched native field output
- resident observation buffer reuse
- stable default Metal dispatch at 64 threads per threadgroup

The package keeps the proven `hornlab_solver` import for compatibility and also
exports the forward-looking `hornlab_metal_bem` namespace for new integrations.

## Status

This is an Apple Silicon/macOS solver backend. It is intended to replace or
augment local acoustic BEM backends on Apple Silicon, not NVIDIA CUDA backends
on Windows.

The solver uses a NumPy-only mesh/grid/function-space loader and does not
depend on `bempp-cl`. There is no OpenCL/Bempp fallback path in this package.

## Fastest Validated Path

Use the Bempp-free public Metal namespace:

```python
from hornlab_metal_bem import native_config, solve

config = native_config()

result = solve("waveguide.msh", config)
```

The compatibility namespace remains available for existing HornLab callers:

```python
from hornlab_solver import SolveConfig, solve
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
python -m pytest tests/test_config.py tests/test_metal_native.py -q
```

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
