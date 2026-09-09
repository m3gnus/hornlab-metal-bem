# Cyclic full-3D m=0 reference prototype

Date: 2026-09-09. Status: **validation-only prototype; exact discrete
equivalence proved, scalable native assembly not yet implemented.**

## Purpose

The existing quarter meshes are not converged at 20 kHz, so they cannot serve
as the qualification ground truth for Axisymmetric. A rotationally periodic
full-3D P1 mesh provides an independent reference family while retaining the
ordinary triangle discretization and the native full-3D quadrature. For an
axisymmetric source, the full solution lies in the discrete `m=0` subspace: all
P1 pressure degrees of freedom related by one mesh-sector rotation have the
same value.

`hornlab_metal_bem.validation.cyclic_m0` now defines that contract:

- `revolve_meridian_p1` produces a closed, conforming cyclic full surface from
  a counter-clockwise meridian, with explicit vertex and triangle orbits.
- `expand_triangle_orbit_values` preserves an axisymmetric source exactly over
  every rotated triangle sector.
- `reduce_cyclic_m0_representative_rows` selects one test row per vertex orbit
  and sums every column orbit. This is the exact discrete `m=0` restriction,
  not a ring-kernel approximation.
- `expand_cyclic_m0_pressure` reconstructs the complete full-3D P1 pressure
  vector expected by the existing field evaluator.
- `load_and_reduce_native_cyclic_m0` applies the restriction to a corrected
  native full-3D assembly as a qualification oracle.

None of these functions changes product routing.

## Equivalence evidence

A focused test constructs a closed cyclic P1 cylinder, applies a source that is
constant on triangle rotation orbits, and assembles the ordinary corrected
full-3D operator with the existing native triangle quadrature at **20 kHz**. The
representative-row system is solved and expanded, then compared with an
independent solve of the complete matrix. The test requires a full residual
below `2e-4` and a full-solution relative L2 error below `3e-4`.

An additional 20 kHz scaling sample used a 12-orbit meridian:

| Sectors | Full P1 DOFs | Reduced DOFs | Full matrix | Reduced matrix | Full-solution relative L2 |
|---:|---:|---:|---:|---:|---:|
| 8 | 82 | 12 | 53,792 B | 2,304 B | 2.17e-5 |
| 16 | 162 | 12 | 209,952 B | 2,304 B | 2.38e-5 |
| 32 | 322 | 12 | 829,472 B | 2,304 B | 2.74e-5 |

This proves the algebraic reduction and source/reconstruction mapping against
the actual full-3D discretization. It does not claim that this small cylinder is
an acoustically converged horn reference.

## Exact implementation blocker

The current native `assemble_matrix_regular` kernel dispatches
`nDof * nDof` entries and writes a full square matrix. Its singular/adjacent
Duffy correction pass also addresses entries in that same full-DOF layout.
Reducing the matrix after this work is numerically useful as an oracle but does
not remove the upper-band memory or assembly cost.

A scalable implementation needs a new, orbit-aware native operation that:

1. dispatches one representative test DOF per meridian vertex orbit;
2. accumulates every rotated trial-sector contribution directly into one
   meridian-orbit column;
3. applies self and adjacent Duffy corrections in the reduced layout, including
   wraparound-sector adjacency;
4. assembles one representative RHS value from orbit-constant triangle data;
5. returns the reduced pressure plus enough orbit metadata to expand it for the
   unchanged full-3D field evaluator.

For scale, 832 meridian orbits with 512 azimuth sectors would imply roughly
426,000 full P1 unknowns and about 1.45 TB for a complex64 dense full matrix,
versus about 5.5 MB for the reduced 832-square matrix. The 512-sector figure is
only an illustration; the azimuthal convergence ladder must determine the
actual 20 kHz sector count.

Implementing only the regular representative-row kernel would be unsafe: it
would appear fast while silently dropping the singular/adjacent corrections
that distinguish the corrected full-3D operator. The next prototype therefore
has to cover both regular and Duffy paths before it may be used as the
upper-band reference.

## Required qualification ladder

Once the orbit-native assembler exists, compare at least two independent axes:

- meridian refinement (for example the existing 416/832 orbits), and
- azimuthal sectors (for example successive doubling selected from wavelength
  and measured convergence, not a fixed assumed count).

The cyclic full result should be compared with both the current full mesh and
Axisymmetric using the unchanged pressure, directivity, and phase gates. Until
those ladders converge, the 20 kHz ground-truth blocker remains open.
