# Mesh convergence ladder — OSSE freestanding

One geometry at four resolutions, for `scripts/mesh_convergence_study.py`.
Only the `[mesh]` block differs: `angular_segments` and `length_segments` scale
UP by a linear factor, `throat_res_mm` / `mouth_res_mm` / `rear_res_mm` scale
DOWN by the same factor. Everything under `[profile]` and `[cross_section]` is
identical, so any difference in the solved result is mesh resolution alone.

| level | linear factor | triangles | DOF | max edge | 6-el/wl limit |
|-------|---------------|-----------|-----|----------|---------------|
| L1 | 1.0 | 2,766 | 1,385 | 42.4 mm | 1,347 Hz |
| L2 | 1.5 | 5,820 | 2,912 | 29.3 mm | 1,951 Hz |
| L3 | 2.0 | 10,078 | 5,041 | 21.9 mm | 2,611 Hz |
| L4 | 3.0 | 22,256 | 11,130 | 16.5 mm | 3,466 Hz |

Regenerate (needs a gmsh-capable environment — the mesher's, not this repo's):

    python -m hornlab_mesher.cli L1.toml -o L1.msh --allow-large-mesh

The mesher warns "outer wall self-intersects near the throat" on this geometry.
That is the known thin-wall/curvature guard and affects the OUTER wall only; the
acoustic inner surface is unaffected, so it does not invalidate the study.
