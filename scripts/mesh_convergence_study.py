"""Measure what a mesh's frequency limit actually costs you, in charts.

The BEM mesh warning ("supports the solve only to N Hz") comes from a
6-elements-per-wavelength rule applied to the single worst edge in the mesh.
It is routinely read as a validity cliff -- results above N Hz being "invalid".
They are not. This script measures the real thing: solve ONE geometry at several
mesh resolutions and ask how far the answers actually move.

WHAT THIS SCORES, AND WHY
-------------------------
Not absolute SPL. Absolute level differs in the real world anyway, so a dB
offset between two meshes is not the error anyone cares about. What matters is
whether the *charts* agree:

  * FR shape        -- on-axis response with its constant offset removed.
                       Does the curve wiggle the same way? NOTE this metric is
                       relative to the frequency set you pass, because the
                       offset is the mean over those frequencies; the
                       directivity and beamwidth metrics are set-independent,
                       so prefer them when comparing across runs.
  * Directivity     -- each polar referenced to its own on-axis value, compared
                       over the main lobe only.
  * -6 dB beamwidth -- what a horn designer actually reads off the chart.

And explicitly NOT null depth. An rms taken over all angles is dominated by
deep nulls that nobody listens to: in the 2026-08-27 run, one level scored
5.07 dB at 16 kHz purely from a 90 deg null reading -45.7 dB against the
reference's -25.2 dB. Restricting to the main lobe removes that artefact.

RESULT, 2026-08-27 (OSSE freestanding, M1 Max, native Metal)
------------------------------------------------------------
    level    DOF   max edge   6-el/wl limit   solve
    L1     1,385    42.4 mm       1,347 Hz     0.6 s
    L2     2,912    29.3 mm       1,951 Hz     2.1 s
    L3     5,041    21.9 mm       2,611 Hz     8.8 s
    L4    11,130    16.5 mm       3,466 Hz    86.0 s

Against L4, over 500 Hz - 16 kHz:

  * L3 stays within 0.57 dB of L4's normalised polar shape at EVERY frequency
    out to 16 kHz -- six times its own stated limit. FR shape rms 0.28 dB,
    beamwidth within 2.0 deg. Converged, whatever the rule says.
  * L1 is fine to ~4.5x its limit (1.14 dB at 6 kHz), then has genuinely bad
    frequencies at 5.9-7.4x: beamwidth off by +12.1 deg at 8 kHz and -17.3 deg
    at 10 kHz.
  * The error is ERRATIC, not monotone. L1 is much worse at 10 kHz (3.79 dB)
    than at 16 kHz (0.60 dB). Ratio-to-limit predicts nothing at a given
    frequency, so neither "invalid above the limit" nor "always fine" holds.

So the 6-el/wl limit is conservative by roughly 3-4x. Under ~3x, deviations stay
below ~0.5 dB and ~1 deg of beamwidth; 4-7x has isolated bad frequencies; beyond
that it is unpredictable rather than uniformly bad.

CAVEAT, state it whenever citing this: the reference L4 is itself only "valid"
to 3,466 Hz, so above that this measures convergence between two unconverged
solutions rather than truth. The L3-L4 agreement is real evidence of
convergence, but a trustworthy 16 kHz reference needs ~3.6 mm max edge, about
450k triangles, which the dense path cannot do.

USAGE
-----
    python scripts/mesh_convergence_study.py --mesh L1.msh --mesh L2.msh ... \
        --reference-last

Meshes must be the SAME geometry at different resolutions, coarsest first; the
last is used as the reference. Generate them by scaling angular_segments /
length_segments up and throat/mouth/rear_res_mm down by the same linear factor.
Requires a gmsh-capable environment only for mesh generation, not for this
script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SPEED_OF_SOUND_M_S = 343.0
ELEMENTS_PER_WAVELENGTH = 6.0
DEFAULT_FREQUENCIES = [
    500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 16000,
]
# Main lobe only. Beyond this the comparison is dominated by nulls.
MAIN_LOBE_MAX_DEG = 60.0


def max_edge_m(loaded) -> float:
    vertices = np.asarray(loaded.grid.vertices)
    elements = np.asarray(loaded.grid.elements)
    points = vertices.T if vertices.shape[0] == 3 else vertices
    tris = elements.T if elements.shape[0] == 3 else elements
    corners = points[tris]
    edges = np.concatenate(
        [
            corners[:, 1] - corners[:, 0],
            corners[:, 2] - corners[:, 1],
            corners[:, 0] - corners[:, 2],
        ]
    )
    return float(np.linalg.norm(edges, axis=1).max())


def six_element_limit_hz(edge_m: float) -> float:
    return SPEED_OF_SOUND_M_S / (ELEMENTS_PER_WAVELENGTH * edge_m)


def beamwidth_6db_deg(spl_db: np.ndarray, angles_deg: np.ndarray) -> float:
    """Full -6 dB angle, linearly interpolated. NaN if the arc never drops."""
    target = spl_db[0] - 6.0
    for i in range(1, spl_db.size):
        if spl_db[i] <= target:
            a0, a1 = angles_deg[i - 1], angles_deg[i]
            s0, s1 = spl_db[i - 1], spl_db[i]
            if s0 == s1:
                return float(2.0 * a0)
            return float(2.0 * (a0 + (s0 - target) * (a1 - a0) / (s0 - s1)))
    return float("nan")


def solve_level(mesh_path: Path, frequencies: np.ndarray, angle_count: int):
    import hornlab_metal_bem as mb

    loaded = mb.load_mesh(str(mesh_path))
    config = mb.native_config(
        observation=mb.ObservationConfig(
            planes=["horizontal"],
            distance_m=2.0,
            angle_min_deg=0.0,
            angle_max_deg=90.0,
            angle_count=angle_count,
            origin="mouth",
        )
    )
    started = time.time()
    result = mb.solve_frequencies(loaded, frequencies, config)
    elapsed = time.time() - started

    pressure = np.asarray(result.pressure_complex)[:, 0, :]
    spl = 20.0 * np.log10(np.abs(pressure) + 1e-300) + 94.0
    edge = max_edge_m(loaded)
    return {
        "name": mesh_path.stem,
        "dofs": int(result.mesh_info.n_vertices),
        "triangles": int(result.mesh_info.n_triangles),
        "max_edge_mm": edge * 1000.0,
        "limit_hz": six_element_limit_hz(edge),
        "seconds": elapsed,
        "angles_deg": np.asarray(result.observation_angles_deg),
        "spl_db": spl,
    }


def report(levels: list[dict], frequencies: np.ndarray) -> dict:
    reference = levels[-1]
    others = levels[:-1]
    angles = reference["angles_deg"]
    lobe = angles <= MAIN_LOBE_MAX_DEG

    print(f"{'level':<8}{'DOF':>8}{'max edge':>11}{'6-el/wl limit':>15}{'solve':>9}")
    for level in levels:
        print(
            f"{level['name']:<8}{level['dofs']:>8}{level['max_edge_mm']:>9.1f} mm"
            f"{level['limit_hz']:>12.0f} Hz{level['seconds']:>8.1f}s"
        )
    print(f"\nReference: {reference['name']}. NOTE it is itself only 'valid' to "
          f"{reference['limit_hz']:.0f} Hz;\nabove that this measures convergence, "
          "not absolute truth.")

    def fr_shape(level):
        on_axis = level["spl_db"][:, 0]
        return on_axis - on_axis.mean()

    print("\nFR SHAPE deviation from reference (constant offset removed), dB")
    header = "".join(f"{lvl['name']:>9}" for lvl in others)
    print(f"{'Hz':>7} |{header}")
    ref_shape = fr_shape(reference)
    for i, freq in enumerate(frequencies):
        row = "".join(f"{fr_shape(l)[i] - ref_shape[i]:>+9.2f}" for l in others)
        print(f"{int(freq):>7} |{row}")
    rms = "".join(
        f"{np.sqrt(np.mean((fr_shape(l) - ref_shape) ** 2)):>9.2f}" for l in others
    )
    print(f"{'rms':>7} |{rms}")

    print(f"\nDIRECTIVITY SHAPE deviation, rms over 0-{MAIN_LOBE_MAX_DEG:.0f} deg, dB")
    print("(each polar referenced to its own on-axis; nulls beyond the lobe excluded)")
    print(f"{'Hz':>7} |{header} | vs coarsest limit")
    for i, freq in enumerate(frequencies):
        cells = []
        for level in others:
            a = level["spl_db"][i] - level["spl_db"][i][0]
            b = reference["spl_db"][i] - reference["spl_db"][i][0]
            cells.append(np.sqrt(np.mean((a[lobe] - b[lobe]) ** 2)))
        ratio = freq / levels[0]["limit_hz"]
        print(f"{int(freq):>7} |" + "".join(f"{c:>9.2f}" for c in cells)
              + f" | {ratio:>7.1f}x")

    print("\n-6 dB BEAMWIDTH error vs reference, degrees")
    print(f"{'Hz':>7} |{header}")
    for i, freq in enumerate(frequencies):
        ref_bw = beamwidth_6db_deg(reference["spl_db"][i], angles)
        cells = "".join(
            f"{beamwidth_6db_deg(l['spl_db'][i], angles) - ref_bw:>+9.1f}"
            for l in others
        )
        print(f"{int(freq):>7} |{cells}")

    return {
        "frequencies_hz": frequencies.tolist(),
        "levels": [
            {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in level.items()
            }
            for level in levels
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mesh", action="append", required=True, type=Path,
                        help="mesh of the SAME geometry; repeat, coarsest first")
    parser.add_argument("--frequencies", type=float, nargs="*",
                        default=DEFAULT_FREQUENCIES)
    parser.add_argument("--angle-count", type=int, default=19)
    parser.add_argument("--json", type=Path, help="write raw results here")
    args = parser.parse_args()

    if len(args.mesh) < 2:
        parser.error("need at least two meshes; the last is the reference")

    frequencies = np.asarray(args.frequencies, dtype=float)
    levels = []
    for path in args.mesh:
        level = solve_level(path, frequencies, args.angle_count)
        print(f"solved {level['name']}: {level['dofs']} DOF in "
              f"{level['seconds']:.1f}s", flush=True)
        levels.append(level)

    payload = report(levels, frequencies)
    if args.json:
        args.json.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
