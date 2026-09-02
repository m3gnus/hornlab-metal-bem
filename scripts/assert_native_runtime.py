"""Fail the build when a runner that should cover Metal silently cannot.

Every GPU test in this repository guards itself with
``discover_native_runtime(run_smoke_test=True)`` and skips when the Swift/Metal
helper is unavailable. That is right for ubuntu and windows, where the helper
cannot exist -- but on macOS it turns a lost toolchain into a green run over
far less code rather than into a failure. CI calls this first on macOS so the
loss is one legible error instead of a pile of skips nobody reads.
"""

from __future__ import annotations

import sys

from hornlab_metal_bem.metal import discover_native_runtime


def main() -> int:
    status = discover_native_runtime(run_smoke_test=True)
    if status.available:
        print(
            "Native Swift/Metal helper is available; the GPU tests will run "
            "for real.\n"
            f"  platform     {status.platform_system} {status.platform_machine}\n"
            f"  swift        {status.swift_path} (via {status.swift_source})\n"
            f"  helper       {status.helper_executable_path} "
            f"(via {status.helper_source})\n"
            f"  smoke test   ran={status.smoke_test_ran} ok={status.smoke_test_ok}"
        )
        return 0

    print(
        "Native Swift/Metal helper is UNAVAILABLE on a runner that is supposed "
        "to cover it.\n"
        "\n"
        "Every GPU test would skip itself and this job would pass while "
        "testing far less\n"
        "than it claims to, so it fails here instead. Reasons reported by "
        "discovery:\n",
        file=sys.stderr,
    )
    for reason in status.unavailable_reasons:
        print(f"  - {reason}", file=sys.stderr)
    print(
        "\nIf the runner image genuinely no longer provides Swift or a Metal "
        "device, that is\na real loss of coverage: decide whether to move Metal "
        "onto a self-hosted runner\nrather than deleting this check.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
