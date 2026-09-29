"""Fail loudly when a native-helper test would run a stale Swift binary.

``discover_native_runtime`` only logs a warning when the helper binary is older
than ``main.swift``. Under pytest that warning is invisible, and a green run then
validates the previous build (a September run once passed against a helper built
a week before the source it claimed to test). The validation tests call
``require_fresh_native_helper`` instead of skipping on their own.

The check compares the helper's mtime with every Swift build input, the same
rule ``scripts/build_metal_native_release.py`` uses to decide whether to rebuild.
It applies only to the in-tree ``swift-package`` helper: an explicitly chosen
helper (``HORNLAB_METAL_BEM_NATIVE`` or ``MetalNativeRuntimeConfig.helper_executable``)
is the caller's responsibility.

Trap: SwiftPM does not relink after an edit that leaves the object code unchanged
(a comment), so the helper stays older than the source although it is current.
``setup.py`` touches the helper after its build for the same reason.
"""

from __future__ import annotations

import pytest

from hornlab_metal_bem.metal import discover_native_runtime

# Wheel installers extract files one after another, so a helper can look a
# moment older than its source even when both came from one build.
_MTIME_SLACK_S = 2.0


def stale_helper_message(status) -> str | None:
    """Return the failure text when the in-tree helper is older than its sources."""
    if not status.available or status.helper_source != "swift-package":
        return None
    helper = status.helper_executable_path
    package_dir = status.native_package_dir
    inputs = [
        *sorted((package_dir / "Sources").rglob("*.swift")),
        package_dir / "Package.swift",
    ]
    newer = [
        path.name
        for path in inputs
        if path.is_file()
        and path.stat().st_mtime > helper.stat().st_mtime + _MTIME_SLACK_S
    ]
    if not newer:
        return None
    return (
        f"native helper {helper} is older than {', '.join(newer)}; "
        f"run `swift build -c release` in {package_dir} so the tests exercise "
        "the current source (if that reports nothing to relink, as after a "
        "comment-only edit, `touch` the helper afterwards: setup.py does the same)"
    )


def require_fresh_native_helper():
    """Skip when the helper cannot run; fail when it is older than its sources."""
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    message = stale_helper_message(status)
    if message is not None:
        pytest.fail(message, pytrace=False)
    return status
