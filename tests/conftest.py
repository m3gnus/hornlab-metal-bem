"""Fail native-helper tests when the in-tree Swift helper is stale (see native_helper_guard).

Every test module that runs the real helper (``discover_native_runtime(run_smoke_test=True)``
or the guard) is covered, so a stale helper cannot silently validate the previous build. The check
is per module, not per test: every test in such a module errors on a stale helper, including its
pure-Python tests. Modules that never mention the helper (config, mesh, observation tests) are
unaffected.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from hornlab_metal_bem.metal import discover_native_runtime
from native_helper_guard import stale_helper_message

_NATIVE_MARKERS = ("run_smoke_test=True", "require_fresh_native_helper")


@lru_cache(maxsize=None)
def _module_uses_native_helper(path: str) -> bool:
    text = Path(path).read_text(encoding="utf-8")
    return any(marker in text for marker in _NATIVE_MARKERS)


@lru_cache(maxsize=1)
def _stale_message() -> str | None:
    return stale_helper_message(discover_native_runtime(run_smoke_test=False))


@pytest.fixture(autouse=True)
def _fail_on_stale_native_helper(request):
    if _module_uses_native_helper(str(request.node.fspath)):
        message = _stale_message()
        if message is not None:
            pytest.fail(message, pytrace=False)
