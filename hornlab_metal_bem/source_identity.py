"""Identity of the Python sources that are actually imported by this process."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname


_PACKAGE = "hornlab-metal-bem"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _installed_identity(package_dir: Path) -> tuple[str, str | None]:
    """Use PEP 610 only when its distribution owns the imported package.

    An unrelated installed wheel may coexist with a source checkout on
    PYTHONPATH. Its version and commit do not describe that checkout. Editable
    installs and wheels without a VCS record honestly report an unknown commit.
    """
    try:
        dist = metadata.distribution(_PACKAGE)
        raw = dist.read_text("direct_url.json")
        doc = json.loads(raw) if raw else {}
        owns_package = Path(dist.locate_file("hornlab_metal_bem")).resolve() == package_dir.resolve()
        if not owns_package and isinstance(doc, dict):
            info = doc.get("dir_info")
            url = urlparse(doc.get("url", ""))
            if isinstance(info, dict) and info.get("editable") is True and url.scheme == "file" and not url.netloc:
                owns_package = (Path(url2pathname(url.path)) / "hornlab_metal_bem").resolve() == package_dir.resolve()
        if not owns_package:
            return "0+unknown", None
        version = dist.version
        vcs = doc.get("vcs_info", {}) if isinstance(doc, dict) else {}
        commit = vcs.get("commit_id") if isinstance(vcs, dict) and vcs.get("vcs") == "git" else None
        if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
            commit = None
        return version, commit
    except (OSError, ValueError, TypeError, metadata.PackageNotFoundError):
        return "0+unknown", None


def _source_sha256(package_dir: Path) -> str | None:
    """Hash named Python source files, without paths outside the package.

    Length-prefix paths and contents so neither file boundaries nor names can
    collide. A read error leaves identity unknown rather than yielding a hash
    for an incomplete tree. Compiled helper identity is recorded separately
    from the helper that actually executed.
    """
    try:
        def fail(exc: OSError) -> None:
            raise exc

        paths = []
        for directory, subdirs, names in os.walk(package_dir, onerror=fail):
            # Build output and bytecode are identified separately or cannot
            # affect the imported Python source. SwiftPM may link .build.
            subdirs[:] = [name for name in subdirs if name not in {".build", "__pycache__"}]
            if any((Path(directory) / name).is_symlink() for name in subdirs):
                return None
            paths.extend(Path(directory) / name for name in names if name.endswith(".py"))
        paths.sort()
        if not paths:
            return None
        digest = hashlib.sha256()
        for path in paths:
            name = path.relative_to(package_dir).as_posix().encode("utf-8")
            content = path.read_bytes()
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return digest.hexdigest()
    except (OSError, ValueError):
        return None


def solver_source_identity() -> dict[str, Any]:
    """Measure source at solve entry; do not substitute a configured pin."""
    package_dir = Path(__file__).resolve().parent
    version, commit = _installed_identity(package_dir)
    return {
        "name": _PACKAGE,
        "version": version,
        "repository_revision": commit,
        # A PEP 610 commit alone cannot prove the installed files are clean.
        "repository_dirty": None,
        "source_sha256": _source_sha256(package_dir),
    }
