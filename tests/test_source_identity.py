"""Provenance must describe imported sources, not an unrelated installed pin."""

from __future__ import annotations

import json

from hornlab_metal_bem import source_identity


class Distribution:
    def __init__(self, root, *, commit="a" * 40):
        self.root = root
        self.version = "1.2.3"
        self.commit = commit

    def locate_file(self, name):
        return self.root / name

    def read_text(self, name):
        assert name == "direct_url.json"
        return json.dumps({"vcs_info": {"vcs": "git", "commit_id": self.commit}})


def test_only_the_distribution_owning_imported_files_supplies_identity(monkeypatch, tmp_path):
    package = tmp_path / "hornlab_metal_bem"
    package.mkdir()
    monkeypatch.setattr(source_identity.metadata, "distribution", lambda _: Distribution(tmp_path))
    assert source_identity._installed_identity(package) == ("1.2.3", "a" * 40)
    assert source_identity._installed_identity(tmp_path / "other" / "hornlab_metal_bem") == (
        "0+unknown", None,
    )
    monkeypatch.setattr(source_identity.metadata, "distribution", lambda _: Distribution(tmp_path, commit="main"))
    assert source_identity._installed_identity(package) == ("1.2.3", None)


def test_python_source_digest_changes_with_content_and_file_names(tmp_path):
    (tmp_path / "one.py").write_text("x = 1\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "two.py").write_text("y = 2\n")
    first = source_identity._source_sha256(tmp_path)
    assert first and len(first) == 64
    (nested / "two.py").write_text("y = 3\n")
    second = source_identity._source_sha256(tmp_path)
    assert first != second
    (nested / "two.py").rename(nested / "renamed.py")
    assert source_identity._source_sha256(tmp_path) != second


def test_editable_distribution_uses_its_own_checkout_version(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    package = checkout / "hornlab_metal_bem"
    package.mkdir(parents=True)

    class Editable(Distribution):
        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps({"url": checkout.as_uri(), "dir_info": {"editable": True}})

    monkeypatch.setattr(source_identity.metadata, "distribution", lambda _: Editable(tmp_path / "site"))
    assert source_identity._installed_identity(package) == ("1.2.3", None)
    assert source_identity._installed_identity(tmp_path / "other" / "hornlab_metal_bem") == ("0+unknown", None)


def test_build_artifacts_do_not_change_python_source_hash(tmp_path):
    (tmp_path / "one.py").write_text("x = 1\n")
    first = source_identity._source_sha256(tmp_path)
    build = tmp_path / ".build"
    build.mkdir()
    (build / "generated.py").write_text("not an imported package file\n")
    assert source_identity._source_sha256(tmp_path) == first


def test_unreadable_source_never_yields_a_partial_digest(monkeypatch, tmp_path):
    (tmp_path / "one.py").write_text("x = 1\n")
    monkeypatch.setattr(source_identity.Path, "read_bytes", lambda _: (_ for _ in ()).throw(OSError()))
    assert source_identity._source_sha256(tmp_path) is None
