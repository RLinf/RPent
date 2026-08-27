"""Tests for RPent path-resolution and environment helpers."""

from __future__ import annotations

import sys
from pathlib import Path

from rpent.utils.config import bootstrap_rlinf_import, get_repo_root


def test_bootstrap_rlinf_import_prefers_env_checkout(monkeypatch, tmp_path: Path):
    checkout = tmp_path / "rlinf"
    checkout.mkdir()
    (checkout / "__init__.py").touch()
    monkeypatch.setenv("RLINF_REPO_PATH", str(checkout))

    root = bootstrap_rlinf_import()

    assert root == checkout.resolve()
    assert str(root) in sys.path


def test_bootstrap_rlinf_import_falls_back_to_sibling_checkout(monkeypatch):
    monkeypatch.delenv("RLINF_REPO_PATH", raising=False)
    expected = (get_repo_root().parent / "rlinf").resolve()

    root = bootstrap_rlinf_import()

    assert root == expected
    assert str(root) in sys.path
