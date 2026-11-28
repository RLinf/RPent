# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ``rpent.install.require``."""

from __future__ import annotations

import importlib
import json

import pytest

from rpent.install import installer
from rpent.install.require import MissingComponentError, require

require_mod = importlib.import_module("rpent.install.require")


def test_require_imports_available_module_even_in_different_fingerprinted_venv(
    tmp_path, monkeypatch
):
    venv = tmp_path / ".venv-unit"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (venv / installer.FINGERPRINT_NAME).write_text(
        json.dumps(
            {"targets": ["infra-unit_cpu"], "components": {"infra-unit_cpu": {}}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(require_mod, "current_venv", lambda: venv)

    mod = require("model-openpi", "json")
    assert mod is json


def test_require_formats_actionable_hint_when_module_is_missing(tmp_path, monkeypatch):
    venv = tmp_path / ".venv-libero"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (venv / installer.FINGERPRINT_NAME).write_text(
        json.dumps({"targets": ["robot-libero"], "components": {"robot-libero": {}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(require_mod, "current_venv", lambda: venv)

    with pytest.raises(MissingComponentError) as exc_info:
        require("model-molmo", "rpent_definitely_missing_pkg_xyz")
    msg = str(exc_info.value)
    assert "model-molmo" in msg
    assert "rpent-install install model-molmo" in msg


def test_require_preserves_internal_missing_dependency_errors(tmp_path, monkeypatch):
    module_file = tmp_path / "broken_mod_xyz.py"
    module_file.write_text("import nonexistent_inner_dep_xyz\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError) as exc_info:
        require("model-demo", "broken_mod_xyz")
    assert not isinstance(exc_info.value, MissingComponentError)
    assert exc_info.value.name == "nonexistent_inner_dep_xyz"


def test_require_repairs_active_alternative_provider_instead_of_adding_conflict(
    tmp_path, monkeypatch
):
    venv = tmp_path / ".venv-robodojo"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (venv / installer.FINGERPRINT_NAME).write_text(
        json.dumps(
            {
                "targets": ["robot-robodojo"],
                "components": {"robot-robodojo": {}, "sim-robodojo": {}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(require_mod, "current_venv", lambda: venv)

    with pytest.raises(MissingComponentError) as exc_info:
        require(
            "infra-rlinf",
            "rpent_definitely_missing_pkg_xyz",
            alternatives=("sim-robodojo",),
        )
    msg = str(exc_info.value)
    assert "sim-robodojo" in msg
    assert "rpent-install install robot-robodojo" in msg
    assert "infra-rlinf --force" not in msg
