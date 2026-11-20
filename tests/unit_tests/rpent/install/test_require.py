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

"""Optional dependency guards name the owning component and installation hint."""

import importlib
import json
from pathlib import Path

import pytest

# ``rpent.install.require`` is also re-exported as a function from the package,
# so bind the module explicitly to monkeypatch its vendored helpers.
components = importlib.import_module("rpent.install.require")

MISSING = "definitely_missing_rpent_module"


def _fingerprint(monkeypatch, fingerprint):
    monkeypatch.setattr(components, "current_venv", lambda: Path("/venv"))
    monkeypatch.setattr(components, "read_fingerprint", lambda venv: fingerprint)


def test_require_returns_an_importable_module():
    assert components.require("robot-alpha", "json") is json


def test_missing_module_names_component_and_install(monkeypatch):
    monkeypatch.setattr(components, "current_venv", lambda: None)

    with pytest.raises(components.MissingComponentError) as error:
        components.require("model-sam3", MISSING)

    message = str(error.value)
    assert f"component 'model-sam3' requires '{MISSING}'" in message
    assert "rpent-install install model-sam3" in message


def test_component_inside_the_environment_reruns_the_same_targets(monkeypatch):
    _fingerprint(
        monkeypatch,
        {
            "targets": ["model-sam3", "robot-libero"],
            "components": {
                "model-sam3": {"manifest_sha256": "a"},
                "robot-libero": {"manifest_sha256": "b"},
                "sim-libero": {"manifest_sha256": "c"},
            },
        },
    )

    with pytest.raises(components.MissingComponentError) as error:
        components.require("sim-libero", MISSING)

    assert "rerun `rpent-install install model-sam3 robot-libero`" in str(error.value)


def test_component_outside_the_environment_gets_its_own_install(monkeypatch):
    _fingerprint(
        monkeypatch,
        {
            "targets": ["robot-libero"],
            "components": {
                "robot-libero": {"manifest_sha256": "a"},
                "sim-libero": {"manifest_sha256": "b"},
            },
        },
    )

    with pytest.raises(components.MissingComponentError) as error:
        components.require("model-molmo", MISSING)

    message = str(error.value)
    assert "installed for robot-libero" in message
    assert "`rpent-install install model-molmo`" in message
    assert "`rpent-install install robot-libero model-molmo`" in message


def test_fingerprint_without_targets_names_no_empty_command(monkeypatch):
    # `doctor` rejects such a fingerprint, but the hint must not print an empty
    # `rpent-install install` command while it is still in use.
    _fingerprint(monkeypatch, {"components": {"model-sam3": {}}})

    with pytest.raises(components.MissingComponentError) as error:
        components.require("model-sam3", MISSING)

    message = str(error.value)
    assert "installed for no recorded components" in message
    assert "install `" not in message
    assert "None" not in message
