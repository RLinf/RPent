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


@pytest.fixture(autouse=True)
def _no_ambient_environment(monkeypatch):
    # An environment created by `rpent-install` records its components in the
    # venv that runs the suite; no test may inherit that.
    monkeypatch.setattr(components, "current_venv", lambda: None)


def _fingerprint(monkeypatch, fingerprint):
    monkeypatch.setattr(components, "current_venv", lambda: Path("/venv"))
    monkeypatch.setattr(components, "read_fingerprint", lambda venv: fingerprint)


def test_require_returns_an_importable_module():
    assert components.require("robot-alpha", "json") is json


def test_a_component_outside_the_recorded_environment_is_reported(monkeypatch):
    # The installer editable-installs the checkout, so a component's module
    # imports in every environment it builds. The recorded component set is
    # what says whether the environment holds it.
    _fingerprint(
        monkeypatch,
        {"targets": ["infra-minimal"], "components": {"infra-minimal": {}}},
    )

    with pytest.raises(components.MissingComponentError) as error:
        components.require("tool-flywheel", "json")

    message = str(error.value)
    assert "component 'tool-flywheel' is not installed" in message
    assert (
        "`rpent-install install infra-minimal tool-flywheel --venv /venv --force`"
        in message
    )


def test_an_environment_without_a_fingerprint_trusts_the_import(monkeypatch):
    monkeypatch.setattr(components, "current_venv", lambda: None)

    assert components.require("tool-flywheel", "json") is json


def test_missing_module_names_component_and_install(monkeypatch):
    monkeypatch.setattr(components, "current_venv", lambda: None)

    with pytest.raises(components.MissingComponentError) as error:
        components.require("model-sam3", MISSING)

    message = str(error.value)
    assert f"component 'model-sam3' requires '{MISSING}'" in message
    assert "./rpent-install.sh install model-sam3" in message


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

    # The repair has to name this environment, or it would build the default one.
    assert "rerun `rpent-install install model-sam3 robot-libero --venv /venv`" in str(
        error.value
    )


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
    # Adding a component changes the environment's identity, so the rebuild
    # needs `--force`; without it `install` rejects the command.
    assert (
        "`rpent-install install robot-libero model-molmo --venv /venv --force`"
        in message
    )


def test_missing_parent_package_names_the_component(monkeypatch):
    monkeypatch.setattr(components, "current_venv", lambda: None)

    with pytest.raises(components.MissingComponentError) as error:
        components.require("model-sam3", f"{MISSING}.backend")

    assert f"requires '{MISSING}.backend'" in str(error.value)


def test_failure_inside_the_module_keeps_its_original_exception(monkeypatch):
    # The component and its package are installed; something it imports is
    # not. Reinstalling the component cannot fix that, so the guard stays out
    # of the way and the original traceback reaches the user.
    def broken_import(module):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(components.importlib, "import_module", broken_import)

    with pytest.raises(ModuleNotFoundError) as error:
        components.require("model-sam3", "components.models.sam3.backend")

    assert not isinstance(error.value, components.MissingComponentError)
    assert error.value.name == "torch"


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
