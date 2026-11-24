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

"""Report a missing optional dependency as a component installation error.

Components declare their packages in ``component.toml``; importing them
directly at the point of use keeps core imports working without every robot
package installed. This guard replaces local ``try: import ... except
ImportError: ... = None`` fallbacks with one actionable message. The first
argument is the component that owns the package, which is also the name
``rpent-install`` accepts, not the import name::

    torch = require("model-sam3", "torch")
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

from rpent.install.fingerprint import current_venv, read_fingerprint


class MissingComponentError(ImportError):
    """An optional module is missing from the environment that should own it."""


def _current_environment() -> tuple[Path | None, dict[str, Any] | None]:
    """Return the active venv and the fingerprint ``rpent-install`` recorded."""
    venv = current_venv()
    return venv, read_fingerprint(venv) if venv is not None else None


def _install_hint(
    component: str, venv: Path | None, fingerprint: dict[str, Any] | None
) -> str:
    if fingerprint is None:
        return (
            f"install the environment that provides component {component!r} from "
            f"an RPent checkout with `./rpent-install.sh install {component}`, or "
            f"add {component!r} to an environment that already holds the "
            "components it runs with"
        )
    # Every suggestion that rebuilds this environment has to name it: without
    # ``--venv`` the installer targets the default path instead.
    venv_flag = f" --venv {venv}" if venv is not None else ""
    targets = list(fingerprint.get("targets") or [])
    installed = ", ".join(targets) or "no recorded components"
    if targets and component in fingerprint.get("components", {}):
        return (
            f"component {component!r} belongs to this environment, so rerun "
            f"`rpent-install install {' '.join(targets)}{venv_flag}` to repair it"
        )
    combined = " ".join([*targets, component])
    return (
        f"this environment was installed for {installed}; component "
        f"{component!r} is not part of it. Install it in its own environment with "
        f"`rpent-install install {component}`, or rebuild this environment with "
        f"it: `rpent-install install {combined}{venv_flag} --force`"
    )


def require(component: str, module: str) -> ModuleType:
    """Import ``module`` or raise an error naming ``component`` and its install.

    An environment ``rpent-install`` recorded must also list ``component``: the
    installer editable-installs the checkout, so the component's module is
    importable in every environment it builds, including one installed for a
    different component set. An environment without a fingerprint is trusted to
    whatever is importable.

    Only a missing ``module`` or one of its parent packages becomes a
    ``MissingComponentError``. A component whose own imports fail — a package
    its manifest forgot, a broken native library — keeps its original
    exception, so the message does not send the user to reinstall something
    that is already there.

    Raises:
        MissingComponentError: ``component`` is not installed here, or
            ``module`` or a parent package is unimportable.
    """
    venv, fingerprint = _current_environment()
    recorded = fingerprint.get("components") if fingerprint is not None else None
    if isinstance(recorded, dict) and component not in recorded:
        raise MissingComponentError(
            f"component {component!r} is not installed in this environment; "
            f"{_install_hint(component, venv, fingerprint)}"
        )
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name
        if missing is None or not (
            missing == module or module.startswith(f"{missing}.")
        ):
            raise
        raise MissingComponentError(
            f"component {component!r} requires {module!r}, which is not "
            f"importable in this environment; "
            f"{_install_hint(component, venv, fingerprint)}"
        ) from exc
