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

"""Create a component environment, install it, and record its fingerprint."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rpent.install.discovery import Catalog, discover
from rpent.install.errors import InstallError
from rpent.install.fingerprint import (
    FINGERPRINT_NAME,
    FINGERPRINT_SCHEMA_FIELD,
    freeze_sha256,
    read_fingerprint,
    runtime_platform,
    source_state,
    write_fingerprint,
    write_freeze,
)
from rpent.install.manifest import Component
from rpent.install.requirements import (
    FINGERPRINT_SCHEMA,
    read_core_dependencies,
    read_planner_dependencies,
    render_requirements,
    requirements_sha256,
)
from rpent.install.resolve import (
    Resolution,
    choose_python,
    python_errors,
    resolve,
)
from rpent.utils.config import get_repo_root

#: Per-checkout state: the virtualenvs an install created.
STATE_DIR = ".rpent"
#: The rendered requirements stay inside the environment they describe, next to
#: the fingerprint. A copy under the checkout would outlive the environment and
#: duplicate state that belongs to it.
REQUIREMENTS_NAME = "rpent-install.requirements.txt"
#: The rendered requirements open with the components the environment holds.
#: An install that failed before recording a fingerprint still identifies its
#: environment that way; the rendered lines alone change with a manifest's
#: packages, which describe the same environment.
_IDENTITY_PREFIX = "# rpent-install components: "


def environment_id(targets: Iterable[str]) -> str:
    """Return the file-name-safe identifier of a target set."""
    return "+".join(sorted(set(targets)))


def default_venv(root: Path, targets: Iterable[str]) -> Path:
    """Return the default virtualenv path for a target set."""
    return root / STATE_DIR / "venvs" / environment_id(targets)


def requirements_file(venv: Path) -> Path:
    """Return the rendered requirements path inside ``venv``."""
    return Path(venv) / REQUIREMENTS_NAME


def venv_python(venv: Path) -> Path:
    """Return the interpreter path inside a virtualenv."""
    candidate = Path(venv) / "bin" / "python"
    if candidate.exists():
        return candidate
    return Path(venv) / "Scripts" / "python.exe"


def find_uv() -> str:
    """Return the uv executable or raise an actionable error."""
    override = os.environ.get("RPENT_UV")
    if override:
        return override
    uv = shutil.which("uv")
    if uv is None:
        raise InstallError(
            "uv is required to install RPent environments: "
            "https://docs.astral.sh/uv/getting-started/installation/"
        )
    return uv


#: A command that neither finishes nor fails leaves the installer waiting
#: forever. The limit is generous because a first GPU install downloads several
#: gigabytes, and it exists to turn a hung network call into an error.
DEFAULT_COMMAND_TIMEOUT = 3600.0


def run_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    capture: bool = False,
    cwd: Path | None = None,
    timeout: float = DEFAULT_COMMAND_TIMEOUT,
) -> str:
    """Run ``command`` and return its captured stdout.

    ``cwd`` is passed to every ``uv`` call that resolves or installs packages:
    uv discovers ``[tool.uv]`` from its working directory, so resolution must
    happen in the checkout rather than in the caller's directory.
    """
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=capture,
            env=env,
            cwd=cwd,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise InstallError(f"command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise InstallError(
            f"command timed out after {timeout:g}s: {' '.join(command)}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        message = f"command failed ({exc.returncode}): {' '.join(command)}"
        if detail:
            message += f"\n{detail}"
        raise InstallError(message) from exc
    return result.stdout if capture else ""


def resolve_targets(
    targets: Iterable[str],
    catalog: Catalog,
    *,
    root: Path,
) -> Resolution:
    """Resolve component names against the checkout's manifests."""
    return resolve(
        targets,
        catalog,
        core_dependencies=read_core_dependencies(root),
        planner_dependencies=read_planner_dependencies(root),
    )


#: Distributions a virtualenv holds before anything is installed into it.
_BASE_DISTRIBUTIONS = frozenset(
    {"distutils", "pip", "pkg-resources", "setuptools", "wheel"}
)


def _installed_distributions(python: Path) -> list[str] | None:
    """Return the distributions in ``python``'s environment, or ``None``.

    ``None`` means the interpreter could not be asked, which is itself a
    broken environment; the caller reports that rather than an empty list.
    """
    source = (
        "import importlib.metadata\n"
        "print('\\n'.join(sorted({"
        "distribution.metadata['Name'] or '' "
        "for distribution in importlib.metadata.distributions()"
        "})))\n"
    )
    try:
        output = run_command([str(python), "-I", "-c", source], capture=True)
    except (InstallError, OSError):
        return None
    return [line for line in output.splitlines() if line]


def _rendered_identity(rendered: Path) -> tuple[str, ...] | None:
    """Return the component closure a rendered requirements file names."""
    try:
        first = rendered.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    if not first.startswith(_IDENTITY_PREFIX):
        return None
    return tuple(first[len(_IDENTITY_PREFIX) :].split())


def _reject_foreign_venv(destination: Path, resolution: Resolution) -> None:
    """Reject a virtualenv that holds packages but no installation fingerprint.

    Such an environment was not created by ``rpent-install``, so its packages
    belong to whoever installed them; adding this environment's packages would
    mix two dependency sets in one interpreter. Two exceptions are ours: a
    ``uv venv`` nothing was installed into yet is empty, and an environment
    whose rendered requirements name the same component closure is one whose
    install failed after writing them, so rerunning the command repairs it even
    if a manifest's packages changed. Requirements naming another closure
    belong to an install of that other environment.
    """
    rendered = requirements_file(destination)
    if rendered.is_file():
        if _rendered_identity(rendered) == closure(resolution.components):
            return
        raise InstallError(
            f"{destination} holds the requirements of a different environment "
            f"and no {FINGERPRINT_NAME}: an install that failed after writing "
            "them. Install this environment under a different `--venv`, or pass "
            "`--force` to replace it"
        )
    python = venv_python(destination)
    if not python.exists():
        return
    packages = _installed_distributions(python)
    if packages is None:
        raise InstallError(
            f"{destination} has a Python interpreter that cannot be run; "
            "remove it and install the environment again"
        )
    foreign = sorted(
        name
        for name in packages
        if name.lower().replace("_", "-") not in _BASE_DISTRIBUTIONS
    )
    if not foreign:
        return
    shown = ", ".join(foreign[:5])
    if len(foreign) > 5:
        shown += f", and {len(foreign) - 5} more"
    raise InstallError(
        f"{destination} already contains packages ({shown}) but has no "
        f"{FINGERPRINT_NAME}, so rpent-install did not create it; install this "
        "environment under a different `--venv`, remove it first, or pass "
        "`--force` to replace it"
    )


def closure(components: dict[str, Component]) -> tuple[str, ...]:
    """Return an environment's identity: the components in its closure.

    The closure decides what the environment contains, so it decides whether
    two command lines describe the same environment. The names typed on the
    command line name its directory and record how it was created. A manifest
    that changed its packages describes the same environment, so rerunning the
    command rebuilds it from what the manifests now declare. A manifest that
    changed its ``requires`` describes another environment.
    """
    return tuple(sorted(components))


def recorded_closure(fingerprint: dict[str, Any]) -> tuple[str, ...] | None:
    """Return the identity a fingerprint records, or ``None`` without one."""
    recorded = fingerprint.get("components")
    if not isinstance(recorded, dict) or not recorded:
        return None
    names = tuple(sorted(recorded))
    return names if all(isinstance(name, str) for name in names) else None


def _replace(destination: Path) -> Path | None:
    """Move an environment aside so it can be rebuilt.

    ``--force`` asks to rebuild the environment at this path, and changed
    installation inputs make a rebuild necessary; neither means "delete whatever
    is here": only a virtualenv is replaced. It is renamed rather than deleted
    because a virtualenv's scripts embed the path it was created at, so the
    rebuild has to create the new environment at that path, and a rebuild that
    fails restores the old one instead of leaving none.
    """
    if not destination.exists():
        return None
    if not (destination / "pyvenv.cfg").is_file():
        raise InstallError(f"{destination} is not a virtualenv; refusing to replace it")
    backup = destination.with_name(f".{destination.name}.replaced-{os.getpid()}")
    if backup.exists():
        raise InstallError(
            f"{backup} is left from an interrupted --force rebuild; remove it and rerun"
        )
    destination.rename(backup)
    return backup


def _restore_replaced(destination: Path, backup: Path | None) -> None:
    """Put a replaced environment back after its rebuild failed."""
    if backup is None:
        return
    shutil.rmtree(destination, ignore_errors=True)
    backup.rename(destination)


def _discard_replaced(backup: Path | None) -> None:
    """Remove the environment a completed rebuild replaced."""
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _check_existing(
    destination: Path,
    resolution: Resolution,
    *,
    force: bool,
    expected_inputs: str,
) -> Path | None:
    """Reject a virtualenv that holds a different environment.

    Two command lines that name different components can still describe the
    same ``requires`` closure, so the environment is identified by the
    components it contains rather than by the names that were typed.

    A matching closure whose recorded installation inputs changed is rebuilt:
    installing the new requirements over the old interpreter would keep the
    packages an earlier manifest added.

    Returns the environment moved aside for a rebuild, or ``None`` when the
    environment can be reused.
    """
    if force:
        return _replace(destination)
    existing = read_fingerprint(destination)
    if existing is None:
        _reject_foreign_venv(destination, resolution)
        return None
    recorded = recorded_closure(existing)
    if recorded is None:
        # A fingerprint without a component table cannot identify what the
        # environment holds; installing over it would mix unknown packages
        # with the requested closure and record only the latter.
        raise InstallError(
            f"{destination} has a fingerprint without a component table, so the "
            "components it holds are unknown; pass `--force` to rebuild it"
        )
    wanted = closure(resolution.components)
    if recorded != wanted:
        added = sorted(set(wanted) - set(recorded))
        dropped = sorted(set(recorded) - set(wanted))
        details = [f"  adds {name}" for name in added]
        details += [f"  drops {name}" for name in dropped]
        raise InstallError(
            f"{destination} already holds a different environment:\n"
            + "\n".join(details)
            + "\ninstall this set under a different `--venv`, or pass `--force` to "
            "replace it"
        )
    if existing.get("requirements_sha256") != expected_inputs:
        # The components are the same but their packages, overrides, or index
        # settings are not. A rebuild is the only way to drop what an earlier
        # manifest installed.
        return _replace(destination)
    return None


def install(
    targets: Iterable[str],
    *,
    root: Path | None = None,
    venv: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Install ``targets`` and their closure into one environment.

    ``force`` replaces the environment at ``destination`` instead of rejecting
    it when it holds different components or was not created by
    ``rpent-install``. Without ``force``, an environment with the requested
    closure is reused when its recorded installation inputs still match and
    rebuilt when they changed.

    Returns the written fingerprint.
    """
    repo_root = (root or get_repo_root()).resolve()
    catalog = discover(repo_root)
    resolution = resolve_targets(targets, catalog, root=repo_root)
    if not resolution.ok:
        raise InstallError("\n".join(resolution.errors))
    core = read_core_dependencies(repo_root)
    planner = read_planner_dependencies(repo_root)
    requirements = render_requirements(
        resolution, core_dependencies=core, planner_dependencies=planner
    )
    destination = (venv or default_venv(repo_root, resolution.targets)).resolve()

    if dry_run:
        print(requirements, end="")
        return {
            "targets": list(resolution.targets),
            "venv": str(destination),
            "requirements": requirements,
        }

    expected_inputs = requirements_sha256(
        resolution,
        root=repo_root,
        core_dependencies=core,
        planner_dependencies=planner,
    )

    # Find uv before ``--force`` replaces anything: an environment is only
    # replaced when the rebuild can start.
    uv = find_uv()
    replaced = _check_existing(
        destination,
        resolution,
        force=force,
        expected_inputs=expected_inputs,
    )
    if verbose and replaced is not None and not force:
        print(f"installation inputs changed; rebuilding {destination}")
    try:
        if venv_python(destination).exists():
            interpreter = _python_version(venv_python(destination))
        else:
            interpreter = choose_python(resolution.components.values())
        unsupported = python_errors(
            interpreter, resolution.components.values(), owner=str(destination)
        )
        if unsupported:
            raise InstallError(
                "\n".join(unsupported)
                + "\nno interpreter supports every component; install a Python "
                "the closure supports and rerun with `--force` to rebuild "
                f"{destination}"
            )

        python = venv_python(destination)
        if not python.exists():
            if verbose:
                print(f"$ {uv} venv --python {interpreter} {destination}")
            run_command(
                [uv, "venv", "--python", interpreter, str(destination)],
                cwd=repo_root,
            )
            python = venv_python(destination)
        if not python.exists():
            raise InstallError(
                f"{destination} exists but has no Python interpreter; remove it and "
                f"rerun `{repair_command(resolution)}`"
            )

        requirements_path = requirements_file(destination)
        # Rendered requirements without a fingerprint mean the previous install
        # stopped after writing package metadata. ``uv`` treats that metadata as
        # an installed package even when its files are gone, so the repair
        # reinstalls every requirement instead of accepting what the failed run
        # left behind.
        repairing = (
            requirements_path.is_file() and read_fingerprint(destination) is None
        )
        requirements_path.write_text(
            f"{_IDENTITY_PREFIX}{' '.join(closure(resolution.components))}\n"
            + requirements,
            encoding="utf-8",
        )

        install_command = [uv, "pip", "install", "--python", str(python)]
        install_command.extend(index_options(resolution))
        for component in resolution.components.values():
            for override in component.overrides:
                install_command.extend(["--override", str(override)])
        if repairing:
            install_command.append("--reinstall")
        install_command.extend(["-r", str(requirements_path)])
        if verbose:
            print(f"$ {' '.join(install_command)}")
        # uv discovers ``[tool.uv]`` by walking up from its working directory, so
        # run it in the checkout: the recorded inputs live there, and an
        # environment installed from another directory must not resolve with
        # other settings.
        run_command(install_command, cwd=repo_root)

        editable = [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "-e",
            str(repo_root),
        ]
        if verbose:
            print(f"$ {' '.join(editable)}")
        run_command(editable, cwd=repo_root)

        _probe_components(resolution, python, verbose=verbose)
        run_command([uv, "pip", "check", "--python", str(python)], cwd=repo_root)

        frozen = run_command(
            [uv, "pip", "freeze", "--python", str(python)], capture=True, cwd=repo_root
        )
        write_freeze(destination, frozen)
        commit, dirty = source_state(repo_root)
        fingerprint = {
            FINGERPRINT_SCHEMA_FIELD: FINGERPRINT_SCHEMA,
            "targets": list(resolution.targets),
            "python": _python_version(python),
            "platform": runtime_platform(),
            "source_commit": commit,
            "source_dirty": dirty,
            "components": {
                name: {"manifest_sha256": component.manifest_sha256}
                for name, component in resolution.components.items()
            },
            "requirements_sha256": expected_inputs,
            "freeze_sha256": freeze_sha256(frozen),
            "installed_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        written = write_fingerprint(destination, fingerprint)
    except BaseException:
        # The rebuild did not finish, so the environment it replaced is the
        # only working one at this path; put it back.
        try:
            _restore_replaced(destination, replaced)
        except OSError as exc:
            print(f"could not restore {destination}: {exc}", file=sys.stderr)
        raise
    _discard_replaced(replaced)
    if verbose:
        print(f"installation fingerprint: {written}")
    return fingerprint


def index_options(resolution: Resolution) -> list[str]:
    """Return the uv index options the environment resolves with."""
    options: list[str] = []
    if resolution.index_url:
        options.extend(["--index-url", resolution.index_url])
    for url in resolution.extra_index_urls:
        options.extend(["--extra-index-url", url])
    if resolution.index_strategy:
        options.extend(["--index-strategy", resolution.index_strategy])
    return options


def _python_version(python: Path) -> str:
    try:
        return run_command(
            [str(python), "-c", "import platform; print(platform.python_version())"],
            capture=True,
        ).strip()
    except InstallError:
        return "unknown"


#: Role modules a component of each kind may import, relative to its ``probe``
#: package. Importing the package alone is a weak check: a slice's entry point
#: stays lazy so a missing optional dependency does not break discovery, and a
#: package that contains nothing would pass. A probe outside the checkout names
#: the upstream distribution the component installs and has no roles; ``infra``
#: components are always of that kind. A tool is either a process with a
#: ``cli`` or a capability library that a robot slice registers, so only the
#: roles the component ships are probed.
ROLE_MODULES = {
    "robot": ("robot_spec", "env_client"),
    "sim": ("env_server",),
    "model": ("server",),
    "tool": ("cli",),
}


#: The first segment of a probe that names code in this checkout. Any other
#: probe names an upstream distribution the component installs, and that
#: distribution does not own the component's role modules.
CHECKOUT_PROBES = frozenset({"robots", "components"})


def _ships(source: Path, role: str) -> bool:
    """Return whether the component's directory holds ``role``'s module."""
    return (source / f"{role}.py").is_file() or (source / role).is_dir()


def probe_modules(
    probe: str, kind: str, *, source: Path | None = None
) -> tuple[str, ...]:
    """Return the modules an install probe imports, in order.

    ``source`` is the component's directory; with it, only the roles the
    component ships are imported, because a tool exposes either a CLI or a
    library of tool modules and no module exists that every tool owns.
    """
    if probe.split(".")[0] not in CHECKOUT_PROBES:
        return (probe,)
    roles = ROLE_MODULES.get(kind, ())
    if source is not None:
        roles = tuple(role for role in roles if _ships(source, role))
    return (probe, *(f"{probe}.{role}" for role in roles))


def probe_source(probe: str, kind: str, *, source: Path | None = None) -> str:
    """Return the source an install probe runs for a component of ``kind``.

    The probe imports the component's package and the role modules its kind
    owns, so an environment that installed the package without the component's
    own code fails here rather than the first time the component is used.
    """
    modules = probe_modules(probe, kind, source=source)
    lines = [f"import {modules[0]}"]
    if len(modules) > 1:
        lines.insert(0, "import importlib")
        lines.extend(f'importlib.import_module("{module}")' for module in modules[1:])
    return "\n".join(lines) + "\n"


def probe_command(
    python: Path, probe: str, kind: str, *, source: Path | None = None
) -> list[str]:
    """Return the command that runs an install probe inside ``python``.

    ``-I`` keeps the caller's environment and working directory out of
    ``sys.path``, so a probe reports what the environment installed: the
    editable install of the checkout, not the source tree the command happens
    to be running in.
    """
    return [str(python), "-I", "-c", probe_source(probe, kind, source=source)]


def repair_command(resolution: Resolution, *, force: bool = False) -> str:
    """Return the command that rebuilds the same installation.

    ``force`` replaces the environment at the target path. ``install`` requires
    it when the recorded component closure differs; ``doctor`` also asks for it
    when the recorded platform does.
    """
    command = f"rpent-install install {' '.join(resolution.targets)}"
    return f"{command} --force" if force else command


def _probe_components(
    resolution: Resolution,
    python: Path,
    *,
    verbose: bool,
) -> None:
    for name, component in resolution.components.items():
        probe = component.probe
        if verbose:
            print(f"probe {name}: import {probe}")
        try:
            run_command(
                probe_command(python, probe, component.kind, source=component.path),
                capture=True,
            )
        except InstallError as exc:
            raise InstallError(
                f"component {name!r} is not importable in {python.parent.parent}\n"
                f"  probe: import {probe}\n"
                f"  install: {repair_command(resolution)} "
                f"--venv {python.parent.parent}\n"
                f"{exc}"
            ) from exc
