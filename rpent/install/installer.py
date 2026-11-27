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

"""Create, update, and diagnose component virtual environments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rpent.install.errors import InstallError
from rpent.install.manifest import Catalog, Component, discover
from rpent.install.resolve import (
    Resolution,
    choose_python,
    index_options,
    python_errors,
    read_core_dependencies,
    read_planner_dependencies,
    render_requirements,
    requirements_sha256,
    resolve,
)
from rpent.utils.config import get_repo_root

STATE_DIR = ".rpent"
FINGERPRINT_NAME = "rpent-install.json"
DEFAULT_COMMAND_TIMEOUT = 3600.0
OK = "ok"
WARN = "warn"
FAIL = "fail"

_BASE_DISTRIBUTIONS = frozenset(
    {"distutils", "pip", "pkg-resources", "setuptools", "wheel"}
)
#: Role modules probed for checkout-owned components.
ROLE_MODULES = {
    "robot": ("robot_spec", "env_client"),
    "sim": ("env_server",),
    "model": ("server",),
    "tool": ("cli",),
}
CHECKOUT_PROBES = frozenset({"robots", "components"})


@dataclass(frozen=True)
class Check:
    """One diagnostic result returned by :func:`check_installation`."""

    status: str
    message: str


def environment_id(targets: Iterable[str]) -> str:
    """Return the directory-safe identifier of a target set."""
    return "+".join(sorted(set(targets)))


def default_venv(root: Path, targets: Iterable[str]) -> Path:
    """Return the default virtualenv path for ``targets``."""
    return root / STATE_DIR / "venvs" / environment_id(targets)


def venv_python(venv: Path) -> Path:
    """Return the Python executable inside ``venv``."""
    candidate = Path(venv) / "bin" / "python"
    if candidate.exists():
        return candidate
    return Path(venv) / "Scripts" / "python.exe"


def fingerprint_path(venv: Path) -> Path:
    """Return the fingerprint file path for ``venv``."""
    return Path(venv) / FINGERPRINT_NAME


def read_fingerprint(venv: Path) -> dict[str, Any] | None:
    """Return the venv's fingerprint dict, or ``None`` if absent or unreadable."""
    try:
        data = json.loads(fingerprint_path(venv).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_fingerprint(venv: Path, fingerprint: dict[str, Any]) -> Path:
    """Atomically write ``fingerprint`` into ``venv``."""
    destination = fingerprint_path(venv)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(fingerprint, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def fingerprint_sha256(fingerprint: dict[str, Any]) -> str:
    """Return the canonical SHA-256 hash of ``fingerprint``."""
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_sha256(frozen: str) -> str:
    """Return the order-independent SHA-256 hash of a ``uv pip freeze`` listing."""
    lines = sorted(line.strip() for line in frozen.splitlines() if line.strip())
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def runtime_platform() -> str:
    """Return the current platform tag (e.g. ``linux-x86_64``)."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux":
        return "linux-" + ("aarch64" if machine in {"aarch64", "arm64"} else "x86_64")
    if system == "darwin":
        return "darwin-" + ("arm64" if machine in {"aarch64", "arm64"} else "x86_64")
    return f"{system}-{machine}"


def current_venv() -> Path | None:
    """Return the active virtual environment, or ``None``."""
    if os.environ.get("RPENT_INSTALL_BOOTSTRAP") == "1":
        caller_venv = os.environ.get("RPENT_CALLER_VENV")
        if caller_venv and (Path(caller_venv) / "pyvenv.cfg").is_file():
            return Path(caller_venv)
        return None
    env_venv = os.environ.get("VIRTUAL_ENV")
    if env_venv:
        candidate = Path(env_venv)
        if (candidate / "pyvenv.cfg").is_file():
            return candidate
    prefix = Path(sys.prefix)
    if (prefix / "pyvenv.cfg").is_file():
        return prefix
    return None


def find_uv() -> str:
    """Return the ``uv`` executable path or raise an actionable error."""
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


def run_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    capture: bool = False,
    cwd: Path | None = None,
    timeout: float = DEFAULT_COMMAND_TIMEOUT,
) -> str:
    """Run ``command`` and return stdout if ``capture`` is True."""
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
    python: str | None = None,
) -> Resolution:
    """Resolve ``targets`` against the checkout's manifests and ``pyproject.toml``."""
    return resolve(
        targets,
        catalog,
        python=python,
        core_dependencies=read_core_dependencies(root),
        planner_dependencies=read_planner_dependencies(root),
    )


def closure(components: dict[str, Component]) -> tuple[str, ...]:
    """Return the sorted component names in an environment's closure."""
    return tuple(sorted(components))


def recorded_closure(fingerprint: dict[str, Any]) -> tuple[str, ...] | None:
    """Return the sorted component closure recorded in ``fingerprint``."""
    recorded = fingerprint.get("components")
    if not isinstance(recorded, dict) or not recorded:
        return None
    names = tuple(sorted(recorded))
    return names if all(isinstance(name, str) for name in names) else None


def _python_version(python: Path) -> str:
    try:
        return run_command(
            [str(python), "-c", "import platform; print(platform.python_version())"],
            capture=True,
        ).strip()
    except InstallError:
        return "unknown"


def _installed_distributions(python: Path) -> list[str] | None:
    source = (
        "import importlib.metadata\n"
        "print('\\n'.join(sorted({"
        "d.metadata['Name'] or '' for d in importlib.metadata.distributions()"
        "})))\n"
    )
    try:
        output = run_command([str(python), "-I", "-c", source], capture=True)
    except (InstallError, OSError):
        return None
    return [line for line in output.splitlines() if line]


def _check_not_non_venv_path(destination: Path) -> None:
    """Refuse to overwrite or pollute an existing non-virtualenv path."""
    if not destination.exists():
        return
    if (destination / "pyvenv.cfg").is_file():
        return
    if not destination.is_dir() or any(destination.iterdir()):
        raise InstallError(
            f"{destination} exists and is not an empty directory or virtualenv; "
            "choose a different `--venv` path"
        )


def _replace(destination: Path) -> Path | None:
    """Move an existing virtualenv aside so a failed rebuild can restore it."""
    if not destination.exists():
        return None
    if not (destination / "pyvenv.cfg").is_file():
        if destination.is_dir() and not any(destination.iterdir()):
            return None
        raise InstallError(f"{destination} is not a virtualenv; refusing to replace it")
    backup = destination.with_name(f".{destination.name}.replaced-{os.getpid()}")
    if backup.exists():
        raise InstallError(
            f"{backup} is left from an interrupted --force rebuild; remove it and rerun"
        )
    destination.rename(backup)
    return backup


def _prepare_destination(
    destination: Path, resolution: Resolution, *, force: bool
) -> Path | None:
    """Validate ``destination`` and optionally move it aside for ``--force``."""
    _check_not_non_venv_path(destination)
    if force:
        return _replace(destination)

    python = venv_python(destination)
    if (destination / "pyvenv.cfg").is_file() and not python.exists():
        raise InstallError(
            f"{destination} has pyvenv.cfg but no Python interpreter; "
            "pass `--force` to recreate it"
        )

    existing = read_fingerprint(destination)
    if existing is not None:
        recorded = recorded_closure(existing)
        if recorded is None:
            raise InstallError(
                f"{destination} has an invalid fingerprint; pass `--force` to rebuild it"
            )
        wanted = closure(resolution.components)
        if recorded != wanted:
            added = sorted(set(wanted) - set(recorded))
            dropped = sorted(set(recorded) - set(wanted))
            details = [f"  adds {n}" for n in added] + [f"  drops {n}" for n in dropped]
            raise InstallError(
                f"{destination} already holds a different environment:\n"
                + "\n".join(details)
                + "\nuse a different `--venv`, or pass `--force` to replace it"
            )
        return None

    if not python.exists():
        return None
    packages = _installed_distributions(python)
    if packages is None:
        raise InstallError(
            f"{destination} has a Python interpreter that cannot be run; "
            "remove it and rerun"
        )
    foreign = sorted(
        name
        for name in packages
        if name.lower().replace("_", "-") not in _BASE_DISTRIBUTIONS
    )
    if foreign:
        shown = ", ".join(foreign[:5])
        if len(foreign) > 5:
            shown += f", and {len(foreign) - 5} more"
        raise InstallError(
            f"{destination} already contains packages ({shown}) but has no "
            f"{FINGERPRINT_NAME} (either it was created outside rpent-install or "
            "a previous install was interrupted); choose a different `--venv`, "
            "pass `--force` to replace it, or use `uv pip install -e .` if "
            "adding RPent to an externally managed environment"
        )
    return None


def probe_modules(
    probe: str, kind: str, *, source: Path | None = None
) -> tuple[str, ...]:
    """Return the import paths probed for a component."""
    if probe.split(".")[0] not in CHECKOUT_PROBES:
        return (probe,)
    roles = ROLE_MODULES.get(kind, ())
    if source is not None:
        roles = tuple(
            role
            for role in roles
            if (source / f"{role}.py").is_file() or (source / role).is_dir()
        )
    return (probe, *(f"{probe}.{role}" for role in roles))


def probe_command(
    python: Path, probe: str, kind: str, *, source: Path | None = None
) -> list[str]:
    """Return the isolated Python command that probes a component."""
    modules = probe_modules(probe, kind, source=source)
    lines = [f"import {modules[0]}"]
    if len(modules) > 1:
        lines.insert(0, "import importlib")
        lines.extend(f'importlib.import_module("{m}")' for m in modules[1:])
    return [str(python), "-I", "-c", "\n".join(lines) + "\n"]


def repair_command(resolution: Resolution, *, force: bool = False) -> str:
    """Return the command that reinstalls ``resolution``."""
    command = f"rpent-install install {' '.join(resolution.targets)}"
    return f"{command} --force" if force else command


def install(
    targets: Iterable[str],
    *,
    root: Path | None = None,
    venv: Path | None = None,
    python_version: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    index_url: str | None = None,
    torch_backend: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Install ``targets`` and their closure into one virtualenv."""
    repo_root = (root or get_repo_root()).resolve()
    catalog = discover(repo_root)
    resolution = resolve_targets(
        targets, catalog, root=repo_root, python=python_version
    )
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
    uv = find_uv()
    replaced = _prepare_destination(destination, resolution, force=force)
    cmd_env = dict(os.environ)
    if torch_backend:
        cmd_env["UV_TORCH_BACKEND"] = torch_backend

    created = False
    try:
        python = venv_python(destination)
        existed = python.exists()
        if existed:
            interpreter = _python_version(python)
            if python_version is not None and not (
                interpreter == python_version
                or interpreter.startswith(f"{python_version}.")
            ):
                raise InstallError(
                    f"{destination} uses Python {interpreter}, not requested "
                    f"{python_version}; rerun with `--force` to recreate the "
                    "environment"
                )
        else:
            interpreter = python_version or choose_python(
                resolution.components.values()
            )
        unsupported = python_errors(
            interpreter, resolution.components.values(), owner=str(destination)
        )
        if unsupported:
            suffix = (
                "\nrerun with `--force` to recreate the environment with a "
                "supported Python version"
                if existed
                else ""
            )
            raise InstallError("\n".join(unsupported) + suffix)

        if not existed:
            if verbose:
                print(f"$ {uv} venv --python {interpreter} {destination}")
            created = True
            run_command(
                [uv, "venv", "--python", interpreter, str(destination)],
                env=cmd_env,
                cwd=repo_root,
            )
            python = venv_python(destination)
        if not python.exists():
            raise InstallError(f"{destination} has no Python interpreter")
        if created:
            actual_python = _python_version(python)
            actual_unsupported = python_errors(
                actual_python,
                resolution.components.values(),
                owner=str(destination),
            )
            if actual_unsupported:
                raise InstallError("\n".join(actual_unsupported))

        index_args = index_options(resolution, index_url_override=index_url)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", prefix="rpent-req-", encoding="utf-8", delete=False
        ) as req_file:
            req_file.write(requirements)
            req_path = Path(req_file.name)
        try:
            install_cmd = [uv, "pip", "install", "--python", str(python)]
            install_cmd.extend(index_args)
            for component in resolution.components.values():
                for override in component.overrides:
                    install_cmd.extend(["--override", str(override)])
            install_cmd.extend(["-r", str(req_path)])
            if verbose:
                print(f"$ {' '.join(install_cmd)}")
            run_command(install_cmd, env=cmd_env, cwd=repo_root)
        finally:
            req_path.unlink(missing_ok=True)

        editable_cmd = [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            *index_args,
            "--no-deps",
            "-e",
            str(repo_root),
        ]
        if verbose:
            print(f"$ {' '.join(editable_cmd)}")
        run_command(editable_cmd, env=cmd_env, cwd=repo_root)

        for name, component in resolution.components.items():
            if verbose:
                print(f"probe {name}: import {component.probe}")
            try:
                run_command(
                    probe_command(
                        python, component.probe, component.kind, source=component.path
                    ),
                    capture=True,
                )
            except InstallError as exc:
                raise InstallError(
                    f"component {name!r} is not importable in {destination}\n"
                    f"  probe: import {component.probe}\n{exc}"
                ) from exc

        has_overrides = any(comp.overrides for comp in resolution.components.values())
        if not has_overrides:
            run_command([uv, "pip", "check", "--python", str(python)], cwd=repo_root)
        else:
            checked = _run_diag([uv, "pip", "check", "--python", str(python)])
            if checked.returncode != 0 and verbose:
                detail = (checked.stdout + checked.stderr).strip()
                print(
                    "warning: uv pip check reported override-induced mismatches:\n"
                    f"{detail}",
                    file=sys.stderr,
                )
        frozen = run_command(
            [uv, "pip", "freeze", "--python", str(python)], capture=True, cwd=repo_root
        )
        fingerprint = {
            "targets": list(resolution.targets),
            "python": _python_version(python),
            "platform": runtime_platform(),
            "components": {
                name: {"manifest_sha256": comp.manifest_sha256}
                for name, comp in resolution.components.items()
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
        if replaced is not None:
            try:
                shutil.rmtree(destination, ignore_errors=True)
                replaced.rename(destination)
            except OSError as exc:
                print(f"could not restore {destination}: {exc}", file=sys.stderr)
        elif created:
            shutil.rmtree(destination, ignore_errors=True)
        raise
    if replaced is not None:
        shutil.rmtree(replaced, ignore_errors=True)
    if verbose:
        print(f"installation fingerprint: {written}")
    return fingerprint


def _run_diag(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=DEFAULT_COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise InstallError(
            f"command timed out after {DEFAULT_COMMAND_TIMEOUT:g}s: {' '.join(command)}"
        ) from exc
    except OSError as exc:
        raise InstallError(f"cannot run {command[0]}: {exc}") from exc


def check_installation(venv: Path, *, root: Path | None = None) -> list[Check]:
    """Diagnose ``venv`` against its recorded fingerprint and current manifests."""
    repo_root = (root or get_repo_root()).resolve()
    venv = Path(venv).resolve()
    fingerprint = read_fingerprint(venv)
    if fingerprint is None:
        return [
            Check(
                FAIL,
                f"{venv} has no {FINGERPRINT_NAME}; install it with "
                f"`rpent-install install <components> --venv {venv}`",
            )
        ]
    targets = fingerprint.get("targets")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(target, str) and target for target in targets)
    ):
        return [Check(FAIL, f"fingerprint in {venv} records no target components")]

    checks = [
        Check(OK, f"targets: {', '.join(targets)}"),
        Check(OK, f"installed at: {fingerprint.get('installed_at', '?')}"),
    ]
    try:
        catalog = discover(repo_root)
        resolution = resolve_targets(targets, catalog, root=repo_root)
    except (InstallError, ValueError) as exc:
        checks.append(Check(FAIL, f"cannot resolve {' '.join(targets)!r}: {exc}"))
        return checks
    if not resolution.ok:
        checks.append(Check(FAIL, "\n".join(resolution.errors)))
        return checks

    recorded = recorded_closure(fingerprint)
    expected_inputs = requirements_sha256(
        resolution,
        root=repo_root,
        core_dependencies=read_core_dependencies(repo_root),
        planner_dependencies=read_planner_dependencies(repo_root),
    )
    if recorded is None:
        checks.append(
            Check(
                FAIL,
                f"fingerprint in {venv} has an invalid components record; rerun "
                f"`{repair_command(resolution, force=True)} --venv {venv}`",
            )
        )
    elif recorded != closure(resolution.components):
        checks.append(
            Check(
                FAIL,
                "environment holds a different component set; rerun "
                f"`{repair_command(resolution, force=True)} --venv {venv}`",
            )
        )
    elif fingerprint.get("requirements_sha256") != expected_inputs:
        checks.append(
            Check(
                FAIL,
                "environment is out of date with manifests or overrides; rerun "
                f"`{repair_command(resolution)} --venv {venv}`",
            )
        )
    else:
        checks.append(Check(OK, "installation inputs unchanged"))

    python = venv_python(venv)
    if not python.exists():
        checks.append(Check(FAIL, f"no Python interpreter in {venv}"))
        return checks
    recorded_python = str(fingerprint.get("python", ""))
    unsupported = python_errors(
        recorded_python, resolution.components.values(), owner=str(venv)
    )
    if unsupported:
        checks.append(Check(FAIL, "\n".join(unsupported)))
    else:
        checks.append(Check(OK, f"python: {recorded_python}"))

    current_platform = runtime_platform()
    recorded_platform = fingerprint.get("platform")
    if recorded_platform is not None and recorded_platform != current_platform:
        checks.append(
            Check(
                FAIL,
                f"installed for {recorded_platform}, but this machine is "
                f"{current_platform}; rerun "
                f"`{repair_command(resolution, force=True)} --venv {venv}`",
            )
        )
    else:
        checks.append(Check(OK, f"platform: {current_platform}"))

    try:
        uv = find_uv()
    except InstallError as exc:
        checks.append(Check(WARN, str(exc)))
        uv = None
    if uv is not None:
        frozen = _run_diag([uv, "pip", "freeze", "--python", str(python)])
        if frozen.returncode != 0:
            checks.append(Check(FAIL, f"uv pip freeze failed: {frozen.stderr.strip()}"))
        elif freeze_sha256(frozen.stdout) != fingerprint.get("freeze_sha256"):
            checks.append(
                Check(
                    WARN,
                    "installed packages differ from recorded freeze (e.g. overlaid "
                    f"packages); rerun `{repair_command(resolution, force=True)} "
                    f"--venv {venv}` to reset",
                )
            )
        else:
            checks.append(Check(OK, "installed package set matches the fingerprint"))
        has_overrides = any(comp.overrides for comp in resolution.components.values())
        checked = _run_diag([uv, "pip", "check", "--python", str(python)])
        if checked.returncode != 0:
            detail = (checked.stdout + checked.stderr).strip()
            if has_overrides:
                checks.append(
                    Check(
                        WARN,
                        f"uv pip check reported override-induced mismatches: {detail}",
                    )
                )
            else:
                checks.append(Check(FAIL, f"uv pip check failed: {detail}"))
        else:
            checks.append(Check(OK, "uv pip check"))

    for name, component in resolution.components.items():
        result = _run_diag(
            probe_command(
                python, component.probe, component.kind, source=component.path
            )
        )
        if result.returncode != 0:
            last = (
                result.stderr.strip().splitlines()[-1]
                if result.stderr.strip()
                else component.probe
            )
            checks.append(Check(FAIL, f"component {name!r} is not importable: {last}"))
        else:
            checks.append(Check(OK, f"probe {name}: import {component.probe}"))
    return checks


def inspect_installation(venv: Path) -> dict[str, Any] | None:
    """Return the stored fingerprint with its canonical hash, or ``None``."""
    fingerprint = read_fingerprint(venv)
    if fingerprint is None:
        return None
    return {
        "venv": str(Path(venv).resolve()),
        "sha256": fingerprint_sha256(fingerprint),
        "fingerprint": fingerprint,
    }
