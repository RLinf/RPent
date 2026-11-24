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

"""Check repository layout and import boundaries.

Rules:

- K1: top-level directories and root files match the repository layout
  whitelist, and every directory under ``components/`` names a kind the
  installer discovers.
- K2: every component directory declares a ``component.toml`` with the
  matching kind (manifest validity is left to ``rpent-install check``) and has
  the role files its kind requires.
- K3: a slice root must not add a retired name, and a migrated slice must not
  keep one. Components that predate the skeleton only warn until their
  migration lands.
- K4: the component layers are ``rpent`` ← ``components/sim`` ←
  ``components/models`` ← ``robots``. A component may import the layers below
  it and never a sibling; a robot slice may import another robot only when its
  manifest declares it in ``imports`` and one of the two ``requires`` the
  other. ``components/tools`` is outside the layers and depends only on
  ``rpent`` and, for a capability library, the model clients' result types;
  it never imports robots, simulators, or another tool. Core ``rpent/**``
  must not import a component or a concrete ``robots.<name>`` statically;
  importing the ``robots`` package itself for discovery is allowed, and the
  backend launcher resolves a component's module from the selected backend.
- K5: core ``rpent/**`` must not name a robot or benchmark (R8).
  ``rpent/components/models/`` is exempt: model-side contracts and the backend table name
  the embodiment they speak to.
- K6: a slice ``robots/<name>/__init__.py`` only re-exports lazily.
- K7: ``spec.py`` > 80, ``adapter.py`` > 400, and ``native_tools.py`` > 100
  lines are reported as warnings until the slice split lands.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tokenize
from dataclasses import dataclass
from datetime import date
from fnmatch import fnmatch
from pathlib import Path

from rpent.install.discovery import discover_components
from rpent.install.errors import InstallConfigError
from rpent.install.manifest import COMPONENT_DIRECTORIES

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Directories the layout registers an owner for. Hidden directories are tool
#: configuration and are exempt.
ALLOWED_TOP_LEVEL = frozenset({"rpent", "robots", "components", "tests", "docs"})
#: Root files the layout allows next to the top-level directories.
ALLOWED_ROOT_FILES = (
    "pyproject.toml",
    "README*",
    "CONTRIBUTING*",
    "LICENSE",
    "AGENTS.md",
    "CLAUDE.md",
    "MANIFEST.in",
    "setup.py",
    ".gitignore",
    ".pre-commit-config.yaml",
    # The installer bootstrap; a checkout has no console script yet.
    "rpent-install.sh",
)
#: Generated, gitignored directories that may appear in a working checkout.
IGNORED_TOP_LEVEL = frozenset({"memory", "__pycache__"})
#: Core subdirectories K5 does not scan, with the reason and the date by which
#: the exception must be removed or renewed. Model-side wire contracts name the
#: embodiment they control. An expired entry is an error, so a temporary
#: exception cannot become permanent by being forgotten.
CORE_NAME_EXEMPTIONS = {
    "rpent/components/models": (
        "model-side wire contracts and the backend table name the embodiment "
        "they speak to",
        date(2027, 6, 30),
    ),
}
#: Migration exceptions (legacy role files and retired slice names) warn until
#: this date and fail afterwards.
MIGRATION_DEADLINE = date(2027, 6, 30)
#: Package groups that must be importable packages even when empty.
GROUP_PACKAGES = (
    "robots",
    "components",
    "components/models",
    "components/sim",
    "components/tools",
)
#: Component directory name under ``components/`` -> the kind it holds, and the
#: order those directories import in: simulators below models, models below
#: robots.
COMPONENT_MODULE_KINDS = {
    "sim": "sim",
    "models": "model",
    "tools": "tool",
}
#: Core modules that dispatch to a tool. The console script has to exist in a
#: core-only wheel, so its entry point names the tool and imports it inside the
#: function; every other core module stays free of tools.
CORE_TOOL_ENTRY_POINTS = frozenset({"rpent/cli/flywheel.py"})
#: Files that carry wire data rather than behavior. They may use the standard
#: library, NumPy and PIL for raster data, pydantic (a core dependency, like
#: NumPy), core contracts, and their own group's modules; anything else belongs
#: in the adapter or server.
PURE_DATA_FILES = frozenset(
    {"control.py", "observation.py", "contracts.py", "platforms.py"}
)
PURE_DATA_ALLOWED_ROOTS = frozenset({"numpy", "PIL", "pydantic", "typing_extensions"})
#: Line-count limits from K7; exceeded counts are warnings until the slice
#: split lands.
LINE_LIMITS = {"spec.py": 80, "adapter.py": 400, "native_tools.py": 100}
#: Names retired from a slice root once its migration lands.
LEGACY_SLICE_NAMES = frozenset(
    {
        "robot_spec.py",
        "prompt_bundle.py",
        "env_client.py",
        "env_server.py",
        "toolkit.py",
        "tools.py",
        "primitives.py",
        "rlinf_env.py",
        "rpent_env.py",
        "runtime_config.py",
        "tasks.py",
        "suites.py",
        "eval",
    }
)
#: Slices that still carry legacy names, recorded from the baseline. Entries
#: disappear as the slice migration completes; adding one again is a
#: regression.
LEGACY_SLICES = {
    "efe": frozenset(
        {
            "env_client.py",
            "prompt_bundle.py",
            "robot_spec.py",
            "toolkit.py",
            "tools.py",
        }
    ),
    "dual_franka": frozenset(
        {
            "env_client.py",
            "env_server.py",
            "prompt_bundle.py",
            "robot_spec.py",
            "rpent_env.py",
            "runtime_config.py",
            "tasks.py",
            "toolkit.py",
            "tools.py",
        }
    ),
    "franka": frozenset(
        {
            "env_client.py",
            "env_server.py",
            "prompt_bundle.py",
            "robot_spec.py",
            "rpent_env.py",
            "runtime_config.py",
            "tasks.py",
            "toolkit.py",
            "tools.py",
        }
    ),
    "libero": frozenset(
        {
            "env_client.py",
            "env_server.py",
            "prompt_bundle.py",
            "robot_spec.py",
            "suites.py",
            "toolkit.py",
            "tools.py",
        }
    ),
    "robocasa": frozenset(
        {
            "env_client.py",
            "env_server.py",
            "eval",
            "primitives.py",
            "prompt_bundle.py",
            "robot_spec.py",
            "toolkit.py",
            "tools.py",
        }
    ),
    "robolab": frozenset(
        {
            "env_client.py",
            "env_server.py",
            "prompt_bundle.py",
            "robot_spec.py",
            "toolkit.py",
        }
    ),
    "robodojo": frozenset(
        {
            "env_client.py",
            "env_server.py",
            "primitives.py",
            "prompt_bundle.py",
            "rlinf_env.py",
            "robot_spec.py",
            "tasks.py",
            "toolkit.py",
            "tools.py",
        }
    ),
    "robotwin": frozenset(
        {
            "env_client.py",
            "env_server.py",
            "primitives.py",
            "prompt_bundle.py",
            "rlinf_env.py",
            "robot_spec.py",
            "toolkit.py",
            "tools.py",
        }
    ),
    "yam": frozenset(
        {
            "env_client.py",
            "env_server.py",
            "primitives.py",
            "prompt_bundle.py",
            "rlinf_env.py",
            "robot_spec.py",
            "runtime_config.py",
            "tasks.py",
            "toolkit.py",
            "tools.py",
        }
    ),
}
#: Slices whose migration is complete; K3 becomes an error for them.
MIGRATED_SLICES: frozenset[str] = frozenset()
#: Required role files per kind.
REQUIRED_ROLE_FILES = {
    "robot": (
        "__init__.py",
        "spec.py",
        "config.py",
        "adapter.py",
        "prompts/__init__.py",
        "prompts/domain.py",
    ),
    # A simulator owns the environment server and the translation its robot
    # slice calls into; role files beyond that follow what it needs.
    "sim": ("__init__.py", "env_server.py"),
    # A model service only needs an importable package and its server; config
    # and README are useful conventions, not skeleton requirements.
    "model": ("__init__.py", "server.py"),
    "tool": ("__init__.py",),
}
#: Components that predate the target skeleton; missing role files warn until
#: their migration lands.
LEGACY_COMPONENTS = frozenset(
    {
        ("robot", "dual_franka"),
        ("robot", "efe"),
        ("robot", "franka"),
        ("robot", "libero"),
        ("robot", "robocasa"),
        ("robot", "robolab"),
        ("robot", "robodojo"),
        ("robot", "robotwin"),
        ("robot", "yam"),
        ("model", "apxinf"),
        ("model", "cosmos3"),
        ("model", "cosmos_policy"),
        ("model", "fast_wam"),
        ("model", "molmo"),
        ("model", "openpi"),
        ("model", "sam3"),
    }
)
ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    """One layout violation."""

    path: Path
    line: int
    message: str
    level: str = ERROR


def _module_parts(path: Path, root: Path) -> tuple[str, ...]:
    """Return the dotted package parts that contain ``path``.

    A package entry point is its own package; any other module belongs to its
    directory. Relative imports are resolved against this package, so leaving
    the module name in would shift every level by one.
    """
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts:
        parts.pop()
    return tuple(parts)


def _relative_target(
    path: Path, root: Path, node: ast.ImportFrom
) -> tuple[str, ...] | None:
    """Resolve ``from .. import x`` to the absolute module it names.

    Relative imports cross the same component boundaries as absolute ones, so
    K4 resolves them instead of skipping every ``node.level > 0``. ``from ..
    import franka`` names the ``franka`` package through a bare alias, which
    the module path alone does not show; such an alias is expanded when it
    names a package or module under the resolved base.
    """
    package = _module_parts(path, root)
    # Each extra dot climbs one package; the last one names the top level.
    if node.level - 1 > len(package):
        return None
    base = package[: len(package) - (node.level - 1)]
    if node.module:
        base = (*base, *node.module.split("."))
    return base or None


def _named_imports(
    root: Path, base: tuple[str, ...], node: ast.ImportFrom
) -> list[tuple[int, str]]:
    """Return ``from base import name`` imports whose name is a module here.

    A bare alias names a package or module without spelling out its dotted
    path, so K4 expands it: ``from robots import beta`` crosses into the
    ``beta`` slice exactly like ``from robots.beta import spec`` does.
    """
    imports: list[tuple[int, str]] = []
    for alias in node.names:
        if alias.name == "*":
            continue
        candidate = root.joinpath(*base, alias.name)
        if candidate.is_dir() or candidate.with_suffix(".py").is_file():
            imports.append((node.lineno, ".".join((*base, alias.name))))
    return imports


def _imports(path: Path, root: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if not node.module:
                    continue
                imports.append((node.lineno, node.module))
                imports.extend(
                    _named_imports(root, tuple(node.module.split(".")), node)
                )
                continue
            base = _relative_target(path, root, node)
            if base is None:
                continue
            imports.append((node.lineno, ".".join(base)))
            imports.extend(_named_imports(root, base, node))
        elif isinstance(node, ast.Call):
            target = _literal_import_target(node)
            if target is not None:
                imports.append((node.lineno, target))
    return imports


def _literal_import_target(node: ast.Call) -> str | None:
    """Return the module named by ``import_module("...")`` and friends."""
    function = node.func
    if isinstance(function, ast.Name):
        name = function.id
    elif isinstance(function, ast.Attribute):
        name = function.attr
    else:
        return None
    if name not in {"import_module", "__import__"}:
        return None
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return None
    value = node.args[0].value
    return value if isinstance(value, str) else None


def _concrete_robot(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == "robots":
        return parts[1]
    return None


def _sibling_package(module: str, package: str) -> str | None:
    """Return ``<package>.<name>``'s name for a sibling import."""
    parts = module.split(".")
    package_parts = package.split(".")
    if len(parts) > len(package_parts) and parts[: len(package_parts)] == package_parts:
        return parts[len(package_parts)]
    return None


def _under(module: str, package: str) -> bool:
    """Return whether ``module`` is ``package`` or a submodule of it."""
    return module == package or module.startswith(f"{package}.")


def _owner(path: Path, root: Path) -> tuple[str, str] | None:
    relative = path.relative_to(root).parts
    if not relative:
        return None
    if relative[0] == "robots" and len(relative) > 1:
        return "robot", relative[1]
    if relative[0] == "components" and len(relative) > 2:
        kind = COMPONENT_MODULE_KINDS.get(relative[1])
        if kind is not None:
            return kind, relative[2]
    if relative[0] == "rpent":
        return "core", ""
    return None


def _component_directory(root: Path, owner: tuple[str, str]) -> Path:
    """Return the directory that holds ``owner``'s component."""
    kind, name = owner
    if kind == "robot":
        return root / "robots" / name
    group = next(
        (
            group
            for group, group_kind in COMPONENT_MODULE_KINDS.items()
            if group_kind == kind
        ),
        kind,
    )
    return root / "components" / group / name


def _component_target(module: str) -> tuple[str, str] | None:
    """Return ``(kind, name)`` when ``module`` lives inside a component."""
    parts = module.split(".")
    if len(parts) >= 3 and parts[0] == "components":
        kind = COMPONENT_MODULE_KINDS.get(parts[1])
        if kind is not None:
            return kind, parts[2]
    robot = _concrete_robot(module)
    if robot is not None:
        return "robot", robot
    return None


def _declared_imports(root: Path) -> dict[str, set[str]]:
    """Return robot directory -> directory names its manifest lets it import."""
    try:
        components = discover_components(root)
    except InstallConfigError:
        return {}
    return {
        name.removeprefix("robot-"): {
            other.removeprefix("robot-") for other in component.imports
        }
        for name, component in components.items()
        if component.kind == "robot"
    }


def check_top_level(root: Path) -> list[Finding]:
    """Return K1 violations."""
    findings: list[Finding] = []
    for entry in sorted(root.iterdir()):
        name = entry.name
        if name.startswith(".") or name in IGNORED_TOP_LEVEL:
            continue
        if name.endswith(".egg-info"):
            continue
        if entry.is_dir():
            if name not in ALLOWED_TOP_LEVEL:
                findings.append(
                    Finding(
                        entry,
                        0,
                        f"top-level directory {name!r} has no owner; add it "
                        "to the layout whitelist before creating it",
                    )
                )
        elif not any(fnmatch(name, pattern) for pattern in ALLOWED_ROOT_FILES):
            findings.append(
                Finding(
                    entry,
                    0,
                    f"root file {name!r} is not whitelisted; move it into an "
                    "owned directory or extend the layout whitelist",
                )
            )
    kinds = sorted(
        Path(directory).name
        for directory in COMPONENT_DIRECTORIES.values()
        if directory.startswith("components/")
    )
    components = root / "components"
    if components.is_dir():
        for kind_dir in sorted(components.iterdir()):
            if not kind_dir.is_dir() or kind_dir.name.startswith((".", "_")):
                continue
            if kind_dir.name not in kinds:
                findings.append(
                    Finding(
                        kind_dir,
                        0,
                        f"components/{kind_dir.name} is not a component kind; "
                        f"the installer only discovers {', '.join(kinds)}",
                    )
                )
    return findings


def check_imports(root: Path) -> list[Finding]:
    """Return K4 import boundary violations under ``root``."""
    findings: list[Finding] = []
    declared_imports = _declared_imports(root)
    for path in sorted(root.rglob("*.py")):
        if path.name.startswith("."):
            # AppleDouble metadata for a synced checkout, not source.
            continue
        # The skip names directories of the checkout, not of the absolute path:
        # a checkout that itself lives under a directory named `tests` or
        # `docs` must still be checked.
        inside = path.relative_to(root)
        if any(part in {".git", ".rpent", "tests", "docs"} for part in inside.parts):
            continue
        relative = inside.as_posix()
        owner = _owner(path, root)
        if owner is None:
            continue
        kind, name = owner
        for line, module in _imports(path, root):
            target = _component_target(module)
            if kind == "core":
                if _under(module, "components.models"):
                    findings.append(
                        Finding(
                            path,
                            line,
                            "core must not import model services; keep them "
                            "behind the rpent.components.models client contract",
                        )
                    )
                elif _under(module, "components.sim"):
                    findings.append(
                        Finding(
                            path,
                            line,
                            "core must not import simulator components; keep "
                            "them behind the rpent.components.env contract",
                        )
                    )
                elif _under(module, "components.tools"):
                    if relative not in CORE_TOOL_ENTRY_POINTS:
                        findings.append(
                            Finding(
                                path,
                                line,
                                "core must not import tools; a tool depends on "
                                "the core, not the other way round",
                            )
                        )
                elif target is not None:
                    findings.append(
                        Finding(
                            path,
                            line,
                            f"core must not import concrete robot {target[1]!r}",
                        )
                    )
                continue
            if target is None:
                continue
            target_kind, target_name = target
            if target_name == name:
                continue
            if kind == "robot":
                if target_kind == "robot" and target_name not in declared_imports.get(
                    name, set()
                ):
                    findings.append(
                        Finding(
                            path,
                            line,
                            f"robot {name!r} must not import robot {target_name!r}; "
                            "declare it in component.toml imports when the "
                            "integration intentionally extends another slice",
                        )
                    )
            elif kind == "model":
                if target_kind == "robot":
                    findings.append(
                        Finding(path, line, f"model {name!r} must not import robots")
                    )
                elif target_kind == "model":
                    findings.append(
                        Finding(
                            path,
                            line,
                            f"model {name!r} must not import sibling "
                            f"{target_name!r}; declare a shared contract instead",
                        )
                    )
                elif target_kind == "tool":
                    findings.append(
                        Finding(
                            path,
                            line,
                            f"model {name!r} must not import tool {target_name!r}; "
                            "tools depend only on rpent",
                        )
                    )
            elif kind == "tool":
                if target_kind != "model":
                    findings.append(
                        Finding(
                            path,
                            line,
                            f"tool {name!r} must not import {target_kind} "
                            f"{target_name!r}; a tool reads core contracts and, "
                            "when it is a capability library, model clients only",
                        )
                    )
            else:
                findings.append(
                    Finding(
                        path,
                        line,
                        f"{kind} {name!r} must not import {target_kind} "
                        f"{target_name!r}; simulators sit below models and robots",
                    )
                )
    return findings


def check_core_robot_names(root: Path, *, today: date | None = None) -> list[Finding]:
    """Return K5 findings for robot or benchmark names in core sources.

    Only ``rpent/`` is scanned. The exceptions in ``CORE_NAME_EXEMPTIONS``
    carry a reason and an expiry date: while they apply they suppress a
    violation and report one warning saying why, and after the date they stop
    suppressing and add an error so the exception is removed or renewed. A
    component or slice may name a robot because it is about that robot by
    definition.
    """
    names = _component_names(root / "robots")
    if not names:
        return []
    today = today or date.today()
    # Underscores are not word boundaries here: UPPER_SNAKE constants such as
    # ``ROBOTWIN_QPOS`` embed a robot name and must be reported too.
    pattern = re.compile(
        r"(?<![0-9A-Za-z])(?:%s)(?![0-9A-Za-z])"
        % "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True)),
        re.IGNORECASE,
    )
    findings: list[Finding] = []
    exemptions: dict[Path, tuple[str, date, bool]] = {}
    for relative, (reason, deadline) in CORE_NAME_EXEMPTIONS.items():
        expired = today > deadline
        exemptions[root / relative] = (reason, deadline, expired)
        if expired:
            findings.append(
                Finding(
                    root / relative,
                    0,
                    f"K5 exemption for {relative!r} expired on "
                    f"{deadline.isoformat()}; remove it or renew it with a new "
                    "reason and date",
                )
            )
    used: set[Path] = set()
    for path in sorted((root / "rpent").rglob("*.py")):
        if any(
            part in {".git", ".rpent", "__pycache__"}
            for part in path.relative_to(root).parts
        ):
            continue
        exemption = next(
            (
                (directory, values)
                for directory, values in exemptions.items()
                if path.is_relative_to(directory)
            ),
            None,
        )
        if exemption is not None and not exemption[1][2]:
            used.add(exemption[0])
            continue
        try:
            with path.open("rb") as file:
                tokens = list(tokenize.tokenize(file.readline))
        except (OSError, SyntaxError, UnicodeDecodeError, tokenize.TokenError):
            continue
        for token in tokens:
            if token.type not in {
                tokenize.NAME,
                tokenize.STRING,
                tokenize.COMMENT,
            }:
                continue
            match = pattern.search(token.string)
            if match is None:
                continue
            findings.append(
                Finding(
                    path,
                    token.start[0],
                    f"core names {match.group(0)!r}; move the robot or "
                    "benchmark detail into its slice",
                )
            )
    for directory in sorted(used):
        reason, deadline, _ = exemptions[directory]
        findings.append(
            Finding(
                directory,
                0,
                f"K5 exemption for {str(directory.relative_to(root))} applies until "
                f"{deadline.isoformat()}: {reason}",
                WARNING,
            )
        )
    return findings


def check_group_packages(root: Path) -> list[Finding]:
    """Return K2 findings for package groups without ``__init__.py``."""
    findings: list[Finding] = []
    for relative in GROUP_PACKAGES:
        directory = root / relative
        if directory.is_dir() and not (directory / "__init__.py").is_file():
            findings.append(
                Finding(
                    directory,
                    0,
                    f"package group {relative}/ needs __init__.py so an empty "
                    "group remains importable",
                )
            )
    return findings


def check_pure_data_imports(root: Path) -> list[Finding]:
    """Return K4 findings for behavior imports in pure-data modules.

    Wire contracts, control/observation adapters, and model platform tables
    must stay importable without a simulator or model runtime. NumPy and core
    contracts are data dependencies; everything else belongs in an adapter,
    server, or tool implementation. A model package entry point follows the
    same rule: importing the package must not load its runtime, whether the
    import names a third-party package or one of the package's own modules.
    """
    findings: list[Finding] = []
    stdlib = sys.stdlib_module_names
    for path in sorted(root.rglob("*.py")):
        if any(
            part in {".git", ".rpent", "tests", "docs"}
            for part in path.relative_to(root).parts
        ):
            continue
        owner = _owner(path, root)
        if path.name in PURE_DATA_FILES:
            if owner is None:
                continue
        elif path.name == "__init__.py":
            if owner is None or owner[0] != "model":
                continue
            entry = path == _component_directory(root, owner) / "__init__.py"
            findings.extend(_model_entry_findings(path, root, owner, entry=entry))
            continue
        else:
            continue
        for line, module in _imports(path, root):
            top = module.split(".", 1)[0]
            if (
                top in stdlib
                or top in PURE_DATA_ALLOWED_ROOTS
                or top in {"rpent", "robots", "components"}
            ):
                continue
            findings.append(
                Finding(
                    path,
                    line,
                    f"pure-data module imports third-party {top!r}; keep wire "
                    "data independent of runtimes and move behavior into the "
                    "adapter or server",
                )
            )
    return findings


def _model_entry_findings(
    path: Path, root: Path, owner: tuple[str, str], *, entry: bool
) -> list[Finding]:
    """Return K4 findings for imports a model package entry point runs eagerly.

    The entry point may export literal settings and type-checking imports; any
    other statement runs when a consumer imports a module of the package, so an
    import there loads the runtime that module was meant to keep out. Nested
    packages keep reporting third-party imports only: re-exporting their own
    modules is how a subpackage is assembled.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    eager: set[int] = set()
    for node in tree.body:
        if _is_lazy_entry_statement(node):
            continue
        eager.update(
            child.lineno
            for child in ast.walk(node)
            if isinstance(child, (ast.Import, ast.ImportFrom))
        )
    findings: list[Finding] = []
    reported: set[int] = set()
    stdlib = sys.stdlib_module_names
    for line, module in _imports(path, root):
        if line not in eager or line in reported:
            continue
        top = module.split(".", 1)[0]
        if entry and _component_target(module) == owner:
            message = (
                "package entry point imports its own runtime module; import "
                "it in the server or tool implementation so the package stays "
                "importable without that runtime"
            )
        elif (
            top in stdlib
            or top in PURE_DATA_ALLOWED_ROOTS
            or top in {"rpent", "robots", "components"}
        ):
            continue
        else:
            message = (
                f"package entry point imports third-party {top!r}; keep the "
                "package importable without a runtime and move behavior into "
                "the server or tool implementation"
            )
        findings.append(Finding(path, line, message))
        reported.add(line)
    return findings


def _is_lazy_entry_statement(node: ast.stmt) -> bool:
    """Return whether ``node`` keeps a package entry point free of runtime."""
    if _is_lazy_statement(node):
        return True
    # Model packages export literal settings from their entry point
    # (``USES_SESSIONS = False``); assigning a constant loads no runtime.
    return (
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and all(isinstance(target, ast.Name) for target in node.targets)
    )


def check_lazy_slices(root: Path) -> list[Finding]:
    """Return K6 findings for slice ``__init__.py`` files that import eagerly."""
    findings: list[Finding] = []
    robots = root / "robots"
    if not robots.is_dir():
        return findings
    for path in sorted(robots.glob("*/__init__.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if _is_lazy_statement(node):
                continue
            findings.append(
                Finding(
                    path,
                    node.lineno,
                    "slice __init__.py must stay lazy; export through "
                    "__getattr__ so a missing optional dependency does not "
                    "break robot discovery",
                )
            )
    return findings


def _is_lazy_statement(node: ast.stmt) -> bool:
    """Return whether ``node`` is part of the lazy slice entry pattern."""
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        # Module docstring.
        return isinstance(node.value.value, str)
    if isinstance(node, ast.ImportFrom):
        return node.module in {"__future__", "typing", "typing_extensions"}
    if isinstance(node, (ast.Import,)):
        return False
    if isinstance(node, ast.Assign):
        return all(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    if isinstance(node, ast.FunctionDef):
        return node.name == "__getattr__"
    if isinstance(node, ast.If):
        # Type-checking imports never execute at runtime.
        return _is_type_checking(node.test) and all(
            isinstance(child, (ast.Import, ast.ImportFrom)) for child in node.body
        )
    return False


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def check_coverage(root: Path) -> list[Finding]:
    """Return K2 findings for missing or mismatched component manifests."""
    findings: list[Finding] = []
    try:
        components = discover_components(root)
    except InstallConfigError as exc:
        return [Finding(root / "component.toml", 0, str(exc))]
    declared = {component.path.resolve() for component in components.values()}
    roots = [
        (root / "robots", "robot"),
        (root / "components" / "models", "model"),
        (root / "components" / "sim", "sim"),
        (root / "components" / "tools", "tool"),
        (root / "components" / "infra", "infra"),
    ]
    for directory, kind in roots:
        if not directory.is_dir():
            continue
        for child in sorted(directory.iterdir()):
            if not child.is_dir() or child.name.startswith((".", "_")):
                continue
            if kind == "model" and not any(child.glob("*.py")):
                continue
            if child.resolve() not in declared:
                findings.append(
                    Finding(
                        child / "component.toml",
                        0,
                        f"{kind} directory has no component.toml",
                    )
                )
    return findings


def check_role_files(root: Path, *, today: date | None = None) -> list[Finding]:
    """Return K2 findings for components missing their skeleton files.

    Components that predate the skeleton warn until ``MIGRATION_DEADLINE`` and
    fail afterwards, so a migration exception cannot outlive its schedule.
    """
    findings: list[Finding] = []
    expired = (today or date.today()) > MIGRATION_DEADLINE
    roots = [
        (root / "robots", "robot"),
        (root / "components" / "models", "model"),
        (root / "components" / "sim", "sim"),
        (root / "components" / "tools", "tool"),
    ]
    for directory, kind in roots:
        if not directory.is_dir():
            continue
        for child in sorted(directory.iterdir()):
            if not child.is_dir() or child.name.startswith((".", "_")):
                continue
            if not any(child.glob("*.py")):
                # A component that owns no code has no role files to check; its
                # manifest is what it contributes.
                continue
            required = REQUIRED_ROLE_FILES[kind]
            missing = [name for name in required if not (child / name).exists()]
            if not missing:
                continue
            legacy = (kind, child.name) in LEGACY_COMPONENTS
            level = WARNING if legacy and not expired else ERROR
            note = (
                f" (migration deadline {MIGRATION_DEADLINE.isoformat()} passed)"
                if legacy and expired
                else ""
            )
            findings.append(
                Finding(
                    child,
                    0,
                    f"{kind} {child.name!r} is missing its skeleton files: "
                    f"{', '.join(missing)}{note}",
                    level,
                )
            )
    return findings


def check_legacy_names(root: Path, *, today: date | None = None) -> list[Finding]:
    """Return K3 findings for retired slice-root names.

    Pre-migration names warn until ``MIGRATION_DEADLINE``; after that they are
    errors, not a permanent allowance.
    """
    findings: list[Finding] = []
    expired = (today or date.today()) > MIGRATION_DEADLINE
    robots = root / "robots"
    if not robots.is_dir():
        return findings
    for slice_dir in sorted(robots.iterdir()):
        if not slice_dir.is_dir() or slice_dir.name.startswith((".", "_")):
            continue
        baseline = LEGACY_SLICES.get(slice_dir.name, frozenset())
        present = sorted(
            entry.name
            for entry in slice_dir.iterdir()
            if entry.name in LEGACY_SLICE_NAMES
        )
        if not present:
            continue
        if slice_dir.name in MIGRATED_SLICES:
            findings.append(
                Finding(
                    slice_dir,
                    0,
                    f"slice {slice_dir.name!r} is migrated but keeps retired "
                    f"names: {', '.join(present)}",
                )
            )
            continue
        added = [name for name in present if name not in baseline]
        if added:
            findings.append(
                Finding(
                    slice_dir,
                    0,
                    f"slice {slice_dir.name!r} added retired names "
                    f"{', '.join(added)}; use the role names instead",
                )
            )
        kept = [name for name in present if name in baseline]
        if kept:
            note = (
                f"; the migration deadline {MIGRATION_DEADLINE.isoformat()} has passed"
                if expired
                else "; the slice migration renames them and turns this into an error"
            )
            findings.append(
                Finding(
                    slice_dir,
                    0,
                    f"slice {slice_dir.name!r} still has {len(kept)} pre-migration "
                    f"names ({', '.join(kept)}){note}",
                    ERROR if expired else WARNING,
                )
            )
    return findings


def check_file_sizes(root: Path) -> list[Finding]:
    """Return K7 warnings for role files over their line-count limit."""
    findings: list[Finding] = []
    for directory in ("robots", "components/models", "components/sim"):
        for path in sorted((root / directory).rglob("*.py")):
            limit = LINE_LIMITS.get(path.name)
            if limit is None:
                continue
            try:
                lines = sum(1 for _ in path.open(encoding="utf-8"))
            except OSError:
                continue
            if lines > limit:
                findings.append(
                    Finding(
                        path,
                        0,
                        f"{path.name} has {lines} lines (limit {limit}); "
                        "split it along the role files",
                        WARNING,
                    )
                )
    return findings


def _component_names(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {
        child.name
        for child in directory.iterdir()
        if child.is_dir() and not child.name.startswith((".", "_"))
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tests.tooling.layout",
        description="Check the repository layout and import boundaries.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to check.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    findings = (
        check_top_level(root)
        + check_coverage(root)
        + check_group_packages(root)
        + check_role_files(root)
        + check_legacy_names(root)
        + check_imports(root)
        + check_pure_data_imports(root)
        + check_core_robot_names(root)
        + check_lazy_slices(root)
        + check_file_sizes(root)
    )
    # A bare alias reaches a rule twice: once as the package path
    # (``components.models.sam3``) and once expanded to the imported module
    # (``components.models.sam3.server``). One violation is one line, so identical
    # findings collapse while different ones on the same line remain.
    findings = list(
        {
            (finding.path, finding.line, finding.level, finding.message): finding
            for finding in findings
        }.values()
    )
    for finding in findings:
        location = (
            finding.path.relative_to(root)
            if finding.path.is_relative_to(root)
            else finding.path
        )
        suffix = f":{finding.line}" if finding.line else ""
        print(f"{location}{suffix}: {finding.level}: {finding.message}")
    errors = [finding for finding in findings if finding.level == ERROR]
    if errors:
        print(f"{len(errors)} layout violation(s)", file=sys.stderr)
        return 1
    print("layout ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
