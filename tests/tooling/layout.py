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

"""Check component manifests, import boundaries, and lazy package entries.

Rules enforced against the current tree:

- Every component directory under ``robots/`` and ``components/{sim,models,tools,infra}/``
  declares a valid ``component.toml``, and package groups carry ``__init__.py``.
- Layer import direction is ``rpent`` ← ``components/sim`` ← ``components/models``
  ← ``robots``. Core ``rpent/`` never imports concrete robots, simulators, models,
  or tools (except ``rpent/cli/flywheel.py``). Tools depend only on ``rpent`` and
  model clients. Sibling imports across robots require an explicit ``imports``
  declaration in ``component.toml``; sibling model/sim/tool imports are forbidden.
- ``robots/<name>/__init__.py`` stays lazy so robot discovery works without
  optional robot dependencies; ``components/models/<name>/__init__.py`` avoids
  eager runtime imports.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

from rpent.install.errors import InstallConfigError
from rpent.install.manifest import COMPONENT_DIRECTORIES, discover_components

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = ("rpent", "robots", "components")
GROUP_PACKAGES = (
    "robots",
    "components",
    "components/models",
    "components/sim",
    "components/tools",
)
COMPONENT_MODULE_KINDS = {"sim": "sim", "models": "model", "tools": "tool"}
CORE_TOOL_ENTRY_POINTS = frozenset({"rpent/cli/flywheel.py"})
PURE_DATA_ALLOWED_ROOTS = frozenset({"numpy", "PIL", "pydantic", "typing_extensions"})

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    """One layout or boundary violation."""

    path: Path
    line: int
    message: str
    level: str = ERROR


def _iter_source_py(root: Path) -> list[Path]:
    paths: list[Path] = []
    for top in SOURCE_ROOTS:
        directory = root / top
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            if path.name.startswith(".") or "__pycache__" in path.parts:
                continue
            paths.append(path)
    return paths


def _module_parts(path: Path, root: Path) -> tuple[str, ...]:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts:
        parts.pop()
    return tuple(parts)


def _relative_target(
    path: Path, root: Path, node: ast.ImportFrom
) -> tuple[str, ...] | None:
    package = _module_parts(path, root)
    if node.level - 1 > len(package):
        return None
    base = package[: len(package) - (node.level - 1)]
    if node.module:
        base = (*base, *node.module.split("."))
    return base or None


def _named_imports(
    root: Path, base: tuple[str, ...], node: ast.ImportFrom
) -> list[tuple[int, str]]:
    imports: list[tuple[int, str]] = []
    for alias in node.names:
        if alias.name == "*":
            continue
        candidate = root.joinpath(*base, alias.name)
        if candidate.is_dir() or candidate.with_suffix(".py").is_file():
            imports.append((node.lineno, ".".join((*base, alias.name))))
    return imports


def _literal_import_target(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr
    else:
        return None
    if name not in {"import_module", "__import__"}:
        return None
    if (
        node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


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
            else:
                base = _relative_target(path, root, node)
                if base is not None:
                    imports.append((node.lineno, ".".join(base)))
                    imports.extend(_named_imports(root, base, node))
        elif isinstance(node, ast.Call):
            target = _literal_import_target(node)
            if target is not None:
                imports.append((node.lineno, target))
    return imports


def _under(module: str, package: str) -> bool:
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


def _component_target(module: str) -> tuple[str, str] | None:
    parts = module.split(".")
    if len(parts) >= 3 and parts[0] == "components":
        kind = COMPONENT_MODULE_KINDS.get(parts[1])
        if kind is not None:
            return kind, parts[2]
    if len(parts) >= 2 and parts[0] == "robots":
        return "robot", parts[1]
    return None


def _declared_imports(root: Path) -> dict[str, set[str]]:
    try:
        components = discover_components(root)
    except InstallConfigError:
        return {}
    return {
        name.removeprefix("robot-"): {
            other.removeprefix("robot-") for other in comp.imports
        }
        for name, comp in components.items()
        if comp.kind == "robot"
    }


def check_components(root: Path) -> list[Finding]:
    """Verify component manifests and package-group ``__init__.py`` files."""
    findings: list[Finding] = []
    for relative in GROUP_PACKAGES:
        directory = root / relative
        if directory.is_dir() and not (directory / "__init__.py").is_file():
            findings.append(
                Finding(directory, 0, f"package group {relative}/ needs __init__.py")
            )
    try:
        components = discover_components(root)
    except InstallConfigError as exc:
        return [*findings, Finding(root / "component.toml", 0, str(exc))]

    declared = {comp.path.resolve() for comp in components.values()}
    for kind, rel_dir in COMPONENT_DIRECTORIES.items():
        directory = root / rel_dir
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


def check_imports(root: Path) -> list[Finding]:
    """Verify layer import direction across ``rpent/``, ``robots/``, and ``components/``."""
    findings: list[Finding] = []
    declared_imports = _declared_imports(root)
    for path in _iter_source_py(root):
        owner = _owner(path, root)
        if owner is None:
            continue
        relative = path.relative_to(root).as_posix()
        kind, name = owner
        for line, module in _imports(path, root):
            target = _component_target(module)
            if kind == "core":
                if _under(module, "components.models"):
                    findings.append(
                        Finding(path, line, "core must not import components.models")
                    )
                elif _under(module, "components.sim"):
                    findings.append(
                        Finding(path, line, "core must not import components.sim")
                    )
                elif _under(module, "components.tools"):
                    if relative not in CORE_TOOL_ENTRY_POINTS:
                        findings.append(
                            Finding(path, line, "core must not import components.tools")
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
            if target is None or target == owner:
                continue
            target_kind, target_name = target
            if kind == "robot":
                if target_kind == "robot" and target_name not in declared_imports.get(
                    name, set()
                ):
                    findings.append(
                        Finding(
                            path,
                            line,
                            f"robot {name!r} must not import robot {target_name!r} "
                            "without declaring it in component.toml imports",
                        )
                    )
            elif kind == "model":
                if target_kind in {"robot", "model", "tool"}:
                    findings.append(
                        Finding(
                            path,
                            line,
                            f"model {name!r} must not import {target_kind} {target_name!r}",
                        )
                    )
            elif kind == "tool":
                if target_kind != "model":
                    findings.append(
                        Finding(
                            path,
                            line,
                            f"tool {name!r} must not import {target_kind} {target_name!r}",
                        )
                    )
            else:
                findings.append(
                    Finding(
                        path,
                        line,
                        f"{kind} {name!r} must not import {target_kind} {target_name!r}",
                    )
                )
    return findings


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _is_lazy_statement(node: ast.stmt) -> bool:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return isinstance(node.value.value, str)
    if isinstance(node, ast.ImportFrom):
        return node.module in {"__future__", "typing", "typing_extensions"}
    if isinstance(node, ast.Import):
        return False
    if isinstance(node, ast.Assign):
        return all(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    if isinstance(node, ast.FunctionDef):
        return node.name == "__getattr__"
    if isinstance(node, ast.If):
        return (
            _is_type_checking(node.test)
            and not node.orelse
            and all(
                isinstance(child, (ast.Import, ast.ImportFrom)) for child in node.body
            )
        )
    return False


def _is_lazy_model_statement(node: ast.stmt) -> bool:
    if _is_lazy_statement(node):
        return True
    return (
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and all(isinstance(target, ast.Name) for target in node.targets)
    )


def check_lazy_entries(root: Path) -> list[Finding]:
    """Verify robot and model ``__init__.py`` entry points do not import eagerly."""
    findings: list[Finding] = []
    robots = root / "robots"
    if robots.is_dir():
        for path in sorted(robots.glob("*/__init__.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for node in tree.body:
                if not _is_lazy_statement(node):
                    findings.append(
                        Finding(
                            path,
                            node.lineno,
                            "robot __init__.py must stay lazy; export via __getattr__",
                        )
                    )

    models = root / "components" / "models"
    if models.is_dir():
        stdlib = sys.stdlib_module_names
        for path in sorted(models.glob("*/__init__.py")):
            owner = _owner(path, root)
            if owner is None:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            eager_lines: set[int] = set()
            for node in tree.body:
                if _is_lazy_model_statement(node):
                    continue
                for child in ast.walk(node):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        eager_lines.add(child.lineno)
                    elif (
                        isinstance(child, ast.Call)
                        and _literal_import_target(child) is not None
                    ):
                        eager_lines.add(child.lineno)
            reported: set[int] = set()
            for line, module in _imports(path, root):
                if line not in eager_lines or line in reported:
                    continue
                top = module.split(".", 1)[0]
                if _component_target(module) == owner:
                    findings.append(
                        Finding(
                            path,
                            line,
                            "model __init__.py must not eagerly import its own runtime module",
                        )
                    )
                    reported.add(line)
                elif (
                    top not in stdlib
                    and top not in PURE_DATA_ALLOWED_ROOTS
                    and top not in {"rpent", "robots", "components"}
                ):
                    findings.append(
                        Finding(
                            path,
                            line,
                            f"model __init__.py must not eagerly import third-party {top!r}",
                        )
                    )
                    reported.add(line)
    return findings


def main(argv: list[str] | None = None) -> int:
    """Run all layout checks under ``--root``."""
    parser = argparse.ArgumentParser(
        prog="tests.tooling.layout",
        description="Check component manifests, import boundaries, and lazy entries.",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    findings = check_components(root) + check_imports(root) + check_lazy_entries(root)
    unique = list({(f.path, f.line, f.level, f.message): f for f in findings}.values())
    for finding in unique:
        location = (
            finding.path.relative_to(root)
            if finding.path.is_relative_to(root)
            else finding.path
        )
        suffix = f":{finding.line}" if finding.line else ""
        print(f"{location}{suffix}: {finding.level}: {finding.message}")
    errors = [f for f in unique if f.level == ERROR]
    if errors:
        print(f"{len(errors)} layout violation(s)", file=sys.stderr)
        return 1
    print("layout ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
