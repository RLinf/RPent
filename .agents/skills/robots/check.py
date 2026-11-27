#!/usr/bin/env python3
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

"""Check changed RPent robot integrations, packaging extras, and GPU e2e wiring.

Usage::

    python3 .agents/skills/robots/check.py            # files changed against main
    python3 .agents/skills/robots/check.py FILE ...
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import route  # noqa: E402

PYPROJECT = "pyproject.toml"
GPU_RUNNER = "tests/e2e_tests/run_gpu_suite.sh"
GPU_TARGETS = "tests/tooling/ci_targets.py"
GPU_SUITES_PREFIX = "tests/e2e_tests/suites/"

REQUIRED_EXPORTS = ("get_robot_spec", "get_toolkit")

# Heavyweight optional dependencies that must be imported lazily inside
# functions/branches rather than at top level in robot entry modules.
OPTIONAL_ROOTS = frozenset(
    {
        "curobo",
        "cv2",
        "franky",
        "franky_control",
        "isaacsim",
        "lerobot",
        "libero",
        "liberopro",
        "mujoco",
        "omni",
        "openpi",
        "pyrealsense2",
        "rlinf",
        "robocasa",
        "robosuite",
        "rospy",
        "sam3",
        "torch",
        "torchvision",
        "transformers",
    }
)

GIT_URL = re.compile(r"git\+https?://[^\s;]+")
GIT_PIN = re.compile(r"@[^\s;]+$")


def _is_type_checking_guard(test: ast.expr) -> bool:
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _top_level_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """Return ``(lineno, top_level_package)`` for eager module-level imports."""
    found: list[tuple[int, str]] = []

    def visit(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    found.append((stmt.lineno, alias.name.split(".")[0]))
            elif isinstance(stmt, ast.ImportFrom):
                if stmt.level == 0 and stmt.module:
                    found.append((stmt.lineno, stmt.module.split(".")[0]))
            elif isinstance(stmt, ast.If):
                if not _is_type_checking_guard(stmt.test):
                    visit(stmt.body)
                visit(stmt.orelse)
            elif isinstance(stmt, ast.Try):
                visit(stmt.body)
                for handler in stmt.handlers:
                    visit(handler.body)
                visit(stmt.orelse)
                visit(stmt.finalbody)

    visit(tree.body)
    return found


def _exported_names(tree: ast.Module) -> set[str]:
    """Names directly defined, imported, or declared in __all__ (with __getattr__)."""
    names: set[str] = set()
    has_getattr = False
    all_names: set[str] = set()
    const_sequences: dict[str, set[str]] = {}

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
            if stmt.name == "__getattr__":
                has_getattr = True
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                names.add(alias.asname or alias.name)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if not isinstance(target, ast.Name):
                    continue
                if isinstance(stmt.value, (ast.List, ast.Tuple)):
                    vals = {
                        elt.value
                        for elt in stmt.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    }
                    const_sequences[target.id] = vals
                    if target.id == "__all__":
                        all_names.update(vals)
                elif (
                    target.id == "__all__"
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id == "list"
                    and len(stmt.value.args) == 1
                    and isinstance(stmt.value.args[0], ast.Name)
                ):
                    all_names.update(const_sequences.get(stmt.value.args[0].id, set()))

    if has_getattr:
        names.update(all_names)
    return names


def check_exports(robot_dir: str, root: Path | None = None) -> list[route.Finding]:
    """Check that robots/<name>/__init__.py exposes get_robot_spec and get_toolkit."""
    base = root or route.REPO_ROOT
    robot_path = base / robot_dir
    init_rel = f"{robot_dir}/__init__.py"
    init_path = base / init_rel
    if not init_path.is_file():
        # Manifest-only robot components (e.g. robots/libero_plus) declare
        # dependencies without owning a Python package.
        if (robot_path / "component.toml").is_file() and not any(
            robot_path.glob("*.py")
        ):
            return []
        return [
            route.Finding(
                "major",
                robot_dir,
                None,
                "robots/exports",
                "robot package is missing __init__.py exposing get_robot_spec and get_toolkit",
            )
        ]
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=init_rel)
    except SyntaxError as exc:
        return [
            route.Finding("critical", init_rel, exc.lineno, "robots/exports", str(exc))
        ]
    exported = _exported_names(tree)
    missing = [name for name in REQUIRED_EXPORTS if name not in exported]
    if not missing:
        return []
    return [
        route.Finding(
            "major",
            init_rel,
            None,
            "robots/exports",
            f"robot package __init__.py does not expose {', '.join(missing)}; "
            "rpent.robots.base requires both factories",
        )
    ]


def check_lazy_imports(rel: str, root: Path | None = None) -> list[route.Finding]:
    """Flag top-level imports of optional simulator/VLA/hardware packages in entry modules."""
    base = root or route.REPO_ROOT
    path = base / rel
    if not path.is_file():
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
    except SyntaxError:
        return []
    findings: list[route.Finding] = []
    for lineno, pkg in _top_level_imports(tree):
        if pkg in OPTIONAL_ROOTS:
            findings.append(
                route.Finding(
                    "major",
                    rel,
                    lineno,
                    "robots/lazy-import",
                    f"top-level import of optional dependency {pkg!r}; import it "
                    "inside the function or branch that uses it so robot discovery "
                    "and CLI help work without this extra installed",
                )
            )
    return findings


def check_packaging(root: Path | None = None) -> list[route.Finding]:
    """Validate pyproject.toml extras, conflicts, git pins, and package discovery."""
    if tomllib is None:
        print("skipped: pyproject.toml check needs tomllib (Python 3.11+) or tomli")
        return []
    base = root or route.REPO_ROOT
    pyproject_path = base / PYPROJECT
    if not pyproject_path.is_file():
        return []
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    findings: list[route.Finding] = []

    extras = data.get("project", {}).get("optional-dependencies", {})
    for extra_name, deps in extras.items():
        for dep in deps:
            match = GIT_URL.search(dep)
            if match and not GIT_PIN.search(match.group(0)):
                findings.append(
                    route.Finding(
                        "major",
                        PYPROJECT,
                        None,
                        "robots/packaging",
                        f"extra {extra_name!r} has unpinned git dependency {dep!r}; "
                        "pin to @<commit|tag|branch>",
                    )
                )

    conflicts = data.get("tool", {}).get("uv", {}).get("conflicts", [])
    for group in conflicts:
        for item in group:
            extra_name = item.get("extra") if isinstance(item, dict) else None
            if extra_name and extra_name not in extras:
                findings.append(
                    route.Finding(
                        "major",
                        PYPROJECT,
                        None,
                        "robots/packaging",
                        f"[tool.uv] conflicts references unknown extra {extra_name!r}",
                    )
                )

    find_section = (
        data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find")
    )
    if find_section is not None or not (base / "setup.py").is_file():
        includes = set((find_section or {}).get("include", []))
        for required in ("rpent*", "robots", "robots.*"):
            if required not in includes:
                findings.append(
                    route.Finding(
                        "major",
                        PYPROJECT,
                        None,
                        "robots/packaging",
                        f"[tool.setuptools.packages.find] include is missing {required!r}",
                    )
                )
    return findings


def check_e2e_syntax(rel: str, root: Path | None = None) -> list[route.Finding]:
    base = root or route.REPO_ROOT
    result = subprocess.run(
        ["bash", "-n", rel], cwd=base, capture_output=True, text=True
    )
    if result.returncode == 0:
        return []
    return [
        route.Finding("critical", rel, None, "robots/e2e-syntax", result.stderr.strip())
    ]


def check_e2e_targets(root: Path | None = None) -> list[route.Finding]:
    """Validate declarative E2E suite TOMLs under tests/e2e_tests/suites/."""
    if tomllib is None:
        return []
    base = root or route.REPO_ROOT
    suites_dir = base / "tests/e2e_tests/suites"
    if not suites_dir.is_dir():
        return []

    findings: list[route.Finding] = []
    toml_files = sorted(suites_dir.glob("*.toml"))
    if not toml_files:
        findings.append(
            route.Finding(
                "major",
                "tests/e2e_tests/suites",
                None,
                "robots/e2e-targets",
                "no declarative E2E suite *.toml files found",
            )
        )
        return findings

    for path in toml_files:
        rel = path.relative_to(base).as_posix()
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            findings.append(
                route.Finding("critical", rel, None, "robots/e2e-targets", str(exc))
            )
            continue
        if not data.get("component"):
            findings.append(
                route.Finding(
                    "major",
                    rel,
                    None,
                    "robots/e2e-targets",
                    "suite TOML is missing required 'component' field",
                )
            )
        if not data.get("cases"):
            findings.append(
                route.Finding(
                    "major",
                    rel,
                    None,
                    "robots/e2e-targets",
                    "suite TOML must define at least one [[cases]] entry",
                )
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    args = route.check_parser(__doc__.splitlines()[0]).parse_args(argv)
    selection = route.select("robots", args)
    findings: list[route.Finding] = []

    touched_robots: set[str] = set()
    for change in selection.changes:
        if change.status == "D":
            continue
        match = re.match(r"^(robots/[a-zA-Z0-9_]+)/", change.path)
        if match and not match.group(1).endswith("/__pycache__"):
            touched_robots.add(match.group(1))
        if change.path.startswith("tests/e2e_tests/") and change.path.endswith(".sh"):
            findings += check_e2e_syntax(change.path)

    if selection.all_lines:
        robots_root = route.REPO_ROOT / "robots"
        if robots_root.is_dir():
            for child in sorted(robots_root.iterdir()):
                if child.is_dir() and not child.name.startswith(("_", ".")):
                    touched_robots.add(f"robots/{child.name}")

    for robot_dir in sorted(touched_robots):
        findings += check_exports(robot_dir)
        for entry in (f"{robot_dir}/__init__.py", f"{robot_dir}/robot_spec.py"):
            if (route.REPO_ROOT / entry).is_file():
                entry_findings = check_lazy_imports(entry)
                findings += (
                    entry_findings
                    if selection.all_lines
                    else selection.relevant(entry_findings)
                )

    changed_paths = {c.path for c in selection.changes}
    if selection.all_lines or PYPROJECT in changed_paths:
        findings += check_packaging()
    if (
        selection.all_lines
        or {GPU_RUNNER, GPU_TARGETS} & changed_paths
        or any(path.startswith(GPU_SUITES_PREFIX) for path in changed_paths)
    ):
        findings += check_e2e_targets()

    return route.report("robots", findings)


if __name__ == "__main__":
    raise SystemExit(main())
