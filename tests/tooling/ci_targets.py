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

"""Select fine-grained install and GPU eval CI jobs affected by a Git diff."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_INSTALL_GLOBAL_PREFIXES = ("rpent/install/",)
_INSTALL_GLOBAL_FILES = frozenset(
    {
        "rpent/cli/install.py",
        "rpent-install.sh",
        "pyproject.toml",
        "setup.py",
        "MANIFEST.in",
        "tests/e2e_tests/run_gpu_suite.sh",
        ".github/workflows/embodied-gpu-e2e.yml",
    }
)
_RUNTIME_GLOBAL_FILES = frozenset(
    {
        "tests/tooling/ci_targets.py",
        "tests/tooling/gpu_slots.py",
    }
)


def _is_component_install_file(rel_path: str) -> bool:
    """Whether a file inside a component directory affects its installation."""
    path = Path(rel_path)
    if path.name == "component.toml":
        return True
    if "requirements" in path.parts:
        return True
    return path.name.endswith(("requirements.txt", "override.txt", "constraints.txt"))


def _load_suites(root: Path) -> list[dict[str, Any]]:
    suites_dir = root / "tests/e2e_tests/suites"
    if not suites_dir.is_dir():
        return []
    result: list[dict[str, Any]] = []
    for path in sorted(suites_dir.glob("*.toml"), key=lambda p: p.stem):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        data.setdefault("suite", path.stem)
        data.setdefault("robot", path.stem)
        data.setdefault("optional_models", [])
        data.setdefault("cases", [])
        result.append(data)
    return result


def _load_manifests(
    root: Path,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    """Return (dir->name, name->requires, name->kind)."""
    owners: dict[str, str] = {}
    requires: dict[str, list[str]] = {}
    kinds: dict[str, str] = {}
    for pattern in ("robots/*/component.toml", "components/*/*/component.toml"):
        for path in sorted(root.glob(pattern)):
            manifest = tomllib.loads(path.read_text(encoding="utf-8"))
            rel_dir = path.parent.relative_to(root).as_posix()
            name = manifest["name"]
            owners[rel_dir] = name
            requires[name] = [str(r) for r in (manifest.get("requires") or ())]
            kinds[name] = "robot" if rel_dir.startswith("robots/") else "component"
    return owners, requires, kinds


def _closure(component: str, requires: Mapping[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    pending = [component]
    while pending:
        name = pending.pop()
        if name not in seen:
            seen.add(name)
            pending.extend(requires.get(name, ()))
    return seen


def _case_is_opt_in_disabled(
    case: Mapping[str, Any], environ: Mapping[str, str]
) -> bool:
    for var in case.get("require_env") or ():
        if not environ.get(str(var), "").strip():
            return True
    opt_in = case.get("opt_in_env")
    if opt_in and environ.get(str(opt_in), "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        return True
    return False


def plan_ci(
    paths: list[str],
    root: Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compute fine-grained install_jobs and eval_jobs matrices for changed paths.

    Distinguishes install-affecting changes (which trigger fresh install jobs)
    from runtime-only code changes (which skip install jobs and run eval
    directly against content-hashed cached venvs).
    """
    env = os.environ if environ is None else environ
    suites = _load_suites(root)
    owners, requires, kinds = _load_manifests(root)
    suites_by_name = {s["suite"]: s for s in suites}

    # Build dependency reverse map: component -> set of E2E robot components using it.
    e2e_robot_components = {s["component"] for s in suites}
    consumers_by_component: dict[str, set[str]] = {}
    covered_by_e2e: set[str] = set()
    for robot_comp in e2e_robot_components:
        deps = _closure(robot_comp, requires)
        covered_by_e2e.update(deps)
        for dep in deps:
            consumers_by_component.setdefault(dep, set()).add(robot_comp)

    # Catalog all (suite, case) entries in deterministic order.
    all_eval_keys: list[tuple[str, str]] = []
    cases_by_key: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    cases_by_component: dict[str, list[tuple[str, str]]] = {}
    cases_by_robot: dict[str, list[tuple[str, str]]] = {}
    cases_by_model: dict[str, list[tuple[str, str]]] = {}
    for suite in suites:
        s_name = suite["suite"]
        s_comp = suite["component"]
        s_robot = suite["robot"]
        for case in suite["cases"]:
            c_name = case["name"]
            key = (s_name, c_name)
            all_eval_keys.append(key)
            cases_by_key[key] = (suite, case)
            cases_by_component.setdefault(s_comp, []).append(key)
            cases_by_robot.setdefault(s_robot, []).append(key)
            for model in case.get("models") or ():
                cases_by_model.setdefault(str(model), []).append(key)

    selected_installs: set[str] = set()
    selected_evals: set[tuple[str, str]] = set()
    full_install = False
    full_eval = False

    for filename in paths:
        if filename.startswith("tests/unit_tests/"):
            continue
        name = filename
        if (
            name.startswith(("docs/", ".agents/", ".claude/"))
            or name.endswith((".md", ".rst"))
            and not name.startswith(("robots/", "rpent/", "components/"))
        ):
            continue

        # 1. Global install-affecting core changes -> full install + full eval.
        if name.startswith(_INSTALL_GLOBAL_PREFIXES) or name in _INSTALL_GLOBAL_FILES:
            full_install = True
            full_eval = True
            continue

        # 2. Suite TOML edit -> eval all cases in that suite (cache hash handles [setup] changes).
        if name.startswith("tests/e2e_tests/suites/") and name.endswith(".toml"):
            suite_stem = Path(name).stem
            if suite_stem in suites_by_name:
                for case in suites_by_name[suite_stem]["cases"]:
                    selected_evals.add((suite_stem, case["name"]))
                continue

        # 3. Component-owned file (robots/* or components/*/*).
        owner_dir = next(
            (owner for owner in owners if name.startswith(owner + "/")), None
        )
        if owner_dir is not None:
            comp_name = owners[owner_dir]
            is_install = _is_component_install_file(name)

            # Determine affected eval cases.
            affected_evals: list[tuple[str, str]] = []
            if owner_dir.startswith("components/models/"):
                affected_evals.extend(cases_by_model.get(comp_name, ()))
                if not affected_evals:
                    # Fallback for models listed only via optional_models or requires.
                    short = owner_dir.split("/")[-1]
                    for suite in suites:
                        if short in suite.get("optional_models", ()):
                            affected_evals.extend(
                                (suite["suite"], c["name"]) for c in suite["cases"]
                            )
                    for robot_comp in consumers_by_component.get(comp_name, ()):
                        affected_evals.extend(cases_by_component.get(robot_comp, ()))
            elif owner_dir.startswith(("robots/", "components/sim/")):
                short = owner_dir.split("/")[-1]
                affected_evals.extend(cases_by_component.get(comp_name, ()))
                affected_evals.extend(cases_by_robot.get(short, ()))
                for robot_comp in consumers_by_component.get(comp_name, ()):
                    affected_evals.extend(cases_by_component.get(robot_comp, ()))
            else:
                for robot_comp in consumers_by_component.get(comp_name, ()):
                    affected_evals.extend(cases_by_component.get(robot_comp, ()))

            selected_evals.update(affected_evals)

            # Only trigger install jobs when install metadata/deps changed.
            if is_install:
                selected_installs.add(comp_name)
                for robot_comp in consumers_by_component.get(comp_name, ()):
                    selected_installs.add(robot_comp)
                for s_name, _ in affected_evals:
                    selected_installs.add(suites_by_name[s_name]["component"])
            continue

        # 4. Global runtime-only core changes -> eval all suites, skip install jobs.
        if (
            name.startswith(
                (
                    "rpent/",
                    "components/",
                    "robots/",
                    "tests/e2e_tests/",
                    ".github/workflows/",
                )
            )
            or name in _RUNTIME_GLOBAL_FILES
        ):
            full_eval = True

    if full_eval:
        selected_evals.update(all_eval_keys)

    if full_install:
        selected_installs.update(e2e_robot_components)
        # Also test standalone components not covered by any E2E robot closure
        # (robots, tools, infra) to verify their manifests remain installable.
        for rel_dir, comp_name in owners.items():
            if comp_name in covered_by_e2e:
                continue
            if rel_dir.startswith(
                ("robots/", "components/tools/", "components/infra/")
            ):
                selected_installs.add(comp_name)

    # Order install_jobs deterministically: robot installs first, then standalone components.
    install_jobs = [
        {"component": comp, "kind": kinds.get(comp, "component")}
        for comp in sorted(
            selected_installs,
            key=lambda c: (0 if kinds.get(c) == "robot" else 1, c),
        )
    ]

    # Build eval_jobs in deterministic suite/case order, skipping opt-in cases lacking env.
    eval_jobs: list[dict[str, Any]] = []
    selected_suites_ordered: list[str] = []
    for key in all_eval_keys:
        if key not in selected_evals:
            continue
        suite, case = cases_by_key[key]
        if _case_is_opt_in_disabled(case, env):
            continue
        comp = suite["component"]
        gpus = int(case.get("gpus", 1))
        eval_jobs.append(
            {
                "id": f"{suite['suite']}:{case['name']}",
                "suite": suite["suite"],
                "case": case["name"],
                "component": comp,
                "gpus": gpus,
                "venv_mode": "run" if comp in selected_installs else "cached",
            }
        )
        if suite["suite"] not in selected_suites_ordered:
            selected_suites_ordered.append(suite["suite"])

    return {
        "install_jobs": install_jobs,
        "eval_jobs": eval_jobs,
        "suites": selected_suites_ordered,
    }


if __name__ == "__main__":
    base, head = sys.argv[1:]
    if base and set(base) != {"0"}:
        raw = subprocess.check_output(
            ["git", "diff", "--no-renames", "--name-only", "-z", base, head]
        )
        paths = [os.fsdecode(path) for path in raw.split(b"\0") if path]
    else:
        paths = ["rpent-install.sh"]

    plan = plan_ci(paths, Path.cwd())
    install_jobs = plan["install_jobs"]
    eval_jobs = plan["eval_jobs"]
    suites = plan["suites"]

    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        print(f"suites={json.dumps(suites)}", file=output)
        print(f"install_jobs={json.dumps(install_jobs)}", file=output)
        print(f"eval_jobs={json.dumps(eval_jobs)}", file=output)
        print(f"has_install={'true' if install_jobs else 'false'}", file=output)
        print(f"has_eval={'true' if eval_jobs else 'false'}", file=output)

    print(
        f"install_jobs ({len(install_jobs)}): "
        f"{', '.join(j['component'] for j in install_jobs) or 'none'}"
    )
    print(
        f"eval_jobs ({len(eval_jobs)}): "
        f"{', '.join(j['id'] + '[' + j['venv_mode'] + ']' for j in eval_jobs) or 'none'}"
    )
