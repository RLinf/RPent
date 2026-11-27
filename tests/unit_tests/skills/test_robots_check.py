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

from __future__ import annotations

import runpy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
robots_check = runpy.run_path(str(REPO_ROOT / ".agents/skills/robots/check.py"))


def test_real_repo_passes_exports_lazy_imports_packaging_and_e2e_targets():
    for robot_dir in sorted((REPO_ROOT / "robots").iterdir()):
        if not robot_dir.is_dir() or robot_dir.name.startswith(("_", ".")):
            continue
        rel = f"robots/{robot_dir.name}"
        assert robots_check["check_exports"](rel, REPO_ROOT) == []
        for entry in (f"{rel}/__init__.py", f"{rel}/robot_spec.py"):
            if (REPO_ROOT / entry).is_file():
                assert robots_check["check_lazy_imports"](entry, REPO_ROOT) == []
    assert robots_check["check_packaging"](REPO_ROOT) == []
    assert robots_check["check_e2e_targets"](REPO_ROOT) == []


def test_check_exports_handles_direct_and_lazy_getattr_and_flags_missing(tmp_path):
    direct = tmp_path / "robots" / "direct"
    direct.mkdir(parents=True)
    (direct / "__init__.py").write_text(
        "from robots.direct.spec import get_robot_spec, get_toolkit\n",
        encoding="utf-8",
    )
    assert robots_check["check_exports"]("robots/direct", tmp_path) == []

    lazy = tmp_path / "robots" / "lazy"
    lazy.mkdir(parents=True)
    (lazy / "__init__.py").write_text(
        "__all__ = ['get_robot_spec', 'get_toolkit']\ndef __getattr__(name): ...\n",
        encoding="utf-8",
    )
    assert robots_check["check_exports"]("robots/lazy", tmp_path) == []

    broken = tmp_path / "robots" / "broken"
    broken.mkdir(parents=True)
    (broken / "__init__.py").write_text("def get_robot_spec(): ...\n", encoding="utf-8")
    findings = robots_check["check_exports"]("robots/broken", tmp_path)
    assert len(findings) == 1
    assert "get_toolkit" in findings[0].message


def test_check_lazy_imports_allows_function_and_type_checking_imports(tmp_path):
    mod = tmp_path / "robots" / "demo" / "robot_spec.py"
    mod.parent.mkdir(parents=True)
    mod.write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    import torch\n"
        "def build():\n"
        "    import mujoco\n"
        "    return mujoco\n",
        encoding="utf-8",
    )
    assert (
        robots_check["check_lazy_imports"]("robots/demo/robot_spec.py", tmp_path) == []
    )

    mod.write_text("import torch\nfrom mujoco import MjModel\n", encoding="utf-8")
    findings = robots_check["check_lazy_imports"]("robots/demo/robot_spec.py", tmp_path)
    assert [f.line for f in findings] == [1, 2]
    assert all(f.rule == "robots/lazy-import" for f in findings)


def test_check_packaging_catches_unpinned_git_unknown_conflict_and_missing_include(
    tmp_path,
):
    (tmp_path / "pyproject.toml").write_text(
        """
[project.optional-dependencies]
demo = ["pkg @ git+https://github.com/example/pkg.git"]

[tool.uv]
conflicts = [[{ extra = "demo" }, { extra = "ghost" }]]

[tool.setuptools.packages.find]
include = ["rpent*"]
""",
        encoding="utf-8",
    )
    findings = robots_check["check_packaging"](tmp_path)
    messages = [f.message for f in findings]
    assert any("unpinned git dependency" in m for m in messages)
    assert any("unknown extra 'ghost'" in m for m in messages)
    assert any("missing 'robots'" in m for m in messages)


def test_check_e2e_targets_validates_declarative_suite_tomls(tmp_path):
    suites_dir = tmp_path / "tests" / "e2e_tests" / "suites"
    suites_dir.mkdir(parents=True)
    (suites_dir / "broken.toml").write_text('robot = "broken"\n', encoding="utf-8")
    findings = robots_check["check_e2e_targets"](tmp_path)
    assert len(findings) == 2
    assert all(f.rule == "robots/e2e-targets" for f in findings)
