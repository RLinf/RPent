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
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
build_docs = runpy.run_path(str(REPO_ROOT / ".agents/skills/docs/build_docs.py"))
docs_check = runpy.run_path(str(REPO_ROOT / ".agents/skills/docs/check.py"))
doc_symbols = runpy.run_path(
    str(REPO_ROOT / ".agents/skills/docs/check_doc_symbols.py")
)
writing = runpy.run_path(str(REPO_ROOT / ".agents/skills/docs/check_writing.py"))
route = runpy.run_path(str(REPO_ROOT / ".agents/skills/route.py"))


def test_check_doc_symbols_passes_on_repository_development_docs():
    assert doc_symbols["main"]([]) == 0


def test_check_parity_and_toctrees(tmp_path):
    Change = route["Change"]
    en_dir = tmp_path / "docs" / "source-en" / "rst_source" / "simulators"
    zh_dir = tmp_path / "docs" / "source-zh" / "rst_source" / "simulators"
    en_dir.mkdir(parents=True)
    zh_dir.mkdir(parents=True)

    (en_dir / "index.rst").write_text(
        "Simulators\n==========\n\n.. toctree::\n   :hidden:\n\n   libero\n   robocasa\n",
        encoding="utf-8",
    )
    (zh_dir / "index.rst").write_text(
        "仿真环境\n========\n\n.. toctree::\n   :hidden:\n\n   libero\n",
        encoding="utf-8",
    )
    (en_dir / "only_en.rst").write_text("Only EN\n=======\n", encoding="utf-8")

    parity = docs_check["check_parity"](
        [
            Change("A", "docs/source-en/rst_source/simulators/only_en.rst"),
            Change("M", "docs/source-en/rst_source/simulators/index.rst"),
        ],
        tmp_path,
    )
    assert any(f.severity == "major" and "does not exist" in f.message for f in parity)
    assert any(f.severity == "minor" and "changed while" in f.message for f in parity)

    toctrees = docs_check["check_toctrees"](
        ["docs/source-en/rst_source/simulators/index.rst"], tmp_path
    )
    assert len(toctrees) == 1
    assert "only in EN: robocasa" in toctrees[0].message


def test_check_hard_wrap_allows_escaped_newline_and_flags_bare_cjk_wrap(tmp_path):
    escaped = tmp_path / "escaped.rst"
    escaped.write_text(
        "标题\n====\n\n评测模式读取已有记忆，\\\n按环境规定判断任务结果。\n",
        encoding="utf-8",
    )
    assert docs_check["check_hard_wrap"](escaped, "docs/source-zh/escaped.rst") == []

    bare = tmp_path / "bare.rst"
    bare.write_text(
        "标题\n====\n\n评测模式读取已有记忆，\n按环境规定判断任务结果。\n",
        encoding="utf-8",
    )
    findings = docs_check["check_hard_wrap"](bare, "docs/source-zh/bare.rst")
    assert len(findings) == 1
    assert findings[0].rule == "docs/zh-hard-wrap"
    assert findings[0].line == 5


def test_check_writing_catches_internal_code_names_missing_sections_and_style(tmp_path):
    sim = tmp_path / "sim.rst"
    sim.write_text(
        "NewSim\n======\n\n"
        "This simply calls ``rpent/cli/main.py`` and ``ProcessDaemon``.\n",
        encoding="utf-8",
    )
    names = frozenset({"ProcessDaemon"})
    findings = writing["check_page"](
        sim, "docs/source-en/rst_source/simulators/newsim.rst", names, True
    )
    rules = {f.rule for f in findings}
    assert "docs/code-name" in rules
    assert "docs/structure" in rules
    assert "docs/en-style" in rules

    zh_guide = tmp_path / "zh.rst"
    zh_guide.write_text(
        "指南\n====\n\n值得注意的是，您需要对模型进行评测，这不是A而是B。\n",
        encoding="utf-8",
    )
    zh_findings = writing["check_page"](
        zh_guide, "docs/source-zh/rst_source/guides/demo.rst", names, False
    )
    assert any(f.rule == "docs/zh-style" for f in zh_findings)


def test_check_page_flags_hardcoded_rpent_rtd_links(tmp_path):
    page = tmp_path / "guide.rst"
    page.write_text(
        "Guide\n=====\n\nSee https://rpent.readthedocs.io/en/latest/rst_source/guides/cli.html.\n",
        encoding="utf-8",
    )
    findings = docs_check["check_page"](
        page, "docs/source-en/rst_source/guides/demo.rst", set()
    )
    assert any(f.rule == "docs/rtd-link" and f.severity == "major" for f in findings)


def test_code_names_excludes_registered_user_facing_tools():
    names = writing["code_names"](REPO_ROOT)
    for tool_name in (
        "view_env_state",
        "view_camera_meta",
        "move_delta",
        "rotate_delta",
        "open_gripper",
        "close_gripper",
        "move_pose",
        "rotate_wrist",
        "sample_world_xyz",
        "query_world_map",
        "vla_grasp",
        "efe_spec",
        "efe_observe",
        "efe_status",
        "efe_recover",
    ):
        assert tool_name not in names
    assert "ProcessDaemon" in names
    assert "try_spawn_server" in names


def test_build_docs_falls_back_to_stderr_when_sphinx_fails_without_warning_prefix(
    tmp_path, monkeypatch
):
    fake_result = subprocess.CompletedProcess(
        args=["sphinx"],
        returncode=2,
        stdout="",
        stderr="Extension error (sphinx_sitemap): boom\n",
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)
    records = build_docs["build"](Path(sys.executable), tmp_path, tmp_path / "out")
    assert len(records) == 1
    assert "sphinx-build exited 2" in records[0]
    assert "Extension error (sphinx_sitemap): boom" in records[0]


def test_docs_check_help_does_not_provision_docs_venv(monkeypatch):
    def fail_venv(*args, **kwargs):
        raise AssertionError("ensure_venv must not run on --help")

    monkeypatch.setitem(docs_check["main"].__globals__, "ensure_venv", fail_venv)
    with pytest.raises(SystemExit) as exc:
        docs_check["main"](["--help"])
    assert exc.value.code == 0
