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
        "execute_action",
        "read_image",
    ):
        assert tool_name not in names
    assert "ProcessDaemon" in names
    assert "try_spawn_server" in names


def test_build_docs_falls_back_to_stderr_when_sphinx_fails_without_warning_prefix(
    tmp_path, monkeypatch
):
    captured_cmd: list[str] = []

    def fake_run(cmd, *a, **kw):
        captured_cmd.extend(cmd)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=2,
            stdout="",
            stderr="Extension error (sphinx_sitemap): boom\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    records = build_docs["build"](Path(sys.executable), tmp_path, tmp_path / "out")
    assert len(records) == 1
    assert "sphinx-build exited 2" in records[0]
    assert "Extension error (sphinx_sitemap): boom" in records[0]
    for flag in ("-E", "-N", "-W", "--keep-going"):
        assert flag in captured_cmd


def test_check_writing_allows_robot_config_and_eval_data_paths(tmp_path):
    guide = tmp_path / "franka.rst"
    guide.write_text(
        "Franka\n======\n\n"
        "Edit ``robots/franka/config/example.yaml``, ``robots/yam/config/example.yml``, "
        "and ``robots/robocasa/eval/target50.json``.\n\n"
        "Do not open ``robots/franka/env_server.py``.\n",
        encoding="utf-8",
    )
    findings = writing["check_page"](
        guide,
        "docs/source-en/rst_source/real_world_robots/franka.rst",
        frozenset(),
        False,
    )
    code_hits = [f for f in findings if f.rule == "docs/code-name"]
    assert len(code_hits) == 1
    assert "robots/franka/env_server.py" in code_hits[0].message


def test_docs_check_skips_docutils_and_repo_scans_for_non_rst_changes(
    tmp_path, monkeypatch
):
    Change = route["Change"]
    Selection = route["Selection"]
    selection = Selection("main", (Change("M", "README.md"),), False)

    def fail(*args, **kwargs):
        raise AssertionError("must not run when no .rst page changed")

    monkeypatch.setitem(
        docs_check["main"].__globals__["route"].__dict__,
        "select",
        lambda area, args: selection,
    )
    monkeypatch.setitem(docs_check["main"].__globals__, "_ensure_docutils", fail)
    monkeypatch.setitem(
        docs_check["main"].__globals__["check_doc_symbols"].__dict__,
        "vocabulary",
        fail,
    )
    assert docs_check["main"]([]) == 0


def test_docs_check_help_does_not_provision_docs_venv(monkeypatch):
    def fail_venv(*args, **kwargs):
        raise AssertionError("ensure_venv must not run on --help")

    monkeypatch.setitem(docs_check["main"].__globals__, "ensure_venv", fail_venv)
    with pytest.raises(SystemExit) as exc:
        docs_check["main"](["--help"])
    assert exc.value.code == 0


def test_check_writing_scans_toolkits_json_keys_sphinx_roles_and_allows_cli_entries(
    tmp_path,
):
    assert writing["page_type"]("docs/source-en/rst_source/toolkits/mcp.rst") == "guide"

    # Synthetic tree verifies toolkits/ Python definitions and *.json keys.
    (tmp_path / "toolkits").mkdir()
    (tmp_path / "toolkits" / "internal_helper.py").write_text(
        "class InternalGeometryHelper:\n    pass\n\ndef json_config_key():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "robots" / "demo").mkdir(parents=True)
    (tmp_path / "robots" / "demo" / "spec.json").write_text(
        '{"nested": {"json_config_key": 16}}\n',
        encoding="utf-8",
    )
    names = writing["code_names"](tmp_path)
    assert "InternalGeometryHelper" in names
    assert "json_config_key" not in names

    repo_names = writing["code_names"](REPO_ROOT)
    assert "RPentEfeAgent" not in repo_names
    assert "history_length" not in repo_names

    page = tmp_path / "mcp.rst"
    page.write_text(
        "MCP Tools\n=========\n\n"
        "See ``toolkits/back_project.py``, ``InternalGeometryHelper``, "
        ":class:`rpent.tools.toolkit.Toolkit`, "
        ":meth:`~robots.libero.toolkit.get_toolkit`, "
        ":mod:`Label <toolkits.segment>`, "
        "and ``RPentEfeAgent`` with ``history_length``.\n",
        encoding="utf-8",
    )
    findings = writing["check_page"](
        page, "docs/source-en/rst_source/toolkits/mcp.rst", names, False
    )
    hits = [f.message for f in findings if f.rule == "docs/code-name"]
    assert any("toolkits/back_project.py" in m for m in hits)
    assert any("InternalGeometryHelper" in m for m in hits)
    assert any("rpent.tools.toolkit.Toolkit" in m for m in hits)
    assert any("robots.libero.toolkit.get_toolkit" in m for m in hits)
    assert any("toolkits.segment" in m for m in hits)
    assert not any("RPentEfeAgent" in m or "history_length" in m for m in hits)


def test_check_doc_symbols_uses_cwd_git_toplevel(tmp_path, monkeypatch):
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "rpent").mkdir()
    (tmp_path / "rpent" / "demo.py").write_text(
        "class WorktreeOnlySymbol:\n    pass\n", encoding="utf-8"
    )
    dev_dir = tmp_path / "docs" / "source-en" / "rst_source" / "development"
    dev_dir.mkdir(parents=True)
    (dev_dir / "demo.rst").write_text(
        "Demo\n====\n\nUse ``WorktreeOnlySymbol``.\n", encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    assert doc_symbols["main"]([]) == 0

    (dev_dir / "demo.rst").write_text(
        "Demo\n====\n\nUse ``MissingWorktreeSymbol``.\n", encoding="utf-8"
    )
    assert doc_symbols["main"]([]) == 1
