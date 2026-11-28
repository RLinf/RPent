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

"""Unit tests for component resolution, Python selection, and requirements rendering."""

from pathlib import Path

import pytest

from rpent.install import installer
from rpent.install.errors import InstallConfigError, InstallError
from rpent.install.resolve import (
    choose_python,
    index_options,
    manylinux_baseline,
    python_errors,
    render_requirements,
    requirements_sha256,
    resolve,
    resolve_resolution,
)


def test_resolve_closure_and_render_requirements(make_component, make_catalog):
    sim = make_component("sim-s", dependencies=["mujoco==3.3.0"])
    robot = make_component(
        "robot-r", planner=True, requires=["sim-s"], dependencies=["scipy"]
    )
    catalog = make_catalog([sim, robot])

    res = resolve(
        ["robot-r", "sim-s"],
        catalog,
        core_dependencies=("numpy",),
        planner_dependencies=("httpx",),
    )
    assert res.ok
    assert set(res.components) == {"robot-r", "sim-s"}
    rendered = render_requirements(
        res, core_dependencies=("numpy",), planner_dependencies=("httpx",)
    )
    assert "# Rerun: rpent-install install robot-r sim-s" in rendered
    assert "numpy" in rendered and "httpx" in rendered and "mujoco==3.3.0" in rendered


def test_resolve_detects_conflicts_and_index_disagreement(make_component, make_catalog):
    a = make_component(
        "sim-a",
        env_conflicts=["sim-b"],
        dependencies=["torch==2.6.0"],
        index_url="https://idx-a",
        extra_index_urls=["https://extra-1"],
    )
    b = make_component(
        "sim-b",
        dependencies=["torch==2.8.0"],
        index_url="https://idx-b",
        extra_index_urls=["https://extra-2"],
    )
    res = resolve(["sim-a", "sim-b"], make_catalog([a, b]))
    assert not res.ok
    joined = "\n".join(res.errors)
    assert "env_conflicts" in joined
    assert "conflicting pins for 'torch'" in joined
    assert "selects index" in joined
    assert res.extra_index_urls == ("https://extra-1", "https://extra-2")


def test_choose_python_and_python_errors_handle_patch_bounds_and_unknown(
    make_component,
):
    comp = make_component("robot-a", python=">=3.11.4,<3.12")
    assert choose_python([comp]) == "3.11"
    assert python_errors("3.11", [comp]) == []
    assert python_errors("3.11.5", [comp]) == []
    assert python_errors("3.11.2", [comp])
    assert python_errors("unknown", [comp])
    assert python_errors("", [comp])


def test_requirements_sha256_ignores_override_comments(
    make_component, make_catalog, tmp_path
):
    override = tmp_path / "override.txt"
    override.write_text("# comment 1\nnumpy<2\n", encoding="utf-8")
    comp = make_component("sim-a", overrides=[override])
    res = resolve(["sim-a"], make_catalog([comp]))
    h1 = requirements_sha256(res, root=tmp_path)

    override.write_text("# different comment\nnumpy<2\n", encoding="utf-8")
    h2 = requirements_sha256(res, root=tmp_path)
    assert h1 == h2

    override.write_text("numpy<1.26\n", encoding="utf-8")
    h3 = requirements_sha256(res, root=tmp_path)
    assert h3 != h1


def test_manylinux_baseline_selects_newest(make_component):
    a = make_component("sim-a", manylinux="2_31")
    b = make_component("sim-b", manylinux="2_35")
    assert manylinux_baseline([a, b]) == "2_35"


def test_resolve_resolution_invokes_uv_pip_compile_and_propagates_errors(
    make_component, make_catalog, tmp_path, monkeypatch
):
    override = tmp_path / "override.txt"
    override.write_text("numpy<2\n", encoding="utf-8")
    comp = make_component(
        "sim-a",
        manylinux="2_35",
        dependencies=["scipy==1.14.0"],
        overrides=[override],
        index_url="https://idx.example/simple",
        extra_index_urls=["https://extra.example/simple"],
        index_strategy="unsafe-best-match",
    )
    res = resolve(["sim-a"], make_catalog([comp]), core_dependencies=("numpy",))
    assert res.ok

    seen: dict[str, object] = {}

    def fake_run(cmd, *, capture=False, cwd=None, **kwargs):
        seen["cmd"] = list(cmd)
        seen["capture"] = capture
        seen["cwd"] = cwd
        seen["input_text"] = Path(cmd[3]).read_text(encoding="utf-8")
        out_index = cmd.index("--output-file") + 1
        Path(cmd[out_index]).write_text(
            "numpy==1.26.4\nscipy==1.14.0\n", encoding="utf-8"
        )
        return ""

    monkeypatch.setattr(installer, "run_command", fake_run)
    compiled = resolve_resolution(
        res,
        root=tmp_path,
        uv="/usr/bin/uv",
        python="3.11",
        core_dependencies=("numpy",),
    )

    assert compiled == "numpy==1.26.4\nscipy==1.14.0\n"
    assert seen["capture"] is True
    assert seen["cwd"] == tmp_path
    cmd = seen["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:3] == ["/usr/bin/uv", "pip", "compile"]
    assert cmd[cmd.index("--python-version") + 1] == "3.11"
    assert cmd[cmd.index("--python-platform") + 1] == "x86_64-manylinux_2_35"
    assert "--no-header" in cmd and "--no-annotate" in cmd
    assert cmd[cmd.index("--index-url") + 1] == "https://idx.example/simple"
    assert cmd[cmd.index("--extra-index-url") + 1] == "https://extra.example/simple"
    assert cmd[cmd.index("--index-strategy") + 1] == "unsafe-best-match"
    assert cmd[cmd.index("--override") + 1] == str(override)
    assert "numpy" in str(seen["input_text"])
    assert "scipy==1.14.0" in str(seen["input_text"])

    def fail_run(*args, **kwargs):
        raise InstallError("uv pip compile failed")

    monkeypatch.setattr(installer, "run_command", fail_run)
    with pytest.raises(InstallError, match="uv pip compile failed"):
        resolve_resolution(res, root=tmp_path, uv="/usr/bin/uv", python="3.11")


def test_resolve_detects_override_conflicts_and_ignores_markers_or_disjoint(
    make_component, make_catalog, tmp_path
):
    ov_a = tmp_path / "a.txt"
    ov_b = tmp_path / "b.txt"
    ov_a.write_text(
        "# comment\n\nNumPy<2\npkg-only-a==1.0\ntorch<2.5 ; sys_platform == 'win32'\n",
        encoding="utf-8",
    )
    ov_b.write_text(
        "numpy<1.26\npkg-only-b==2.0\ntorch<2.6 ; sys_platform == 'win32'\n",
        encoding="utf-8",
    )
    a = make_component("sim-a", overrides=[ov_a])
    b = make_component("sim-b", overrides=[ov_b])

    res = resolve(["sim-a", "sim-b"], make_catalog([a, b]))
    assert not res.ok
    assert res.errors == [
        "components 'sim-a', 'sim-b' both override 'numpy' in one environment"
    ]

    ov_b.write_text("pkg-only-b==2.0\n", encoding="utf-8")
    disjoint = resolve(["sim-a", "sim-b"], make_catalog([a, b]))
    assert disjoint.ok


def test_resolve_detects_requires_env_conflicts_contradiction(
    make_component, make_catalog
):
    sim = make_component("sim-s")
    robot = make_component("robot-r", requires=["sim-s"], env_conflicts=["sim-s"])
    res = resolve(["robot-r"], make_catalog([robot, sim]))
    assert not res.ok
    assert any(
        "requires 'sim-s' but the two declare env_conflicts" in err
        for err in res.errors
    )

    sim_back = make_component("sim-s", env_conflicts=["robot-r"])
    robot_clean = make_component("robot-r", requires=["sim-s"])
    res_back = resolve(["robot-r"], make_catalog([robot_clean, sim_back]))
    assert not res_back.ok
    assert any(
        "requires 'sim-s' but the two declare env_conflicts" in err
        for err in res_back.errors
    )


def test_resolve_detects_index_strategy_conflicts_and_builds_index_options(
    make_component, make_catalog
):
    a = make_component("sim-a", index_strategy="first-index")
    b = make_component("sim-b", index_strategy="unsafe-best-match")
    res = resolve(["sim-a", "sim-b"], make_catalog([a, b]))
    assert not res.ok
    assert any("selects index strategy" in err for err in res.errors)

    c = make_component(
        "sim-c",
        index_url="https://idx.example/simple",
        extra_index_urls=["https://extra.example/simple"],
        index_strategy="unsafe-first-match",
    )
    d = make_component("sim-d", index_strategy="unsafe-first-match")
    res_ok = resolve(["sim-c", "sim-d"], make_catalog([c, d]))
    assert res_ok.ok
    assert res_ok.index_strategy == "unsafe-first-match"
    assert index_options(res_ok) == [
        "--index-url",
        "https://idx.example/simple",
        "--extra-index-url",
        "https://extra.example/simple",
        "--index-strategy",
        "unsafe-first-match",
    ]
    assert index_options(
        res_ok, index_url_override="https://mirror.example/simple"
    ) == [
        "--index-url",
        "https://mirror.example/simple",
        "--extra-index-url",
        "https://extra.example/simple",
        "--index-strategy",
        "unsafe-first-match",
    ]


def test_resolve_reports_empty_unknown_missing_planner_url_pins_and_python_conflict(
    make_component, make_catalog
):
    empty = resolve([], make_catalog([]))
    assert not empty.ok and any("at least one component" in e for e in empty.errors)

    sim = make_component("sim-a")
    unknown = resolve(["robot-missing"], make_catalog([sim]))
    assert not unknown.ok and any(
        "unknown component 'robot-missing'" in e for e in unknown.errors
    )

    needs_planner = make_component("robot-p", planner=True)
    no_planner = resolve(["robot-p"], make_catalog([needs_planner]))
    assert not no_planner.ok and any(
        "[project.optional-dependencies].planner is missing" in e
        for e in no_planner.errors
    )

    url_a = make_component(
        "sim-a", dependencies=["pkg @ https://example.com/pkg-1.whl"]
    )
    url_b = make_component(
        "sim-b", dependencies=["pkg @ https://example.com/pkg-2.whl"]
    )
    url_res = resolve(["sim-a", "sim-b"], make_catalog([url_a, url_b]))
    assert not url_res.ok and any(
        "conflicting pins for 'pkg'" in e for e in url_res.errors
    )

    py312 = make_component("sim-new", python=">=3.12")
    py310 = make_component("sim-old", python="<3.11")
    py_res = resolve(["sim-new", "sim-old"], make_catalog([py312, py310]))
    assert not py_res.ok and any("requires Python" in e for e in py_res.errors)


def test_unreadable_or_non_utf8_override_raises_install_config_error(
    make_component, make_catalog, tmp_path
):
    bad_utf8 = tmp_path / "bad.txt"
    bad_utf8.write_bytes(b"\xff\xfeinvalid\n")
    comp_bad = make_component("sim-bad", overrides=[bad_utf8])
    catalog_bad = make_catalog([comp_bad])

    with pytest.raises(InstallConfigError, match="cannot read override file"):
        resolve(["sim-bad"], catalog_bad)

    missing = tmp_path / "missing.txt"
    comp_missing = make_component("sim-missing", overrides=[missing])
    with pytest.raises(InstallConfigError, match="cannot read override file"):
        resolve(["sim-missing"], make_catalog([comp_missing]))

    valid = tmp_path / "ok.txt"
    valid.write_text("numpy<2\n", encoding="utf-8")
    comp_ok = make_component("sim-ok", overrides=[valid])
    res = resolve(["sim-ok"], make_catalog([comp_ok]))
    assert res.ok
    valid.write_bytes(b"\xff\xfeinvalid\n")
    with pytest.raises(InstallConfigError, match="cannot read override file"):
        requirements_sha256(res, root=tmp_path)
    valid.unlink()
    with pytest.raises(InstallConfigError, match="cannot read override file"):
        requirements_sha256(res, root=tmp_path)
