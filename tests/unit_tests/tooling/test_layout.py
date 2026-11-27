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

"""Unit tests for ``tests.tooling.layout``."""

from pathlib import Path

from tests.tooling import layout


def _write(root: Path, relative: str, text: str = "") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_real_checkout_passes_layout():
    assert layout.main(["--root", str(layout.REPO_ROOT)]) == 0


def test_same_named_sim_cannot_import_robot_layer(tmp_path):
    _write(tmp_path, "robots/__init__.py")
    _write(tmp_path, "robots/alpha/__init__.py", "__all__ = []\n")
    _write(
        tmp_path,
        "robots/alpha/component.toml",
        'schema = 2\nname = "robot-alpha"\nprobe = "robots.alpha"\n',
    )
    _write(tmp_path, "components/__init__.py")
    _write(tmp_path, "components/sim/__init__.py")
    _write(tmp_path, "components/sim/alpha/__init__.py")
    _write(
        tmp_path,
        "components/sim/alpha/component.toml",
        'schema = 2\nname = "sim-alpha"\nprobe = "components.sim.alpha"\n',
    )
    _write(
        tmp_path,
        "components/sim/alpha/env_server.py",
        "from robots.alpha import robot_spec\n",
    )

    findings = layout.check_imports(tmp_path)
    assert any(
        "sim 'alpha' must not import robot 'alpha'" in f.message for f in findings
    )


def test_check_imports_ignores_venv_and_blocks_core_to_component(tmp_path):
    _write(tmp_path, ".venv-demo/lib/bad.py", "import robots.alpha\n")
    _write(tmp_path, "rpent/bad.py", "import components.models.openpi\n")

    findings = layout.check_imports(tmp_path)
    assert len(findings) == 1
    assert "core must not import components.models" in findings[0].message


def test_lazy_entries_reject_type_checking_else_and_dynamic_model_import(tmp_path):
    _write(
        tmp_path,
        "robots/alpha/__init__.py",
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    import os\n"
        "else:\n"
        "    import torch\n",
    )
    _write(
        tmp_path,
        "components/models/demo/__init__.py",
        "import importlib\nimportlib.import_module('torch')\n",
    )

    findings = layout.check_lazy_entries(tmp_path)
    assert any("robot __init__.py must stay lazy" in f.message for f in findings)
    assert any("third-party 'torch'" in f.message for f in findings)


def test_check_components_flags_missing_init_manifests_and_invalid_toml(tmp_path):
    _write(
        tmp_path,
        "robots/alpha/component.toml",
        'schema = 2\nname = "robot-alpha"\nprobe = "robots.alpha"\n',
    )
    _write(tmp_path, "robots/orphan/robot_spec.py", "")
    _write(tmp_path, "robots/.hidden/skip.py", "")
    _write(tmp_path, "components/__init__.py", "")
    _write(tmp_path, "components/models/__init__.py", "")
    _write(tmp_path, "components/models/with_code/server.py", "")
    _write(tmp_path, "components/models/assets_only/weights.bin", "")

    findings = layout.check_components(tmp_path)
    messages = [f.message for f in findings]
    assert "package group robots/ needs __init__.py" in messages
    assert any(
        f.path == tmp_path / "robots/orphan/component.toml"
        and "robot directory has no component.toml" in f.message
        for f in findings
    )
    assert any(
        f.path == tmp_path / "components/models/with_code/component.toml"
        and "model directory has no component.toml" in f.message
        for f in findings
    )
    assert not any("assets_only" in str(f.path) for f in findings)
    assert not any(".hidden" in str(f.path) for f in findings)

    _write(tmp_path, "robots/__init__.py", "")
    _write(
        tmp_path,
        "components/infra/broken/component.toml",
        'schema = 99\nname = "infra-broken"\nprobe = "json"\n',
    )
    invalid = layout.check_components(tmp_path)
    assert any("unsupported schema" in f.message for f in invalid)


def test_check_imports_enforces_declared_cross_robot_imports(tmp_path):
    _write(
        tmp_path,
        "robots/a/component.toml",
        'schema = 2\nname = "robot-a"\nprobe = "robots.a"\n',
    )
    _write(
        tmp_path,
        "robots/b/component.toml",
        'schema = 2\nname = "robot-b"\nprobe = "robots.b"\n',
    )
    _write(tmp_path, "robots/b/helper.py", "")
    _write(
        tmp_path,
        "robots/a/tool.py",
        "import importlib\n"
        "import robots.b\n"
        "from ..b import helper\n"
        "importlib.import_module('robots.b.helper')\n",
    )

    findings = layout.check_imports(tmp_path)
    undeclared = [
        f
        for f in findings
        if "robot 'a' must not import robot 'b' without declaring it" in f.message
    ]
    assert len(undeclared) >= 3

    _write(
        tmp_path,
        "robots/a/component.toml",
        'schema = 2\nname = "robot-a"\nprobe = "robots.a"\n'
        'requires = ["robot-b"]\nimports = ["robot-b"]\n',
    )
    assert layout.check_imports(tmp_path) == []


def test_check_imports_enforces_model_tool_and_core_layer_boundaries(tmp_path):
    _write(
        tmp_path,
        "components/models/m1/server.py",
        "import robots.alpha\n"
        "import components.models.m2\n"
        "import components.tools.t1\n"
        "import components.sim.s1\n",
    )
    _write(
        tmp_path,
        "components/tools/t1/cli.py",
        "import robots.alpha\n"
        "import components.sim.s1\n"
        "import components.tools.t2\n"
        "import components.models.m1\n",
    )
    _write(
        tmp_path,
        "rpent/other.py",
        "import components.sim.s1\nimport components.tools.t1\nimport robots.alpha\n",
    )
    _write(
        tmp_path,
        "rpent/cli/flywheel.py",
        "import components.tools.flywheel\n",
    )

    messages = {f.message for f in layout.check_imports(tmp_path)}
    assert {
        "model 'm1' must not import robot 'alpha'",
        "model 'm1' must not import model 'm2'",
        "model 'm1' must not import tool 't1'",
        "tool 't1' must not import robot 'alpha'",
        "tool 't1' must not import sim 's1'",
        "tool 't1' must not import tool 't2'",
        "core must not import components.sim",
        "core must not import components.tools",
        "core must not import concrete robot 'alpha'",
    } == messages


def test_lazy_entries_reject_model_self_import_and_allow_lazy_patterns(tmp_path):
    _write(tmp_path, "components/models/demo/server.py", "")
    _write(
        tmp_path,
        "components/models/demo/__init__.py",
        "from . import server\nfrom components.models.demo.server import serve\n",
    )
    _write(
        tmp_path,
        "components/models/clean/__init__.py",
        '"""Clean model package."""\n'
        "from __future__ import annotations\n"
        "import os\n"
        "import numpy\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .server import Server\n"
        "DEFAULT_PORT = 8000\n"
        '__all__ = ["Server"]\n'
        "def __getattr__(name: str):\n"
        "    raise AttributeError(name)\n",
    )
    _write(
        tmp_path,
        "robots/clean/__init__.py",
        '"""Clean robot package."""\n'
        "from __future__ import annotations\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .robot_spec import Spec\n"
        '__all__ = ["Spec"]\n'
        "def __getattr__(name: str):\n"
        "    raise AttributeError(name)\n",
    )

    findings = layout.check_lazy_entries(tmp_path)
    assert len(findings) == 2
    assert all(
        "model __init__.py must not eagerly import its own runtime module" in f.message
        for f in findings
    )


def test_layout_main_returns_nonzero_on_violation(tmp_path, capsys):
    _write(tmp_path, "rpent/bad.py", "import robots.alpha\n")

    assert layout.main(["--root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "core must not import concrete robot 'alpha'" in captured.out
    assert "1 layout violation(s)" in captured.err
