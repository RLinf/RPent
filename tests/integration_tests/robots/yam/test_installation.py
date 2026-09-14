# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Run from outside the checkout against a non-editable RPent wheel.

python -m pytest --import-mode=importlib -o pythonpath='' /path/to/this/file.py
The environment needs only the core wheel and pytest, not hardware dependencies.
"""

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path


def test_wheel_contains_robot_and_agent_guide():
    import robots.yam

    installed = Path(robots.yam.__file__).resolve()
    assert installed.is_relative_to(Path(sys.prefix).resolve())
    assert installed.with_name("guides").joinpath("GUIDE_RPENT.md").is_file()
    files = {str(p) for p in importlib.metadata.files("rpent")}
    assert "robots/yam/robot_spec.py" in files
    assert "robots/yam/config.example.json" in files
    assert all(
        p == "robots/__init__.py" or p.startswith("robots/yam/")
        for p in files
        if p.startswith("robots/")
    )


def test_help_without_a_checkout_or_hardware_imports(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("RPENT_REPO_ROOT", None)
    result = subprocess.run(
        [sys.executable, "-m", "rpent.cli.main", "--robot", "yam", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--task-id {103,104}" in result.stdout
    assert "--explore-attempts-per-session" in result.stdout
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from rpent.robots import get_robot_spec; "
            "import sys; s = get_robot_spec('yam'); "
            "assert s.supports_exploration and s.run_diagnostic is not None; "
            "assert not ({'torch', 'mujoco', 'pyrealsense2', 'rlinf'} & set(sys.modules))",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert probe.returncode == 0, probe.stderr
