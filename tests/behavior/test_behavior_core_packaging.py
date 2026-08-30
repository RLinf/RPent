from __future__ import annotations

import inspect
import tomllib
from pathlib import Path

from rpent.robots import enumerate_robots, get_robot_spec
from rpent.robots.robot_spec import RobotSpec, RunConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
BEHAVIOR_INIT = REPO_ROOT / "robots" / "behavior" / "__init__.py"


def test_core_robot_spec_contract_stays_robot_agnostic() -> None:
    assert tuple(RobotSpec.__dataclass_fields__) == (
        "name",
        "prompts",
        "add_cli_args",
        "parse_config",
        "init_runtime",
        "dashboard",
        "resources_repo_id",
    )
    assert tuple(RunConfig.__dataclass_fields__) == (
        "recipe_tag",
        "output_dir",
        "prompt_vars",
        "task_desc",
    )

    signature = inspect.signature(RobotSpec)
    assert tuple(signature.parameters) == tuple(RobotSpec.__dataclass_fields__)
    for field in RobotSpec.__dataclass_fields__:
        lowered = field.lower()
        assert "behavior" not in lowered
        assert "task_success" not in lowered
        assert "tool" not in lowered


def test_core_dashboard_cli_stays_robot_name_agnostic() -> None:
    source = (REPO_ROOT / "rpent" / "cli" / "dashboard.py").read_text("utf-8")

    assert "robot_spec.name == \"behavior\"" not in source
    assert "robots.behavior.dashboard" not in source


def test_behavior_is_enumerated_only_after_robot_spec_entrypoint_lands() -> None:
    if not BEHAVIOR_INIT.is_file():
        assert "behavior" not in enumerate_robots()
        return

    assert "behavior" in enumerate_robots()
    spec = get_robot_spec("behavior")
    assert isinstance(spec, RobotSpec)
    assert spec.name == "behavior"
    assert callable(spec.add_cli_args)
    assert callable(spec.parse_config)
    assert callable(spec.init_runtime)


def test_behavior_packaging_is_optional_and_does_not_expand_package_discovery() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))

    optional = pyproject["project"]["optional-dependencies"]
    assert "behavior" in optional
    assert optional["behavior"], "the behavior extra must declare runtime deps"
    assert "full" in optional
    assert not any("behavior" in dep.lower() for dep in optional["full"])

    packages = pyproject["tool"]["setuptools"]["packages"]["find"]
    assert packages["where"] == ["."]
    assert packages["include"] == ["rpent*"]
    assert "robots*" not in packages.get("include", [])


def test_behavior_docs_have_bilingual_entrypoints() -> None:
    expected = (
        REPO_ROOT / "docs" / "source-en" / "rst_source" / "usage" / "behavior.rst",
        REPO_ROOT / "docs" / "source-zh" / "rst_source" / "usage" / "behavior.rst",
    )
    for path in expected:
        assert path.is_file(), path
        text = path.read_text("utf-8").lower()
        assert "behavior" in text
        assert "memory" in text

    for index in (
        REPO_ROOT / "docs" / "source-en" / "index.rst",
        REPO_ROOT / "docs" / "source-zh" / "index.rst",
    ):
        assert "usage/behavior" in index.read_text("utf-8")
