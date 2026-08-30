from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from robots.behavior import env_client, env_server, runtime
from robots.behavior.runtime import _behavior_python_path
from robots.behavior.toolkit import BehaviorToolkit
from robots.behavior.tools import BehaviorPrimitives
from rpent.dashboard.events import ToolResultEvent
from rpent.tools.toolkit import readonly


def test_env_server_defaults_to_bundled_official_backend(monkeypatch) -> None:
    monkeypatch.delenv("RPENT_BEHAVIOR_ENV_BACKEND_FACTORY", raising=False)

    factory = env_server._backend_factory_from_env()

    assert factory.__module__ == "robots.behavior.official_env_backend"
    assert factory.__name__ == "create_backend"


def test_env_rpc_preserves_png_bytes() -> None:
    payload = b"\x89PNG\r\n\x1a\nbehavior"

    encoded = env_server._jsonable({"_frames_bytes": {"head": payload}})
    decoded = env_client._decode_bytes(encoded)

    assert decoded == {"_frames_bytes": {"head": payload}}


def test_env_client_dashboard_execute_discard_forward_plan_id() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class Client:
        def call(self, method, *, args=(), kwargs=None, timeout_s=None):
            del args, timeout_s
            calls.append((method, dict(kwargs or {})))
            if method == "env.get_env_meta":
                return {"runtime": "behavior_env"}
            return {"status": "ok"}

    client = env_client.BehaviorEnvClient(
        Client(),
        expected_meta={"runtime": "behavior_env"},
    )

    client.dashboard_execute_prepared_command(command_id="cmd_a", plan_id="plan_a")
    client.dashboard_discard_prepared_command(command_id="cmd_b", plan_id="plan_b")

    assert calls[-2:] == [
        (
            "env.dashboard_execute_prepared_command",
            {"command_id": "cmd_a", "plan_id": "plan_a"},
        ),
        (
            "env.dashboard_discard_prepared_command",
            {"command_id": "cmd_b", "plan_id": "plan_b"},
        ),
    ]


def test_behavior_python_path_preserves_virtualenv_symlink(tmp_path) -> None:
    system_python = tmp_path / "system-python"
    system_python.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(system_python)

    selected = _behavior_python_path(venv_python)

    assert selected == venv_python.absolute()
    assert selected != Path(venv_python).resolve()


def test_behavior_component_cuda_flags_are_distinct_single_device_options(tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=tmp_path)
    runtime.add_cli_args(parser, use_dashboard=False)

    args = parser.parse_args(
        [
            "--task-name",
            "picking_up_trash",
            "--public-seed",
            "3",
            "--behavior-mode",
            "explore",
            "--behavior-env-cuda-device",
            "2",
            "--behavior-model-cuda-device",
            "7",
        ]
    )
    runtime.parse_config(args)

    assert args.behavior_env_cuda_device == "2"
    assert args.behavior_model_cuda_device == "7"

    bad = parser.parse_args(
        [
            "--task-name",
            "picking_up_trash",
            "--public-seed",
            "3",
            "--behavior-mode",
            "explore",
            "--behavior-model-cuda-device",
            "2,7",
        ]
    )
    with pytest.raises(ValueError, match="single physical GPU ordinal"):
        runtime.parse_config(bad)


def test_behavior_runtime_routes_env_and_model_cuda_to_separate_children(
    tmp_path,
    monkeypatch,
) -> None:
    behavior_python = tmp_path / "python"
    behavior_python.write_text("", encoding="utf-8")
    captures: list[dict[str, object]] = []

    class CapturingDaemon:
        def __init__(self, *, name, cmd, env_overrides, log_path):
            self.name = name
            self.cmd = list(cmd)
            self.env_overrides = dict(env_overrides)
            self.log_path = log_path
            captures.append(
                {
                    "name": self.name,
                    "cmd": self.cmd,
                    "env_overrides": self.env_overrides,
                    "log_path": self.log_path,
                }
            )

        def start(self):
            return None

    ports = iter((45001, 45002, 45003))
    monkeypatch.setattr(runtime, "ProcessDaemon", CapturingDaemon)
    monkeypatch.setattr(runtime, "pick_free_port", lambda: next(ports))

    args = argparse.Namespace(
        env_endpoint=None,
        vla_endpoint=None,
        dino_endpoint=None,
        task_name="picking_up_trash",
        task=1,
        public_seed=3,
        activity_definition_id=0,
        activity_instance_id=3,
        scene_model="house_double_floor_lower",
        max_episode_steps=24756,
        behavior_repo=str(tmp_path / "rlinf"),
        behavior_python=str(behavior_python),
        activity_instance_dir=None,
        env_config_path=None,
        policy_checkpoint=str(tmp_path / "checkpoint"),
        dino_source_archive=None,
        dino_weights=None,
        dino_cache_dir=None,
        behavior_env_cuda_device="2",
        behavior_model_cuda_device="7",
    )

    runtime._spawn_env_server(args, tmp_path / "env")
    runtime._spawn_vla_server(args, tmp_path / "vla")
    runtime._spawn_dino_server(args, tmp_path / "dino")

    by_name = {capture["name"]: capture for capture in captures}
    assert (
        by_name["behavior_env_server"]["env_overrides"]["CUDA_VISIBLE_DEVICES"] == "2"
    )
    assert (
        by_name["behavior_vla_server"]["env_overrides"]["CUDA_VISIBLE_DEVICES"] == "7"
    )
    assert (
        by_name["behavior_dino_server"]["env_overrides"]["CUDA_VISIBLE_DEVICES"] == "7"
    )
    for name, expected in (
        ("behavior_env_server", "2"),
        ("behavior_vla_server", "7"),
        ("behavior_dino_server", "7"),
    ):
        cmd = by_name[name]["cmd"]
        assert cmd.count("--cuda-device") == 1
        assert cmd[cmd.index("--cuda-device") + 1] == expected
        assert "," not in expected


def test_missing_runtime_components_hide_behavior_tools_but_keep_common(
    tmp_path,
) -> None:
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "output_dir": tmp_path,
        }
    )
    names = {spec["name"] for spec in toolkit.get_tools_spec()}

    assert names == {"read_text_file", "write_text_file", "list_dir", "finish"}


def test_finish_writes_terminal_receipt_without_forging_success(tmp_path) -> None:
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "output_dir": tmp_path,
        }
    )

    result = toolkit.execute_tool(
        "finish",
        {"status": "stopped", "summary": "bounded smoke stop"},
    ).result
    receipt = json.loads((tmp_path / "terminal_receipt.json").read_text("utf-8"))

    assert result["_finish"] is True
    assert result["task_success"] is False
    assert receipt == result


def test_readonly_observe_result_is_published_to_dashboard(tmp_path) -> None:
    class Sink:
        enabled = True

        def __init__(self) -> None:
            self.events = []

        def emit(self, event):
            self.events.append(event)

    @readonly
    def observe(*, camera: str = "head") -> dict[str, object]:
        return {
            "camera": camera,
            "resolved_camera": camera,
            "_image_bytes": b"\x89PNG\r\n\x1a\nbehavior",
        }

    sink = Sink()
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "output_dir": tmp_path,
        },
        dashboard_events=sink,
    )
    toolkit.add_tool(
        "observe",
        {
            "name": "observe",
            "description": "fake observe",
            "input_schema": {"type": "object", "properties": {}},
        },
        observe,
    )

    result = toolkit.execute_tool("observe", {"camera": "head"})

    assert result.result["_image_bytes"].startswith(b"\x89PNG")
    assert len(sink.events) == 1
    assert isinstance(sink.events[0], ToolResultEvent)
    assert sink.events[0].name == "observe"
    assert sink.events[0].result["_image_bytes"].startswith(b"\x89PNG")


def test_shared_vla_client_is_not_closed_by_per_task_toolkit(tmp_path) -> None:
    class Model:
        closed = False

        def close(self) -> None:
            self.closed = True

    model = Model()
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "output_dir": tmp_path,
            "model": model,
            "close_model_on_shutdown": False,
        }
    )

    toolkit.close()

    assert model.closed is False


def test_partial_vla_chunk_counts_only_backend_executed_steps(tmp_path) -> None:
    observation = {
        "main_images": np.zeros((2, 2, 3), dtype=np.uint8),
        "wrist_images": np.zeros((2, 2, 2, 3), dtype=np.uint8),
        "states": np.zeros(32, dtype=np.float32),
        "task_descriptions": "Turn on the radio.",
    }

    class Model:
        def predict_action_batch(self, _obs, *, mode):
            assert mode == "eval"
            return np.zeros((4, 23), dtype=np.float32), {}

    class Env:
        def pi0_nav_pick_chunk_step(self, actions, *, chunk_index):
            assert actions.shape == (4, 23)
            assert chunk_index == 0
            return (
                observation,
                0.0,
                True,
                False,
                {
                    "done": {"success": False},
                    "_rpent": {
                        "pi0_nav_pick_monitor": {
                            "requested_steps": 4,
                            "executed_steps": 2,
                            "stop_reason": "terminated",
                        }
                    },
                },
            )

    primitives = BehaviorPrimitives(
        env=Env(),
        model=Model(),
        output_dir=tmp_path,
        initial_observation=observation,
        task_name="turning_on_radio",
        public_seed=0,
    )

    result = primitives.pi0_nav_pick(instruction="Turn on the radio.", chunks=1)

    assert result["stop_reason"] == "terminated"
    assert result["env_steps_used"] == 2
    assert result["total_env_steps"] == 2
    assert result["full_chunks_executed"] == 0
