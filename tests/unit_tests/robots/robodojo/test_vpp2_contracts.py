# Copyright 2026 The RPent Authors.
"""Offline VPP2 history, native action and request isolation contracts."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from robots.robodojo.env_client import ExecutionSyncError, RoboDojoEnvClient
from robots.robodojo.primitives import RoboDojoPrimitives
from robots.robodojo.vpp2_client import VPP2Client
from rpent.tools.toolkit import ToolCancelled


def observation(step):
    return {
        "instruction": "Original environment task",
        "vision": {
            camera: {
                "color": np.full((2, 3, 3), step % 256, np.uint8),
                "depth": np.ones((2, 3)),
                "intrinsic_matrix": np.eye(3),
                "extrinsic_matrix": np.eye(4),
            }
            for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist")
        },
        "state": {
            f"{arm}_{field}": values
            for arm in ("left", "right")
            for field, values in (
                ("ee_pose", [step, 0, 1, 1, 0, 0, 0]),
                ("ee_joint_state", [0.5]),
            )
        },
    }


class EnvRPC:
    def __init__(self):
        self.step = 0
        self.limit = 1000
        self.success = False
        self.count = 1
        self.failure = None
        self.actions = []

    def status(self):
        return {"step": self.step, "step_limit": self.limit, "success": self.success}

    def call(self, method, args=(), kwargs=None, **options):
        if method == "env.get_env_meta":
            return {}
        if method == "env.get_status":
            return self.status()
        if method == "env.is_success":
            return self.success
        if method == "env.get_obs":
            return observation(self.step)
        if self.failure:
            raise self.failure
        if method == "env.reset":
            self.step = 0
            return observation(0)
        before = self.step
        self.step += self.count
        self.actions.append((method, args, kwargs))
        obs = observation(self.step)
        if method == "env.step":
            return (
                obs,
                0.0,
                self.success,
                {
                    "executed_actions": self.count,
                    "status": self.status(),
                    "step_before": before,
                    "step_after": self.step,
                    "terminated": self.success,
                    "truncated": self.step >= self.limit,
                },
            )
        if method == "env.step_joint_update":
            return {
                "step_before": before,
                "step_after": self.step,
                "observation": obs,
                "executed_actions": self.count,
                "episode_status": self.status(),
                "terminated": self.success,
                "truncated": self.step >= self.limit,
            }
        raise AssertionError(method)


@pytest.fixture
def connected():
    rpc = EnvRPC()
    return rpc, RoboDojoEnvClient(rpc, expected_meta={})


def stamps(payload):
    return [int(views[0][0, 0, 0]) for views in payload["history"]]


def test_history_includes_all_actions_but_not_reads_and_is_bounded(connected):
    rpc, env = connected
    assert stamps(env.policy_observation()) == [0] * 9
    for step in range(1, 201):
        if step % 2:
            env.step_joint_update("left", gripper=0.5)
        else:
            env.step({"left_ee_pose": [0, 0, 1, 1, 0, 0, 0]})
        env.get_obs()
    payload = env.policy_observation()
    assert stamps(payload) == [0, 25, 50, 75, 100, 125, 150, 175, 200]
    assert payload["inference_index"] == 1
    np.testing.assert_array_equal(payload["state"], [200, 0, 1, 1, 0, 0, 0, 0.5] * 2)
    assert len(env._rgb_history) == 177
    assert len(payload["history"]) == 9
    assert all(len(frame) == 3 for _, frame in env._rgb_history)
    assert all(rgb.dtype == np.uint8 for _, frame in env._rgb_history for rgb in frame)
    assert rpc.step == 200
    # Snapshots must not alias the current observation's mutable image buffers.
    env.last_obs["vision"]["cam_head"]["color"][:] = 99
    assert stamps(env.policy_observation())[-1] == 200
    env.reset()
    reset = env.policy_observation()
    assert reset["inference_index"] == 0 and stamps(reset) == [0] * 9


def test_short_history_repeats_only_anchor_and_zero_execution_adds_no_frame(connected):
    rpc, env = connected
    for _ in range(26):
        env.step_joint_update("right", gripper=0.4)
    assert stamps(env.policy_observation()) == [0] * 7 + [1, 26]
    rpc.count = 0
    env.step({})
    env.step_joint_update("left", gripper=0)
    assert stamps(env.policy_observation()) == [0] * 7 + [1, 26]


@pytest.mark.parametrize("mutation", ["step", "step_joint_update", "reset"])
def test_uncertain_mutation_requires_reset_not_read_refresh(connected, mutation):
    rpc, env = connected
    rpc.failure = TimeoutError("lost response")
    with pytest.raises(ExecutionSyncError):
        getattr(env, mutation)(
            *({"step": [{}], "step_joint_update": ["left"], "reset": []}[mutation])
        )
    rpc.failure = None
    env.get_obs()
    with pytest.raises(ExecutionSyncError, match="unavailable"):
        env.step({})
    with pytest.raises(RuntimeError, match="history unavailable"):
        env.policy_observation()
    env.reset()
    assert env.policy_observation()["inference_index"] == 0


@pytest.mark.parametrize("count", [2, -1])
def test_missing_intermediate_feedback_is_not_fabricated(connected, count):
    rpc, env = connected
    rpc.count = count
    with pytest.raises(ExecutionSyncError, match="zero or one"):
        env.step({})
    with pytest.raises(RuntimeError, match="history unavailable"):
        env.policy_observation()


def test_missing_action_observation_invalidates_history(connected, monkeypatch):
    rpc, env = connected
    original = rpc.call

    def missing(method, **kwargs):
        result = original(method, **kwargs)
        if method == "env.step_joint_update":
            result.pop("observation")
        return result

    monkeypatch.setattr(rpc, "call", missing)
    with pytest.raises(ExecutionSyncError, match="observation"):
        env.step_joint_update("left", gripper=0)
    with pytest.raises(RuntimeError, match="history unavailable"):
        env.policy_observation()


def policy_client(prediction=None):
    calls = []
    prediction = (
        np.arange(32 * 16).reshape(32, 16) if prediction is None else prediction
    )

    def call(method, args, **kwargs):
        assert method == "vla.predict"
        calls.append(args[0])
        return prediction

    return VPP2Client(SimpleNamespace(call=call)), calls


def test_client_sends_subtask_not_original_instruction_and_preserves_ee16(connected):
    _, env = connected
    client, calls = policy_client()
    actions = client.infer(
        env.policy_observation(), "Place the red block in the left basket"
    )
    assert calls[0]["instruction"] == "Place the red block in the left basket"
    assert "vision" not in calls[0] and "depth" not in str(calls[0].keys())
    assert len(actions) == 24
    for i, action in enumerate(actions):
        np.testing.assert_array_equal(
            action["left_ee_pose"], np.arange(i * 16, i * 16 + 7)
        )
        np.testing.assert_array_equal(action["left_ee_joint_state"], [i * 16 + 7])
        np.testing.assert_array_equal(
            action["right_ee_pose"], np.arange(i * 16 + 8, i * 16 + 15)
        )
        np.testing.assert_array_equal(action["right_ee_joint_state"], [i * 16 + 15])


@pytest.mark.parametrize("shape", [(24, 16), (32, 14), (1, 32, 16)])
def test_client_rejects_incompatible_prediction(shape):
    client, _ = policy_client(np.zeros(shape))
    with pytest.raises(ValueError, match="shape"):
        client.infer({}, "pick the block")


def test_client_checks_unexecuted_tail_for_nonfinite_values():
    bad = np.zeros((32, 16))
    bad[-1, -1] = np.nan
    client, _ = policy_client(bad)
    with pytest.raises(ValueError, match="finite"):
        client.infer({}, "pick the block")


def test_tool_replans_after_analytic_actions_and_keeps_subtasks_in_one_episode(
    connected,
):
    rpc, env = connected
    client, calls = policy_client()
    primitive = RoboDojoPrimitives(
        env=env, vla_client=client, check_cancelled=lambda: None
    )
    env.step_joint_update("left", gripper=0.5)
    result = primitive.vpp2_act(instruction="pick the red block", chunks=2).data
    assert result["completed"] and result["executed_actions"] == 48
    assert result["executed_chunks"] == 2
    assert result["episode_status"]["remaining_steps"] == 951
    assert stamps(calls[0])[-1] == 1 and stamps(calls[1])[-1] == 25
    assert [call["inference_index"] for call in calls] == [0, 1]
    primitive.vpp2_act(instruction="put it in the basket")
    assert calls[2]["inference_index"] == 2 and stamps(calls[2])[-1] == 49
    assert calls[2]["instruction"] == "put it in the basket"
    assert rpc.actions[-1][0] == "env.step"


@pytest.mark.parametrize(
    "terminal", ["budget", "success", "cancelled", "uncertain", "zero"]
)
def test_partial_or_terminal_execution_is_reported_without_extra_inference(
    connected, terminal
):
    rpc, env = connected
    client, calls = policy_client()

    def check():
        if rpc.step == 4:
            if terminal == "cancelled":
                raise ToolCancelled("cancelled")
            if terminal == "uncertain":
                rpc.failure = TimeoutError("lost response")
            if terminal == "zero":
                rpc.count = 0

    if terminal == "budget":
        rpc.limit = 4
    if terminal == "success":
        rpc.success = True
    primitive = RoboDojoPrimitives(env=env, vla_client=client, check_cancelled=check)
    result = primitive.vpp2_act(instruction="pick the block", chunks=2).data
    assert not result["completed"]
    assert result["requested_actions"] == 48
    assert result["executed_actions"] == (0 if terminal == "success" else 4)
    assert result.get("interrupted", False) == (terminal == "cancelled")
    assert result.get("execution_uncertain", False) == (terminal == "uncertain")
    assert len(calls) == (0 if terminal == "success" else 1)
    assert result["executed_chunks"] == 0
    assert (
        result["stop_reason"]
        == {
            "budget": "budget_exhausted",
            "success": "native_success",
            "cancelled": "cancelled",
            "uncertain": "execution_unknown",
            "zero": "runtime_failure",
        }[terminal]
    )


def test_inference_failure_keeps_completed_chunk_counts(connected):
    _, env = connected
    client, _ = policy_client()
    infer = client.infer

    def fail_second(obs, instruction):
        if obs["inference_index"] == 1:
            raise TimeoutError("model timeout")
        return infer(obs, instruction)

    client.infer = fail_second
    primitive = RoboDojoPrimitives(
        env=env, vla_client=client, check_cancelled=lambda: None
    )
    result = primitive.vpp2_act(instruction="pick the block", chunks=2).data
    assert result["executed_actions"] == 24 and result["executed_chunks"] == 1
    assert not result["completed"] and result["stop_reason"] == "inference_failed"
    assert env.policy_observation()["inference_index"] == 2


def test_toolkit_registers_policy_and_persists_subtask_without_recipe_export(
    connected, tmp_path, monkeypatch
):
    import json

    from robots.robodojo.toolkit import RoboDojoToolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.utils import logging

    rpc, env = connected
    client, _ = policy_client()
    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    toolkit = RoboDojoToolkit(
        primitives_kwargs={"env": env, "vla_client": client},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=tmp_path,
    )
    try:
        names = {tool.name for tool in toolkit.list_tools()}
        assert "vpp2_act" in names
        assert not names.intersection({"pi0_pick", "stabilize", "place_in_bin"})
        result = toolkit.execute_tool("vpp2_act", {"instruction": "pick the red block"})
        assert not result.is_error, result.data
        assert result.data["episode_status"] == {
            "step": 24,
            "step_limit": 1000,
            "remaining_steps": 976,
        }
        record = toolkit._state.latest_record()
        assert record.result["executed_actions"] == 24
        assert record.command["instruction"] == "pick the red block"
        assert "history" not in record.result
        assert toolkit._state.load("cam_head.png").shape == (2, 3, 3)
        rpc.success = True
        manifest = json.loads((tmp_path / "states.json").read_text())
        entry = manifest["steps"][-1]["command"]
        assert entry["action"] == "vpp2_act"
        assert entry["instruction"] == "pick the red block"
        assert manifest["steps"][-1]["result"]["executed_actions"] == 24
        assert toolkit.write_recipe("example") is None
        assert not (tmp_path / "example_recipe.jsonl").exists()
    finally:
        toolkit.close()


@pytest.fixture
def facade(monkeypatch):
    torch = pytest.importorskip("torch")
    from robots.robodojo.vpp2_server import VPP2Facade

    calls = []
    helpers = ModuleType("vpp2.policy")
    helpers._normalized_to_uint8 = lambda image: (
        ((image[0] + 1) * 127.5).round().to(torch.uint8)
    )
    helpers._uint8_to_normalized = lambda frames: (
        torch.stack(frames, dim=1).float() / 255 * 2 - 1
    )
    data = ModuleType("vpp2.data")
    data.DEFAULT_PROMPT = "Task: {task}"
    monkeypatch.setitem(sys.modules, "vpp2", ModuleType("vpp2"))
    monkeypatch.setitem(sys.modules, "vpp2.policy", helpers)
    monkeypatch.setitem(sys.modules, "vpp2.data", data)

    def compose(obs):
        rgb = obs["observation"]["head_camera"]["rgb"]
        # Deliberate fractional pixels distinguish latest image from quantized history.
        return (
            torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255 * 2
            - 1
            + 0.001
        )

    def infer(**kwargs):
        calls.append(kwargs)
        return {"action": torch.ones((1, 32, 16))}

    policy = SimpleNamespace(
        _compose_observation_image=compose,
        _normalize_state=lambda state: torch.from_numpy(state),
        _denormalize_action=lambda action: action.numpy(),
        model=SimpleNamespace(infer_action=infer),
        base_seed=1,
        video_seed_offset=1000003,
        action_horizon=32,
        negative_prompt="",
        text_cfg_scale=1.0,
        num_inference_steps=10,
        sigma_shift=1.0,
        rand_device="cpu",
        tiled=False,
        num_video_frames=17,
    )
    return VPP2Facade(policy), calls


def test_facade_uses_request_history_seed_and_subtask_without_cross_episode_state(
    connected, facade
):
    torch = pytest.importorskip("torch")
    _, env = connected
    server, calls = facade
    request_a = {**env.policy_observation(), "instruction": "pick the red block"}
    request_a["inference_index"] = 7
    for _ in range(26):
        env.step({})
    request_b = {**env.policy_observation(), "instruction": "put the blue block away"}
    request_b["inference_index"] = 0
    for request in (request_a, request_b, request_a):
        assert server.predict(request).shape == (32, 16)
    assert [call["seed"] for call in calls] == [8, 1, 8]
    assert [call["video_seed"] for call in calls] == [1000011, 1000004, 1000011]
    assert calls[1]["prompt"] == "Task: put the blue block away"
    assert calls[1]["condition_video"].shape == (1, 3, 9, 2, 3)
    torch.testing.assert_close(calls[0]["condition_video"], calls[2]["condition_video"])
    assert not torch.equal(
        calls[0]["input_image"], calls[0]["condition_video"][:, :, -1]
    )
    np.testing.assert_array_equal(calls[1]["proprio"].numpy(), request_b["state"])
    assert calls[1]["action_horizon"] == 32 and calls[1]["num_video_frames"] == 17


@pytest.mark.parametrize(
    "change",
    [
        {"instruction": "  "},
        {"inference_index": -1},
        {"inference_index": True},
        {"state": np.zeros(14)},
        {"state": np.full(16, np.nan)},
        {"history": []},
        {"history": [[np.zeros((2, 3, 3), dtype=float)] * 3] * 9},
        {"history": [[np.zeros((2, 3, 3), dtype=np.uint8)] * 2] * 9},
    ],
)
def test_facade_rejects_bad_request_before_model_execution(connected, facade, change):
    _, env = connected
    server, calls = facade
    with pytest.raises(ValueError):
        server.predict({**env.policy_observation(), "instruction": "pick", **change})
    assert not calls


def test_fresh_inspection_restores_analytic_control_without_repairing_history(
    connected,
):
    rpc, env = connected
    env.invalidate_execution()
    with pytest.raises(RuntimeError, match="history unavailable"):
        env.policy_observation()
    env.accept_observation_step(rpc.step, rpc.step)
    env.step_joint_update("left", gripper=0.5)
    assert rpc.step == 1
    with pytest.raises(RuntimeError, match="history unavailable"):
        env.policy_observation()
    env.reset()
    assert stamps(env.policy_observation()) == [0] * 9


def test_vpp2_model_resource_layout(tmp_path):
    from robots.robodojo.vpp2_server import model_paths

    for file in (
        "policy/src/vpp2/policy.py",
        "checkpoints/joint2b_s100000/action.pt",
        "checkpoints/joint2b_s100000/video.pt",
        "checkpoints/joint2b_s100000/dataset_stats.json",
    ):
        path = tmp_path / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "checkpoints/Wan2.1-I2V-14B-480P").mkdir()
    assert (
        model_paths(tmp_path)["checkpoint_path"]
        == tmp_path / "checkpoints/joint2b_s100000/action.pt"
    )
