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

import numpy as np
import pytest

from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.env_server import RoboCasaEnvFacade
from robots.robocasa.primitives import RoboCasaPrimitives
from robots.robocasa.rldx_skill import ACTION_FIELDS, RLDXSkill
from robots.robocasa.toolkit import FORMAL_PRIMITIVE_TOOLS, RoboCasaToolkit
from robots.robocasa.tools import (
    back_project,
    back_project_batch,
    view_camera_meta,
    view_env_state,
)
from robots.robocasa.vla_client import RoboCasaVLAClient
from robots.robocasa.vla_server import RoboCasaVLAFacade
from rpent.session import EnvState

ACTION_SCHEMA = {
    "name": "rldx.robocasa.action.v1",
    "batch_size": 1,
    "max_horizon": 512,
    "flat_dim": 12,
    "fields": [{"name": name, "size": size} for name, size in ACTION_FIELDS.items()],
}


def _env_runtime(
    *,
    episode_id=0,
    sim_step=0,
    success=False,
    perception_isolation=False,
):
    return {
        "protocol_version": 1,
        "runtime_id": "env-runtime",
        "episode_id": episode_id,
        "sim_step": sim_step,
        "success_latched": success,
        "perception_isolation": perception_isolation,
        "robot": "PandaOmron",
        "action_schema": {
            "name": "robocasa.panda_omron.flat.v1",
            "flat_dim": 12,
            "fields": [
                {"name": name, "size": size} for name, size in ACTION_FIELDS.items()
            ],
        },
        "cameras": [
            "mobilebase0_navview",
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
    }


def _vla_runtime(*, warmed_up=True, verified=False):
    return {
        "protocol_version": 1,
        "runtime_id": "vla-runtime",
        "backend": "rldx",
        "model": {
            "kind": "local",
            "fingerprint": "model-fingerprint",
            "verified": verified,
        },
        "warmed_up": warmed_up,
        "action_schema": ACTION_SCHEMA,
        "observation_schema": {},
        "modality": {"video_delta_indices": [-6, -4, -2, 0], "hist_maxlen": 8},
    }


def _raw_obs():
    return {
        "language": "pick up the mug",
        "robot0_eef_pos": np.array([0.0, 0.0, 1.0]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.02, 0.02]),
        "robot0_base_pos": np.array([0.0, 0.0, 0.0]),
        "robot0_base_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_base_to_eef_pos": np.array([0.0, 0.0, 1.0]),
        "robot0_base_to_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    }


def _actions(horizon=4, *, fill=0.0):
    return {
        name: np.full((1, horizon, size), fill, dtype=np.float32)
        for name, size in ACTION_FIELDS.items()
    }


class _EnvRpc:
    def __init__(self):
        self.episode = 0
        self.sim_step = 0
        self.reset_calls = 0
        self.success = False

    def call(self, method, args=(), kwargs=None, timeout_s=None):
        if method == "env.get_env_meta":
            return {
                "task_name": "OpenDrawer",
                "split": "target",
                "seed": 7,
                "camera_h": 256,
                "camera_w": 256,
            }
        if method == "env.get_runtime_info":
            return _env_runtime(
                episode_id=self.episode,
                sim_step=self.sim_step,
                success=self.success,
            )
        if method == "env.reset":
            self.reset_calls += 1
            self.episode += 1
            self.sim_step = 0
            self.success = False
            return _raw_obs()
        if method == "env.step":
            self.sim_step += 1
            return (
                _raw_obs(),
                0.0,
                False,
                {
                    "rpent_runtime_id": "env-runtime",
                    "rpent_episode_id": self.episode,
                    "rpent_sim_step": self.sim_step,
                    "rpent_success_latched": self.success,
                },
            )
        if method == "env.check_success":
            return self.success
        raise AssertionError(f"unexpected RPC method: {method}")


class _FakeVLA:
    runtime_id = "vla-runtime"
    action_schema = ACTION_SCHEMA

    def __init__(self, *, horizon=4):
        self.horizon = horizon
        self.predict_calls = 0
        self.reset_calls = []

    def get_modality_config(self):
        return {"video_delta_indices": [-6, -4, -2, 0], "hist_maxlen": 8}

    def predict(self, obs, options):
        self.predict_calls += 1
        return _actions(self.horizon)

    def reset_session(self, session_id):
        self.reset_calls.append(session_id)
        return {"ok": True, "runtime_id": self.runtime_id, "session_id": session_id}


class _SkillEnv:
    runtime_id = "env-runtime"
    action_dim = 12

    def __init__(self, *, succeed_after=1):
        self.last_obs = _raw_obs()
        self.step_calls = 0
        self.succeed_after = succeed_after
        self.reset_calls = 0

    @property
    def current_raw_obs(self):
        return self.last_obs

    @property
    def eef_pos(self):
        return self.last_obs["robot0_eef_pos"]

    @property
    def eef_quat(self):
        return self.last_obs["robot0_eef_quat"]

    @property
    def gripper_qpos(self):
        return self.last_obs["robot0_gripper_qpos"]

    def get_task_language(self):
        return self.last_obs["language"]

    def render_camera(self, *args, **kwargs):
        return np.zeros((256, 256, 3), dtype=np.uint8)

    def reassemble_env_action(self, value):
        return np.zeros(12, dtype=np.float64)

    def step(self, action):
        assert np.asarray(action).shape == (12,)
        self.step_calls += 1
        return self.last_obs, 0.0, False, {}

    def check_success(self):
        return self.step_calls >= self.succeed_after

    @property
    def terminated(self):
        return self.check_success()

    def grasp_contact(self):
        return False, None

    def reset(self):
        self.reset_calls += 1
        return self.last_obs


def test_navview_is_required_only_for_formal_isolation():
    ordinary = _env_runtime()
    ordinary["cameras"].remove("mobilebase0_navview")
    RoboCasaEnvClient._validate_runtime_info(ordinary)

    formal = {**ordinary, "perception_isolation": True}
    with pytest.raises(RuntimeError, match="mobilebase0_navview"):
        RoboCasaEnvClient._validate_runtime_info(formal)


def test_client_and_primitives_only_reset_once_on_startup(tmp_path):
    rpc = _EnvRpc()
    client = RoboCasaEnvClient(
        rpc,
        expected_meta={
            "task_name": "OpenDrawer",
            "split": "target",
            "seed": 7,
            "camera_h": 256,
            "camera_w": 256,
        },
    )
    assert rpc.reset_calls == 1

    RoboCasaPrimitives(client, str(tmp_path), None, _FakeVLA())
    assert rpc.reset_calls == 1


class _RawEnv:
    action_dim = 12

    def __init__(self, *, reset_success=False):
        self.success = reset_success
        self.reset_success = reset_success
        self.step_calls = 0

    def reset(self):
        self.success = self.reset_success
        return {"obs": 0}

    def step(self, action):
        self.step_calls += 1
        self.success = True
        return {"obs": 1}, 0.0, False, {}

    def _check_success(self):
        return self.success


def test_env_server_latches_success_on_each_simulator_action():
    facade = object.__new__(RoboCasaEnvFacade)
    facade.env = _RawEnv()
    facade._runtime_id = "runtime"
    facade._episode_id = 0
    facade._sim_step = 0
    facade._success_latched = False

    facade.reset()
    _, _, _, info = facade.step(np.zeros(12))
    assert info["rpent_success_now"] is True
    assert info["rpent_success_latched"] is True

    facade.env.success = False
    assert facade.check_success() is True
    with pytest.raises(ValueError, match="shape"):
        facade.step(np.zeros((1, 12)))
    bad = np.zeros(12)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        facade.step(bad)

    facade.reset()
    observations, rewards, dones, infos = facade.chunk_step(
        np.zeros((4, 12)), return_all_frames=True
    )
    assert len(observations) == len(rewards) == len(dones) == len(infos) == 1
    assert infos[0]["rpent_success_latched"] is True


def test_env_server_latches_reset_success_and_refuses_further_motion():
    facade = object.__new__(RoboCasaEnvFacade)
    facade.env = _RawEnv(reset_success=True)
    facade._runtime_id = "runtime"
    facade._episode_id = 0
    facade._sim_step = 0
    facade._success_latched = False

    facade.reset()
    assert facade.check_success() is True
    with pytest.raises(RuntimeError, match="further motion is disabled"):
        facade.step(np.zeros(12))
    assert facade.env.step_calls == 0


def test_rldx_uses_unique_sessions_and_stops_inside_action_chunk():
    env = _SkillEnv(succeed_after=1)
    vla = _FakeVLA(horizon=4)
    skill = RLDXSkill(env, vla_client=vla)
    other = RLDXSkill(env, vla_client=vla)
    assert skill._sid != other._sid
    assert "rc_agent_rldx_0" not in skill._sid

    skill._unmap = lambda value: value
    result = skill.run("pick up the mug", max_chunks=2, n_action_steps=4)
    assert result["status"] == "success"
    assert result["steps_applied"] == 1
    assert result["chunks"] == 1
    assert env.step_calls == 1
    assert vla.predict_calls == 1


def test_vla_server_warmup_and_client_handshake():
    class Policy:
        def __init__(self):
            self.predict_calls = 0
            self.reset_calls = []

        def get_action(self, obs, options):
            self.predict_calls += 1
            return _actions(2), {}

        def reset(self, options):
            self.reset_calls.append(options["session_ids"][0])

    facade = object.__new__(RoboCasaVLAFacade)
    facade._vdi = np.array([-6, -4, -2, 0])
    facade._runtime_id = "warmup-runtime"
    facade._warmed_up = False
    facade.policy = Policy()
    facade._warmup()
    assert facade._warmed_up is True
    assert facade.policy.predict_calls == 1
    assert facade.policy.reset_calls == ["warmup:warmup-runtime"]

    class Rpc:
        def call(self, method, args=(), kwargs=None, timeout_s=None):
            if method == "vla.get_runtime_info":
                return _vla_runtime()
            if method == "vla.get_modality_config":
                return {"video_delta_indices": [-6, -4, -2, 0], "hist_maxlen": 8}
            if method == "vla.predict":
                return _actions(2)
            if method == "vla.reset_session":
                return {"ok": True, "runtime_id": "vla-runtime", "session_id": args[0]}
            raise AssertionError(method)

    client = RoboCasaVLAClient(Rpc())
    assert client.runtime_id == "vla-runtime"
    assert client.get_modality_config()["hist_maxlen"] == 8
    assert client.predict({}, {})["action.base_motion"].shape == (1, 2, 4)

    class ColdRpc(Rpc):
        def call(self, method, args=(), kwargs=None, timeout_s=None):
            if method == "vla.get_runtime_info":
                return _vla_runtime(warmed_up=False)
            return super().call(method, args=args, kwargs=kwargs, timeout_s=timeout_s)

    with pytest.raises(RuntimeError, match="warmup"):
        RoboCasaVLAClient(ColdRpc())

    with pytest.raises(RuntimeError, match="verified"):
        RoboCasaVLAClient(Rpc(), require_verified_model=True)

    class VerifiedRpc(Rpc):
        def call(self, method, args=(), kwargs=None, timeout_s=None):
            if method == "vla.get_runtime_info":
                return _vla_runtime(verified=True)
            return super().call(method, args=args, kwargs=kwargs, timeout_s=timeout_s)

    verified = RoboCasaVLAClient(
        VerifiedRpc(),
        require_verified_model=True,
        expected_model_fingerprint="model-fingerprint",
    )
    assert verified.model_identity["verified"] is True
    with pytest.raises(RuntimeError, match="fingerprint"):
        RoboCasaVLAClient(
            VerifiedRpc(), expected_model_fingerprint="different-fingerprint"
        )


def test_base_and_vla_motion_invalidate_pose_dependent_calibration(tmp_path):
    env = _SkillEnv(succeed_after=999)
    primitives = RoboCasaPrimitives(env, str(tmp_path), None, _FakeVLA())
    primitives._pos_jac = np.eye(3)
    primitives._cam_meta_cache = {
        "agentview": {"old": True},
        "navview": {"old": True},
        "wrist": {"old": True},
    }
    primitives.move_base(forward=0.1, steps=1)
    assert primitives._pos_jac is None
    assert "agentview" not in primitives._cam_meta_cache
    assert "navview" not in primitives._cam_meta_cache
    assert "wrist" in primitives._cam_meta_cache

    primitives._pos_jac = np.eye(3)
    primitives._fwd_offset = 1.0
    primitives._cam_meta_cache = {
        "agentview": {},
        "navview": {},
        "wrist": {},
    }
    primitives._rldx.run = lambda *args, **kwargs: {"ok": True}
    primitives.run_rldx_skill(None, 1, True, "pick up the mug", False, 1, 1, 0.01)
    assert primitives._pos_jac is None
    assert primitives._fwd_offset is None
    assert primitives._cam_meta_cache == {}

    with pytest.raises(ValueError, match="steps"):
        primitives.move_base(steps=0)
    with pytest.raises(ValueError, match="finite"):
        primitives.move_base(forward=np.nan)


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("move_base", {"forward": 0.1, "steps": 5}),
        ("set_gripper", {"gripper": 1.0, "steps": 5}),
        ("rotate_pitch", {"target_pitch": 0.5, "n": 5}),
    ],
)
def test_analytic_primitives_stop_on_first_success(tmp_path, method, kwargs):
    env = _SkillEnv(succeed_after=1)
    primitives = RoboCasaPrimitives(env, str(tmp_path), None, _FakeVLA())

    getattr(primitives, method)(**kwargs)

    assert env.step_calls == 1


def test_perception_isolation_never_reads_or_exposes_task_progress(
    tmp_path, monkeypatch
):
    class IsolatedEnv(_SkillEnv):
        def __init__(self):
            super().__init__(succeed_after=999)
            self.progress_calls = 0
            self.criteria_calls = 0

        def get_task_progress(self):
            self.progress_calls += 1
            return {"hidden_counter": 7}

        def get_success_criteria_text(self):
            self.criteria_calls += 1
            return "hidden predicate"

    monkeypatch.setenv("RLDX_PERCEPTION_ISOLATION", "1")
    env = IsolatedEnv()
    primitives = RoboCasaPrimitives(env, str(tmp_path), None, _FakeVLA())

    state = primitives.current_state_dict()
    assert "task_progress" not in state
    assert "robocasa_terminated" not in state
    assert primitives.task_progress() == {}
    with pytest.raises(PermissionError, match="unavailable"):
        primitives.dump_success_criteria()
    assert env.progress_calls == 0
    assert env.criteria_calls == 0


def test_formal_vla_act_uses_exact_prompt_and_frozen_controls(tmp_path, monkeypatch):
    monkeypatch.setenv("RLDX_PERCEPTION_ISOLATION", "1")
    monkeypatch.setenv("RLDX_MAX_CHUNKS", "40")
    monkeypatch.setenv("RLDX_SETTLE_PATIENCE", "999")
    primitives = RoboCasaPrimitives(
        _SkillEnv(succeed_after=999), str(tmp_path), None, _FakeVLA()
    )
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "ok": True,
            "prompt": "pick up the mug",
            "status": "settled",
            "chunks": 3,
            "steps_applied": 24,
            "grasp_detected": True,
            "grasped": False,
            "grasp_obj": "mug",
        }

    primitives._rldx.run = run
    result = primitives.vla_act("pick up the mug")
    args, kwargs = calls[0]
    assert args == ("pick up the mug", 40, 8)
    assert kwargs == {
        "base_clip": None,
        "settle_patience": 999,
        "settle_eps": 0.012,
        "force_reset": True,
        "recording": False,
        "record_frame": primitives.record_frame,
    }
    assert result == {
        "ok": True,
        "status": "settled",
        "chunks": 3,
        "steps_applied": 24,
        "contact_made": True,
        "holding": False,
    }
    with pytest.raises(ValueError, match="verbatim"):
        primitives.vla_act("pick up a mug")
    assert len(calls) == 1


def test_formal_toolkit_registers_only_frozen_primitive_actions(tmp_path):
    class PrimitiveHandlers:
        _perception_isolation = True

        def __getattr__(self, _name):
            return lambda **_kwargs: {}

    toolkit = object.__new__(RoboCasaToolkit)
    toolkit._state = EnvState(tmp_path)
    toolkit._primitives = PrimitiveHandlers()
    toolkit._tools = {}
    toolkit._register_robocasa_tools()

    readonly_names = {
        "view_env_state",
        "view_camera_meta",
        "back_project",
        "back_project_batch",
        "query_world_map",
        "finish",
    }
    assert set(toolkit._tools) - readonly_names == FORMAL_PRIMITIVE_TOOLS
    assert {"scripted_grasp", "rldx_skill", "rldx_arm", "reset"}.isdisjoint(
        toolkit._tools
    )


def test_env_server_removes_privileged_rpcs_in_isolation():
    facade = object.__new__(RoboCasaEnvFacade)
    facade._perception_isolation = True
    facade._rpc = {}
    facade._readonly_methods = set()
    facade._register_rpc()

    assert "env.get_success_criteria_text" not in facade._rpc
    assert "env.get_task_progress" not in facade._rpc
    with pytest.raises(PermissionError, match="unavailable"):
        facade.get_success_criteria_text()
    with pytest.raises(PermissionError, match="unavailable"):
        facade.get_task_progress()


def test_env_server_allows_only_initial_reset_in_formal_mode():
    facade = object.__new__(RoboCasaEnvFacade)
    facade.env = _RawEnv()
    facade._perception_isolation = True
    facade._allow_reset = False
    facade._initial_reset_consumed = False
    facade._episode_id = 0
    facade._sim_step = 0
    facade._success_latched = False

    assert facade.reset() == {"obs": 0}
    with pytest.raises(PermissionError, match="disabled"):
        facade.reset()
    assert facade._episode_id == 1


def test_env_server_allows_explicit_exploration_reset():
    facade = object.__new__(RoboCasaEnvFacade)
    facade.env = _RawEnv()
    facade._perception_isolation = True
    facade._allow_reset = True
    facade._initial_reset_consumed = True
    facade._episode_id = 1
    facade._sim_step = 12
    facade._success_latched = True

    assert facade.reset() == {"obs": 0}
    assert facade._episode_id == 2
    assert facade._sim_step == 0
    assert facade._success_latched is False


def test_view_state_omits_absent_progress_field(tmp_path):
    state = EnvState(tmp_path)
    with state.record_step(
        state={"robot0_eef_pos": [0.0, 0.0, 1.0]},
        terminated=False,
        extras={
            "task_language": "pick up the mug",
            "success": False,
            "vla_desync": True,
        },
    ):
        pass
    result = view_env_state(step=0, state=state)
    assert "task_progress" not in result


def test_camera_metadata_and_back_projection_tools(tmp_path):
    state = EnvState(tmp_path)
    world = np.zeros((4, 5, 3), dtype=np.float32)
    world[1, 2] = [1.25, -0.5, 0.9]
    meta = {
        "camera_name": "robot0_agentview_left",
        "height": 4,
        "width": 5,
        "intrinsic": np.eye(3).tolist(),
        "extrinsic_cam2world": np.eye(4).tolist(),
        "depth_near": 0.01,
        "depth_far": 10.0,
        "runtime_id": "env-runtime",
        "episode_id": 1,
        "sim_step": 0,
    }
    with state.record_step(state={}, terminated=False) as step:
        state.save("agentview_world.npz", world, step=step)
        state.save("agentview_world_high.npz", world, step=step)
        state.save("agentview_metadata.json", meta, step=step)

    camera_meta = view_camera_meta("agentview", step=0, state=state)
    assert camera_meta["artifact"] == "agentview_metadata.json"
    assert camera_meta["step"] == 0

    point = back_project(1, 2, step=0, state=state)
    assert point["valid"] is True
    assert point["world_xyz"] == [1.25, -0.5, 0.9]
    assert point["source_artifact"] == "agentview_world_high.npz"
    assert point["step"] == 0

    invalid = back_project_batch([[1.5, 2]], step=0, state=state)
    assert invalid["results"][0]["valid"] is False
    assert "integers" in invalid["results"][0]["error"]
    assert "error" in back_project_batch([[0, 0]] * 51, step=0, state=state)
    assert "error" in view_camera_meta("unknown", step=0, state=state)
