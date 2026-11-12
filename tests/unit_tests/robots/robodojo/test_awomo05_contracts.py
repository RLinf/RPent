# Copyright 2026 The RPent Authors.
"""Awomo05 native model configuration, observations, actions and history."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo import awomo05_server as server
from robots.robodojo.access import ARX_X5_JOINT_FIELDS, CAMERAS
from robots.robodojo.awomo05_client import Awomo05Client
from robots.robodojo.primitives import RoboDojoPrimitives
from rpent.tools.toolkit import ToolCancelled


def test_native_payload_keeps_camera_colors_and_named_joint_order():
    calls = []
    state = {
        key: np.full(size, index, np.float32)
        for index, (key, size) in enumerate(ARX_X5_JOINT_FIELDS.items())
    }
    image = np.array([[[255, 17, 3]]], dtype=np.uint8)
    obs = {
        "state": state,
        "vision": {
            camera: {
                "color": image.copy(),
                "depth": np.ones((1, 1)),
                "intrinsic_matrix": np.eye(3),
                "extrinsic_matrix": np.eye(4),
            }
            for camera in CAMERAS
        },
        "instruction": "stack blocks",
    }
    actions = [state] * 8

    def call(method, **kwargs):
        calls.append((method, kwargs))
        return actions

    client = Awomo05Client(SimpleNamespace(call=call))
    client.update_obs(obs, 0)
    sent = calls[-1][1]["args"][0]
    assert sent["instruction"] == obs["instruction"]
    assert all(sent["state"][key] is state[key] for key in ARX_X5_JOINT_FIELDS)
    assert all(
        sent["vision"][name]["color"] is obs["vision"][name]["color"]
        for name in CAMERAS
    )
    assert all(set(camera) == {"color"} for camera in sent["vision"].values())
    assert all("depth" in camera for camera in obs["vision"].values())
    np.testing.assert_array_equal(obs["vision"][CAMERAS[0]]["color"], image)
    assert client.get_action() is actions
    obs["vision"].pop(CAMERAS[-1])
    with pytest.raises(KeyError, match="cam_right_wrist"):
        client.update_obs(obs, 1)


@pytest.mark.parametrize(
    "actions",
    [
        [],
        [[0.0] * 14] * 8,
        [{"left_arm_joint_state": [0] * 6}] * 8,
        [{key: [float("nan")] * size for key, size in ARX_X5_JOINT_FIELDS.items()}] * 8,
        [{key: [0] * size for key, size in ARX_X5_JOINT_FIELDS.items()}] * 7,
        [{key: [0] * size for key, size in ARX_X5_JOINT_FIELDS.items()}] * 9,
    ],
)
def test_invalid_native_actions_are_rejected(actions):
    def call(method, **kwargs):
        return actions

    client = Awomo05Client(SimpleNamespace(call=call))
    with pytest.raises(ValueError):
        client.get_action()


@pytest.fixture
def resources(tmp_path):
    files = [
        "policy/env_cfg/arx_x5.yml",
        "policy/env_cfg/robot/_robot_info.json",
        "policy/XPolicyLab/policy/Awomo05/model.py",
        "policy/XPolicyLab/policy/Awomo05/model_config.yaml",
        "policy/XPolicyLab/policy/Awomo05/deploy.yml",
        "checkpoints/model.pt",
        "checkpoints/dataset_stats.json",
        "checkpoints/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors",
        "checkpoints/flux2-dev/ae.safetensors",
        "checkpoints/Qwen3-VL-4B-Instruct/config.json",
    ]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    (tmp_path / "policy/XPolicyLab/policy/Awomo05/deploy.yml").write_text(
        "history_enabled: true\nnum_inference_steps: 10\neval_batch: true\n"
    )
    return tmp_path


@pytest.fixture
def native_model(resources, monkeypatch):
    native = SimpleNamespace(
        history_grid=object(),
        replan_steps=8,
        run_cfg=SimpleNamespace(data=SimpleNamespace(train={"history": {"fps": 25}})),
    )

    def load(cfg):
        native.model_cfg = cfg
        return native

    monkeypatch.setitem(
        sys.modules,
        "XPolicyLab.policy.Awomo05.model",
        SimpleNamespace(RAW_FPS=25, Model=load, _opt_float=lambda v: v),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        sys, "argv", ["server", "--model-path", str(resources), "--port", "9005"]
    )

    def bind(facade, *args):
        facade._shutdown_event.set()
        return SimpleNamespace(
            serve_forever=lambda: None, shutdown=lambda: None, server_close=lambda: None
        )

    monkeypatch.setattr(server.Awomo05Facade, "_bind_and_announce", bind)
    return native


def test_resource_root_supplies_model_config_without_environment(
    resources, native_model, monkeypatch
):
    monkeypatch.setenv("AWOMO05_WEIGHTS", "/unrelated/weights")
    server.main()
    cfg = native_model.model_cfg
    assert cfg["ckpt"] == str(resources / "checkpoints/model.pt")
    assert cfg["dataset_stats_path"] == str(
        resources / "checkpoints/dataset_stats.json"
    )
    assert cfg["flux2_base_path"] == str(
        resources / "checkpoints/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors"
    )
    assert cfg["ae_path"] == str(resources / "checkpoints/flux2-dev/ae.safetensors")
    assert cfg["qwen_vl_path"] == str(resources / "checkpoints/Qwen3-VL-4B-Instruct")
    assert cfg["history_enabled"] and cfg["num_inference_steps"] == 10
    assert (
        cfg["env_cfg_type"],
        cfg["action_type"],
        cfg["input_color_order"],
        cfg["replan_steps"],
        cfg["seed"],
    ) == ("arx_x5", "joint", "bgr", 8, 0)
    assert sys.path[:2] == [
        str(resources / "policy"),
        str(resources / "policy/XPolicyLab"),
    ]


@pytest.mark.parametrize(
    "missing",
    [
        "checkpoints/model.pt",
        "checkpoints/dataset_stats.json",
        "policy/env_cfg/robot/_robot_info.json",
    ],
)
def test_missing_resources_fail_before_model_import(resources, missing):
    (resources / missing).unlink()
    with pytest.raises(FileNotFoundError, match=missing):
        server.model_paths(resources)


@pytest.mark.parametrize(
    "history_fps,history_enabled,replan_steps,error",
    [
        (None, True, 8, None),
        (25, True, 8, None),
        (10, True, 8, "25 Hz"),
        (25, False, 8, "enabled history"),
        (25, True, 4, "8 executed steps"),
    ],
)
def test_model_loading_checks_only_history_timing_and_chunk_size(
    resources,
    native_model,
    monkeypatch,
    history_fps,
    history_enabled,
    replan_steps,
    error,
):
    import signal

    previous = signal.getsignal(signal.SIGTERM)
    native_model.history_grid = object() if history_enabled else None
    native_model.replan_steps = replan_steps
    with (resources / "policy/XPolicyLab/policy/Awomo05/deploy.yml").open(
        "a"
    ) as stream:
        stream.write(
            f"history_fps: {history_fps if history_fps is not None else 'null'}\n"
        )
    served = []

    serve = server.Awomo05Facade.serve

    def record_serve(facade, **kwargs):
        served.append(facade._model)
        return serve(facade, **kwargs)

    monkeypatch.setattr(server.Awomo05Facade, "serve", record_serve)
    if error:
        with pytest.raises(ValueError, match=error):
            server.main()
        assert not served
    else:
        server.main()
        assert served == [native_model]
    assert signal.getsignal(signal.SIGTERM) is previous


@pytest.mark.parametrize("failure", ["changed", "unavailable"])
def test_task_instruction_comes_from_initial_observation_without_extra_queries(
    monkeypatch, failure, episode_policy, fake_env
):
    native, policy = fake_env(100), episode_policy()
    initial = native.get_obs()
    queries = []
    instructions = []
    update_obs = policy.update_obs

    def record_instruction(obs, step):
        instructions.append(obs["instruction"])
        update_obs({**obs, "instruction": "stack the blocks"}, step)

    monkeypatch.setattr(policy, "update_obs", record_instruction)

    def task_language():
        queries.append(True)
        if failure == "unavailable":
            raise OSError("task instruction unavailable")
        return "Another task"

    # Model observations are available even if the separate metadata RPC fails.
    def observation():
        return {**initial, "state": {**initial["state"], "frame": native.frame}}

    monkeypatch.setattr(native, "get_obs", observation)
    monkeypatch.setattr(native, "get_task_language", task_language)
    primitive = RoboDojoPrimitives(
        env=native,
        vla_client=policy,
        vla_backend="awomo05",
        check_cancelled=lambda: None,
    )
    try:
        for _ in range(2):
            result = primitive.awomo05_act()
            assert not result.is_error
            assert result.data["executed_steps"] == 8
        assert native.frame == 16
        assert queries == []
        assert [e for e in policy.events if e[0] == "infer"] == [
            ("infer", 0),
            ("infer", 8),
        ]
        assert instructions == ["stack the blocks"] * 17
        initial["instruction"] = "Another task"
        native.reset()
        assert not primitive.awomo05_act().is_error
        assert native.frame == 8
        assert instructions[17:] == ["Another task"] * 9
    finally:
        primitive.close()


@pytest.mark.parametrize(
    "failure", [TimeoutError("model result uncertain"), ToolCancelled("lost inference")]
)
def test_model_inference_failure_also_blocks_native_primitives(
    failure, episode_policy, fake_env
):
    native, client = fake_env(), episode_policy()
    executor = RoboDojoPrimitives(
        env=native,
        vla_client=client,
        check_cancelled=lambda: None,
        vla_backend="awomo05",
    )

    def fail():
        raise failure

    client.get_action = fail
    try:
        result = executor.awomo05_act(chunks=2).data
        assert str(failure) in result["error"]
        assert result["stop_reason"] == "synchronization_lost"
        native.resume_execution()
        blocked = executor.awomo05_act(chunks=1)
        assert blocked.is_error and blocked.data["executed_steps"] == 0
        assert "unavailable" in blocked.data["error"]
        result = (
            RoboDojoPrimitives(env=native, check_cancelled=lambda: None)
            .set_gripper(arm="left", val=0.7, steps=2)
            .data
        )
        assert result["stop_reason"] == "synchronization_lost"
        assert result["executed_actions"] == native.frame == 0
        assert client.events.count(("reset",)) == 1
    finally:
        executor.close()


def test_history_tracks_executed_actions_and_resets(policy_runtime):
    env, policy, primitive = policy_runtime("awomo05")
    assert primitive.awomo05_act().data["executed_steps"] == 8
    assert primitive.set_gripper(arm="left", val=0.7, steps=1).data["completed"]
    assert primitive.awomo05_act().data["executed_steps"] == 8
    assert [e[1] for e in policy.events if e[0] == "observe"] == list(range(18))
    assert [e[1] for e in policy.events if e[0] == "infer"] == [0, 9]
    env.reset()
    assert primitive.awomo05_act().data["executed_steps"] == 8
    assert policy.events[-11:] == [
        ("reset",),
        ("observe", 0),
        ("infer", 0),
        *[("observe", step) for step in range(1, 9)],
    ]
    primitive.close()
    primitive.close()
    assert policy.events.count(("end",)) == 1
    assert not env.closed


@pytest.mark.parametrize("invalid", ["state", "rgb", "instruction"])
def test_invalid_observation_does_not_reach_model(fake_env, invalid):
    calls = []
    model = SimpleNamespace(
        reset=lambda: calls.append("reset"),
        update_obs=lambda obs: calls.append("update_obs"),
    )
    facade = server.Awomo05Facade(model)
    obs = fake_env().get_obs()
    if invalid == "state":
        obs["state"]["left_arm_joint_state"][0] = float("nan")
    elif invalid == "rgb":
        obs["vision"]["cam_head"]["color"] = np.zeros((2, 2, 3), dtype=float)
    else:
        obs["instruction"] = " "
    try:
        facade.reset_session(session_id="owner")
        with pytest.raises(ValueError):
            facade.update_obs(obs, 0, session_id="owner")
        assert calls == ["reset"]
        facade.update_obs(fake_env().get_obs(), 0, session_id="owner")
        assert calls == ["reset", "update_obs"]
    finally:
        facade.close()


@pytest.fixture
def awomo_service():
    history = []
    actions = [{key: np.zeros(size) for key, size in ARX_X5_JOINT_FIELDS.items()}] * 8
    model = SimpleNamespace(
        history=history,
        reset=history.clear,
        update_obs=history.append,
        get_action=lambda: actions,
    )
    facade = server.Awomo05Facade(model)

    def connect(session):
        facade.register_session(session)

        def call(method, *, args=(), kwargs=None, **options):
            return facade._dispatch(method, args, kwargs or {}, session_id=session)

        return Awomo05Client(
            SimpleNamespace(call=call, close=lambda: facade.drop_session(session))
        )

    yield model, connect
    facade.close()


def test_native_history_has_one_owner_and_ordered_observations(awomo_service, fake_env):
    model, connect = awomo_service
    owner, other = connect("owner"), connect("other")
    owner.reset_session()
    with pytest.raises(RuntimeError, match="another session"):
        other.reset_session()
    with pytest.raises(RuntimeError, match="observation"):
        owner.get_action()
    obs = fake_env().get_obs()
    owner.update_obs(obs, 0)
    with pytest.raises(ValueError, match="Expected observation step 1"):
        owner.update_obs(obs, 0)
    for _ in range(2):
        assert len(owner.get_action()) == 8
    assert len(model.history) == 1
    owner.end_episode()
    other.reset_session()
    assert model.history == []
    with pytest.raises(RuntimeError, match="acquire"):
        owner.update_obs(obs, 0)
    other.close()
    owner.reset_session()
    owner.update_obs(obs, 0)
    assert len(owner.get_action()) == 8
    owner.close()


@pytest.mark.parametrize("method", ["reset", "update_obs", "get_action"])
def test_native_model_failure_requires_episode_reset(
    awomo_service, fake_env, monkeypatch, method
):
    model, connect = awomo_service
    client = connect("owner")
    obs = fake_env().get_obs()
    client.reset_session()
    client.update_obs(obs, 0)

    def fail(*args):
        raise RuntimeError("native model failed after mutation")

    with monkeypatch.context() as patch:
        patch.setattr(model, method, fail)
        with pytest.raises(RuntimeError, match="native model failed"):
            {
                "reset": client.reset_session,
                "update_obs": lambda: client.update_obs(obs, 1),
                "get_action": client.get_action,
            }[method]()
        with pytest.raises(RuntimeError, match="uncertain"):
            client.get_action()
    client.reset_session()
    client.update_obs(obs, 0)
    assert len(model.history) == 1
    assert len(client.get_action()) == 8
    client.close()
