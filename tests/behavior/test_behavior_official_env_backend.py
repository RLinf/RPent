from __future__ import annotations

import dataclasses
import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BEHAVIOR_ROOT = REPO_ROOT / "robots" / "behavior"
pytestmark = pytest.mark.skipif(
    not BEHAVIOR_ROOT.is_dir(),
    reason="BEHAVIOR robot plugin has not landed in this worktree yet",
)


def _meta(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "task_name": "picking_up_trash",
        "task_language": "Put the soda cans in the kitchen trash can.",
        "activity_definition_id": 0,
        "activity_instance_id": 3,
        "public_seed": 3,
        "scene_model": "house_double_floor_lower",
        "max_episode_steps": 50_000,
    }
    values.update(overrides)
    return values


def _official_omni_config() -> dict[str, Any]:
    return {
        "env": {
            "action_frequency": 30.0,
            "rendering_frequency": 30.0,
            "physics_frequency": 120.0,
            "automatic_reset": False,
            "flatten_action_space": False,
            "flatten_obs_space": True,
            "external_sensors": {},
        },
        "render": {"viewer_width": 1280, "viewer_height": 720},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "house_double_floor_lower",
            "scene_file": {
                "metadata": {
                    "task": {"inst_to_name": {"agent.n.01_1": "robot_r1"}}
                },
                "init_info": {
                    "class_module": "omnigibson.scenes",
                    "class_name": "InteractiveTraversableScene",
                    "args": {},
                },
                "objects_info": {"init_info": {"robot_r1": {}}},
                "state": {
                    "pos": [0.0, 0.0, 0.0],
                    "ori": [0.0, 0.0, 0.0, 1.0],
                    "registry": {
                        "system_registry": {},
                        "object_registry": {"robot_r1": {}},
                    },
                },
            },
        },
        "robots": [
            {
                "type": "R1Pro",
                "name": "robot_r1",
                "proprio_obs": ["joint_qpos"],
                "controller_config": {"base": {"name": "BaseController"}},
            }
        ],
        "objects": [],
        "task": {
            "type": "BehaviorTask",
            "activity_name": "picking_up_trash",
            "activity_definition_id": 0,
            "activity_instance_id": 3,
            "online_object_sampling": False,
            "termination_config": {"max_steps": 50_000},
        },
        "wrapper": {"type": None},
    }


def _write_minimal_rlinf_tree(root: Path) -> None:
    env_config = root / "examples" / "embodiment" / "config" / "env"
    env_config.mkdir(parents=True)
    behavior_env = root / "rlinf" / "envs" / "behavior"
    behavior_env.mkdir(parents=True)
    (behavior_env / "behavior_env.py").write_text("", encoding="utf-8")
    (env_config / "behavior_r1pro.yaml").write_text(
        textwrap.dedent(
            """
            env_type: behavior
            total_num_envs: null
            auto_reset: true
            ignore_terminations: true
            use_fixed_reset_state_ids: true
            max_steps_per_rollout_epoch: 1
            max_episode_steps: 1
            skip_intermediate_obs_in_chunk: false
            num_env_subprocess: 8
            direct_omnigibson_env: false
            video_cfg:
              save_video: true
              info_on_video: true
              video_base_dir: stale
            omni_config:
              env:
                env_wrapper: stale
                automatic_reset: true
                flatten_obs_space: true
                flatten_action_space: true
              camera:
                head_resolution: [1, 1]
                wrist_resolution: [1, 1]
              task:
                type: BehaviorTask
                activity_name: stale_task
                activity_definition_id: 999
                activity_instance_id: 999
                activity_instance_dir: null
                instance_file_format: template
                instance_resample_mode: online
                online_object_sampling: true
                use_presampled_robot_pose: false
                termination_config:
                  max_steps: 1
              scene:
                scene_model: stale_scene
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _obs(state_value: float) -> dict[str, Any]:
    return {
        "main_images": np.full((2, 2, 3), 0.25, dtype=np.float32),
        "wrist_images": np.stack(
            [
                np.full((1, 2, 3), 10, dtype=np.uint8),
                np.full((1, 2, 3), 20, dtype=np.uint8),
            ],
            axis=0,
        ),
        "states": np.array([state_value] * 32, dtype=np.float32),
        "task_descriptions": ["fake task text"],
    }


def test_official_backend_reset_trace_is_disabled_by_default(tmp_path, monkeypatch, capsys):
    from omegaconf import OmegaConf
    from robots.behavior import official_env_backend as backend

    class FakeBehaviorEnv:
        def __init__(self, cfg, **kwargs):
            self.cfg = cfg
            self.kwargs = kwargs

        def reset_raw(self, *, env_idx: int):
            assert env_idx == 0
            return _obs(0.0), {"done": {"success": False}}

    monkeypatch.delenv(backend.RESET_TRACE_ENV, raising=False)
    subject = backend.OfficialBehaviorBackend(
        meta=_meta(),
        output_dir=tmp_path,
        behavior_env_cls=FakeBehaviorEnv,
        cfg=OmegaConf.create({"env_type": "behavior"}),
    )

    subject.reset()

    assert capsys.readouterr().out == ""


def test_official_backend_reset_trace_records_reset_raw_branch(
    tmp_path,
    monkeypatch,
    capsys,
):
    from omegaconf import OmegaConf
    from robots.behavior import official_env_backend as backend

    class FakeBehaviorEnv:
        def __init__(self, cfg, **kwargs):
            self.cfg = cfg
            self.kwargs = kwargs

        def reset_raw(self, *, env_idx: int):
            assert env_idx == 0
            return _obs(0.0), {"done": {"success": False}}

    monkeypatch.setenv(backend.RESET_TRACE_ENV, "1")
    subject = backend.OfficialBehaviorBackend(
        meta=_meta(),
        output_dir=tmp_path,
        behavior_env_cls=FakeBehaviorEnv,
        cfg=OmegaConf.create({"env_type": "behavior"}),
    )

    subject.reset()
    records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]

    assert [record["event"] for record in records] == [
        "official_behavior_backend.reset.enter",
        "official_behavior_backend._reset_raw.enter",
        "official_behavior_backend._reset_raw.exit",
        "official_behavior_backend.reset.exit",
    ]
    assert all(
        record["component"] == "OfficialBehaviorBackend"
        and record["schema_version"] == 1
        for record in records
    )
    assert records[1]["branch"] == "reset_raw"
    assert records[2]["branch"] == "reset_raw"
    assert records[2]["status"] == "ok"
    assert isinstance(records[2]["elapsed_s"], float)
    assert records[3]["status"] == "ok"
    assert records[3]["total_env_steps"] == 0
    assert isinstance(records[3]["elapsed_s"], float)


def test_official_backend_reset_trace_records_reset_fallback_branch(
    tmp_path,
    monkeypatch,
    capsys,
):
    from omegaconf import OmegaConf
    from robots.behavior import official_env_backend as backend

    class FakeBehaviorEnv:
        def __init__(self, cfg, **kwargs):
            self.cfg = cfg
            self.kwargs = kwargs

        def reset(self):
            return _obs(0.0), {"done": {"success": False}}

    monkeypatch.setenv(backend.RESET_TRACE_ENV, "1")
    subject = backend.OfficialBehaviorBackend(
        meta=_meta(),
        output_dir=tmp_path,
        behavior_env_cls=FakeBehaviorEnv,
        cfg=OmegaConf.create({"env_type": "behavior"}),
    )

    subject.reset()
    records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]

    assert [record["event"] for record in records] == [
        "official_behavior_backend.reset.enter",
        "official_behavior_backend._reset_raw.enter",
        "official_behavior_backend._reset_raw.exit",
        "official_behavior_backend.reset.exit",
    ]
    assert records[1]["branch"] == "reset_fallback"
    assert records[2]["branch"] == "reset_fallback"
    assert records[2]["status"] == "ok"
    assert isinstance(records[2]["elapsed_s"], float)
    assert records[3]["status"] == "ok"


def test_config_only_exact_official_uses_closed_config_not_tro_bootstrap(tmp_path):
    from omegaconf import OmegaConf
    from robots.behavior import official_env_backend as backend

    official = _official_omni_config()

    cfg = backend.build_behavior_env_config(
        {
            **_meta(),
            "omni_config_mode": backend.EXACT_OFFICIAL_CONFIG_MODE,
            "omni_config": official,
        },
        output_dir=tmp_path,
    )

    assert cfg.omni_config_mode == backend.EXACT_OFFICIAL_CONFIG_MODE
    assert cfg.use_fixed_reset_state_ids is False
    assert cfg.direct_omnigibson_env is True
    assert cfg.skip_intermediate_obs_in_chunk is True
    assert cfg.omni_config.task.termination_config.max_steps == 50_000
    for synthetic_field in (
        "activity_instance_dir",
        "instance_file_format",
        "instance_resample_mode",
        "use_presampled_robot_pose",
    ):
        assert synthetic_field not in cfg.omni_config.task

    overlay = OmegaConf.to_container(
        cfg.omni_config_effective_overlay,
        resolve=True,
        throw_on_missing=True,
    )
    assert set(overlay["changes"]) == {
        "env.flatten_obs_space",
        "task.termination_config.max_steps",
    }
    assert overlay["changes"]["env.flatten_obs_space"] == {
        "source": True,
        "effective": False,
    }
    assert overlay["changes"]["task.termination_config.max_steps"] == {
        "source": 50_000,
        "effective": 49_999,
    }


def test_vla_model_config_asset_id_resolves_existing_behavior_norm_stats() -> None:
    from omegaconf import OmegaConf
    from robots.behavior import vla_server
    from robots.behavior.policy_checkpoint import SHARED_POLICY_CHECKPOINT_PATH

    cfg = vla_server.build_model_config(SHARED_POLICY_CHECKPOINT_PATH)
    asset_id = OmegaConf.select(cfg, "openpi_data.assets.asset_id", default=None)

    assert cfg.openpi.config_name == "pi05_behavior"
    assert asset_id == "assets/behavior-1k/2025-challenge-demos"
    norm_stats_path = (
        Path(cfg.model_path)
        / asset_id
        / "norm_stats.json"
    )
    assert norm_stats_path.is_file()
    assert norm_stats_path.name == vla_server.NORM_STATS_REL.name
    assert norm_stats_path.relative_to(Path(cfg.model_path)) == vla_server.NORM_STATS_REL

    @dataclasses.dataclass(frozen=True)
    class FakeAssetsConfig:
        asset_id: str | None = None

    @dataclasses.dataclass(frozen=True)
    class FakeDataFactory:
        assets: Any = dataclasses.field(default_factory=FakeAssetsConfig)
        extra_delta_transform: bool = False
        extract_state_from_proprio: bool = False
        use_all_wrist_images: bool = False
        use_quantile_norm: bool = False

        def create(self) -> Any:
            return dataclasses.replace(
                FakeDataConfig(),
                asset_id=self.assets.asset_id,
            )

    @dataclasses.dataclass(frozen=True)
    class FakeDataConfig:
        asset_id: str | None = None

    actor_train_config_data = dataclasses.replace(
        FakeDataFactory(),
        assets=cfg.openpi_data.assets,
    )
    data_config = actor_train_config_data.create()

    assert data_config.asset_id == "assets/behavior-1k/2025-challenge-demos"
    assert Path(cfg.model_path, data_config.asset_id, "norm_stats.json").is_file()


def test_config_only_cached_tro_state_bootstrap_is_explicit(tmp_path, monkeypatch):
    from robots.behavior import official_env_backend as backend

    rlinf_root = tmp_path / "rlinf"
    activity_dir = tmp_path / "activity_instances"
    activity_dir.mkdir()
    bootstrap_template = (
        tmp_path
        / "house_double_floor_lower_task_picking_up_trash_0_0_template.json"
    )
    bootstrap_template.write_text("{}\n", encoding="utf-8")
    _write_minimal_rlinf_tree(rlinf_root)
    monkeypatch.setenv(backend.RLINF_ROOT_ENV, str(rlinf_root))

    cfg = backend.build_behavior_env_config(
        _meta(activity_instance_dir=str(activity_dir)),
        output_dir=tmp_path / "out",
    )

    assert cfg.seed == 3
    assert cfg.total_num_envs == 1
    assert cfg.use_fixed_reset_state_ids is False
    assert cfg.direct_omnigibson_env is True
    assert cfg.num_env_subprocess == 1
    assert cfg.skip_intermediate_obs_in_chunk is True
    assert cfg.omni_config.env.flatten_obs_space is False
    assert cfg.omni_config.env.automatic_reset is False
    assert cfg.omni_config.task.activity_name == "picking_up_trash"
    assert cfg.omni_config.task.activity_definition_id == 0
    assert cfg.omni_config.task.activity_instance_id == 3
    assert cfg.omni_config.task.activity_instance_dir == str(activity_dir.resolve())
    assert cfg.omni_config.task.instance_resample_mode == "disabled"
    assert cfg.omni_config.task.instance_file_format == "tro_state"
    assert cfg.omni_config.task.online_object_sampling is False
    assert cfg.omni_config.task.use_presampled_robot_pose is True
    assert cfg.omni_config.scene.scene_model == "house_double_floor_lower"
    assert cfg.omni_config.scene.scene_file == str(bootstrap_template)
    assert cfg.omni_config.scene.scene_instance is None


def test_config_only_tro_state_bootstrap_accepts_colocated_authorized_template(
    tmp_path,
    monkeypatch,
):
    from robots.behavior import official_env_backend as backend

    rlinf_root = tmp_path / "rlinf"
    activity_dir = tmp_path / "authorized_instance"
    activity_dir.mkdir()
    bootstrap_template = (
        activity_dir
        / "house_double_floor_lower_task_picking_up_trash_0_0_template.json"
    )
    bootstrap_template.write_text("{}\n", encoding="utf-8")
    _write_minimal_rlinf_tree(rlinf_root)
    monkeypatch.setenv(backend.RLINF_ROOT_ENV, str(rlinf_root))

    cfg = backend.build_behavior_env_config(
        _meta(activity_instance_dir=str(activity_dir)),
        output_dir=tmp_path / "out",
    )

    assert cfg.omni_config.task.activity_instance_dir == str(activity_dir.resolve())
    assert cfg.omni_config.scene.scene_file == str(bootstrap_template)
    assert cfg.omni_config.scene.scene_instance is None


def test_official_backend_accepts_rlinf_raw_observation_with_proprio_ndarray() -> None:
    from robots.behavior import official_env_backend as backend

    raw_obs = {
        "robot_r1": {
            "robot_r1:zed_link:Camera:0": {
                "rgb": np.full((2, 3, 4), 0.25, dtype=np.float32),
            },
            "robot_r1:left_realsense_link:Camera:0": {
                "rgb": np.full((1, 2, 3), 10, dtype=np.uint8),
            },
            "robot_r1:right_realsense_link:Camera:0": {
                "rgb": np.full((1, 2, 3), 20, dtype=np.uint8),
            },
            "robot_r1:proprio": np.arange(32, dtype=np.float32),
        }
    }

    obs = backend._normalize_single_observation(
        raw_obs,
        task_language="Put the soda cans in the kitchen trash can.",
    )

    assert obs["main_images"].shape == (2, 3, 3)
    assert obs["main_images"].dtype == np.uint8
    assert obs["wrist_images"].shape == (2, 1, 2, 3)
    assert obs["states"].shape == (32,)
    np.testing.assert_array_equal(obs["states"], np.arange(32, dtype=np.float32))
    assert obs["task_descriptions"] == "Put the soda cans in the kitchen trash can."


def test_fake_loader_bootstraps_backend_without_live_sim_and_latches_raw_success(
    tmp_path,
):
    from omegaconf import OmegaConf
    from robots.behavior import official_env_backend as backend

    class FakeBehaviorEnv:
        def __init__(self, cfg, **kwargs):
            self.cfg = cfg
            self.kwargs = kwargs
            self.actions: list[np.ndarray] = []

        def reset_raw(self, *, env_idx: int):
            assert env_idx == 0
            return _obs(0.0), {"done": {"success": False}}

        def step_raw(self, action, *, env_idx: int):
            assert env_idx == 0
            self.actions.append(np.asarray(action, dtype=np.float32))
            return (
                _obs(float(len(self.actions))),
                1.0,
                False,
                False,
                {"done": {"success": len(self.actions) == 2}},
            )

        def close(self):
            return None

    cfg = OmegaConf.create(
        {
            "env_type": "behavior",
            "skip_intermediate_obs_in_chunk": True,
            "omni_config": _official_omni_config(),
        }
    )
    subject = backend.OfficialBehaviorBackend(
        meta=_meta(),
        output_dir=tmp_path,
        behavior_env_cls=FakeBehaviorEnv,
        cfg=cfg,
    )

    obs, info = subject.reset()
    assert obs["main_images"].dtype == np.uint8
    assert obs["wrist_images"].shape == (2, 1, 2, 3)
    assert obs["states"].shape == (32,)
    assert info["_rpent"]["total_env_steps"] == 0
    assert subject.official_success_latched is False

    stepped, reward, terminated, truncated, info = subject.pi0_nav_pick_chunk_step(
        np.zeros((3, backend.ACTION_DIM), dtype=np.float32),
        chunk_index=7,
    )

    assert stepped is not None
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert subject.total_env_steps == 2
    assert subject.official_success_latched is True
    receipt = subject.official_success_receipt
    assert receipt is not None
    assert receipt["source"] == 'info["done"]["success"]'
    assert receipt["env_step"] == 2
    assert info["_rpent"]["pi0_nav_pick_monitor"] == {
        "chunk_index": 7,
        "requested_steps": 3,
        "executed_steps": 2,
        "stop_reason": "official_task_success",
        "success_step_in_chunk": 1,
        "total_env_steps": 2,
        "official_success_receipt": receipt,
    }
