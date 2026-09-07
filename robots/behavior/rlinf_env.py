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

"""Bundled official BEHAVIOR backend for the RPent env RPC server.

This module is intentionally independent from the historical RPent BEHAVIOR
runtime helpers.  It builds an RLinf ``BehaviorEnv`` config, owns the single
live env instance, and exposes the narrow duck-typed surface consumed by
``robots.behavior.env_server.BehaviorEnvFacade``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from robots.behavior.schemas import ENV_ACTION_SEGMENTS, RAW_PROPRIO_SEGMENTS
from robots.behavior.terminal_success import official_success_receipt_sha256
from rpent.utils.config import get_repo_root, get_rlinf_repo_path

ACTION_DIM = 23
ACTION_HORIZON = 32
GRIPPER_COMMAND_CONTROL_CYCLES = 15
GRIPPER_OPEN_COMMAND = 1.0
GRIPPER_CLOSE_COMMAND = -1.0
PHYSICAL_CAMERAS = ("head", "left_wrist", "right_wrist")
EXACT_OFFICIAL_CONFIG_MODE = "exact_official_v1"
EXACT_OFFICIAL_RUNTIME_SUPPORT_SCHEMA = (
    "rlinf.behavior.exact_official_runtime_support.v1"
)
EXACT_OFFICIAL_OVERLAY_SCHEMA = "rlinf.behavior.exact_official_overlay.v1"
EXACT_OFFICIAL_WRAPPER_SELECTOR = "official_rgb_v1"
RLINF_ROOT_ENV = "RPENT_RLINF_ROOT"
RLINF_ENV_CONFIG_ENV = "RPENT_BEHAVIOR_RLINF_ENV_CONFIG"
ACTIVITY_INSTANCE_DIR_ENV = "RPENT_BEHAVIOR_ACTIVITY_INSTANCE_DIR"
ACTIVITY_INSTANCE_FORMAT_ENV = "RPENT_BEHAVIOR_ACTIVITY_INSTANCE_FORMAT"
EXACT_CONFIG_ENV = "RPENT_BEHAVIOR_EXACT_OFFICIAL_CONFIG"
RESET_TRACE_ENV = "RLINF_BEHAVIOR_RESET_TRACE"
_COMPLETE_EXACT_FIELDS = {
    "omni_config_mode",
    "omni_config",
    "omni_config_semantic_sha256",
    "omni_config_runtime_support",
    "omni_config_runtime_support_sha256",
    "omni_config_effective_overlay",
    "omni_config_effective_overlay_sha256",
    "omni_config_effective_sha256",
}


def discover_rlinf_root() -> Path:
    """Return the RLinf checkout that contains the official BehaviorEnv."""

    root = (get_rlinf_repo_path() or (get_repo_root().parent / "RLinf")).resolve()
    if (root / "rlinf" / "envs" / "behavior" / "behavior_env.py").is_file():
        return root
    raise FileNotFoundError(
        "could not locate RLinf behavior_env.py; set "
        f"{RLINF_ROOT_ENV} to the RLinf checkout. searched: {root}"
    )


def ensure_rlinf_import_path() -> Path:
    """Put the selected RLinf checkout on sys.path and return it."""

    root = discover_rlinf_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_structured_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def _coerce_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _coerce_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return result


def _require_text(meta: Mapping[str, Any], field: str) -> str:
    value = meta.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"meta[{field!r}] must be a non-empty string")
    return value.strip()


def _task_identity(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_name": _require_text(meta, "task_name"),
        "task_language": _require_text(meta, "task_language"),
        "activity_definition_id": _coerce_nonnegative_int(
            meta.get("activity_definition_id"),
            field="activity_definition_id",
        ),
        "activity_instance_id": _coerce_nonnegative_int(
            meta.get("activity_instance_id"),
            field="activity_instance_id",
        ),
        "public_seed": _coerce_nonnegative_int(
            meta.get("public_seed", 0),
            field="public_seed",
        ),
        "scene_model": _require_text(meta, "scene_model"),
        "max_episode_steps": _coerce_positive_int(
            meta.get("max_episode_steps"),
            field="max_episode_steps",
        ),
    }


def _resolution(value: Any, default: tuple[int, int]) -> list[int]:
    if value is None:
        return [int(default[0]), int(default[1])]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("camera resolution must contain two positive integers")
    return [
        _coerce_positive_int(value[0], field="camera resolution"),
        _coerce_positive_int(value[1], field="camera resolution"),
    ]


def _exact_runtime_support(
    *,
    official: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    camera_cfg = (
        official.get("camera") if isinstance(official.get("camera"), Mapping) else {}
    )
    return {
        "schema_version": EXACT_OFFICIAL_RUNTIME_SUPPORT_SCHEMA,
        "source_profile_sha256": _canonical_json_sha256(official),
        "wrapper_selector": EXACT_OFFICIAL_WRAPPER_SELECTOR,
        "macro": {
            "use_gpu_dynamics": bool(meta.get("use_gpu_dynamics", False)),
            "headless": bool(meta.get("headless", True)),
            "enable_flatcache": bool(meta.get("enable_flatcache", True)),
            "enable_object_states": bool(meta.get("enable_object_states", True)),
            "enable_transition_rules": bool(meta.get("enable_transition_rules", True)),
            "render_viewer_camera": bool(meta.get("render_viewer_camera", False)),
            "use_numpy_controller_backend": bool(
                meta.get("use_numpy_controller_backend", True)
            ),
        },
        "camera": {
            "head_resolution": _resolution(
                meta.get("head_resolution", camera_cfg.get("head_resolution")),
                (720, 720),
            ),
            "wrist_resolution": _resolution(
                meta.get("wrist_resolution", camera_cfg.get("wrist_resolution")),
                (480, 480),
            ),
        },
    }


def _exact_overlay(
    official: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    official_copy = json.loads(
        json.dumps(official, ensure_ascii=False, allow_nan=False)
    )
    env_cfg = official_copy.setdefault("env", {})
    task_cfg = official_copy.setdefault("task", {})
    termination = task_cfg.setdefault("termination_config", {})
    source_max_steps = _coerce_positive_int(
        termination.get("max_steps"),
        field="task.termination_config.max_steps",
    )
    flatten_source = env_cfg.get("flatten_obs_space")
    if flatten_source is not True:
        raise ValueError(
            "exact official omni_config.env.flatten_obs_space must be True "
            "so RLinf can apply its reviewed runtime overlay"
        )
    overlay = {
        "schema_version": EXACT_OFFICIAL_OVERLAY_SCHEMA,
        "changes": {
            "env.flatten_obs_space": {"source": True, "effective": False},
            "task.termination_config.max_steps": {
                "source": source_max_steps,
                "effective": source_max_steps - 1,
            },
        },
    }
    effective = json.loads(
        json.dumps(official_copy, ensure_ascii=False, allow_nan=False)
    )
    effective["env"]["flatten_obs_space"] = False
    effective["task"]["termination_config"]["max_steps"] = source_max_steps - 1
    return overlay, effective


def _assert_official_identity(
    official: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> None:
    identity = _task_identity(meta)
    task_cfg = official.get("task")
    scene_cfg = official.get("scene")
    if not isinstance(task_cfg, Mapping) or not isinstance(scene_cfg, Mapping):
        raise ValueError("exact official config must contain task and scene mappings")
    mismatches = {
        "task.activity_name": (
            identity["task_name"],
            task_cfg.get("activity_name"),
        ),
        "task.activity_definition_id": (
            identity["activity_definition_id"],
            task_cfg.get("activity_definition_id"),
        ),
        "task.activity_instance_id": (
            identity["activity_instance_id"],
            task_cfg.get("activity_instance_id"),
        ),
        "scene.scene_model": (
            identity["scene_model"],
            scene_cfg.get("scene_model"),
        ),
    }
    bad = {
        key: {"expected": expected, "actual": actual}
        for key, (expected, actual) in mismatches.items()
        if actual != expected
    }
    if bad:
        raise ValueError(f"exact official config identity mismatch: {bad}")


def _exact_config_from_official(
    official: Mapping[str, Any],
    meta: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    official_dict = json.loads(
        json.dumps(official, ensure_ascii=False, allow_nan=False)
    )
    identity = _task_identity(meta)
    _assert_official_identity(official_dict, meta)
    support = dict(
        meta.get("omni_config_runtime_support") or {}
    ) or _exact_runtime_support(
        official=official_dict,
        meta=meta,
    )
    overlay = meta.get("omni_config_effective_overlay")
    effective: dict[str, Any] | None = None
    if isinstance(overlay, Mapping):
        effective = None
        overlay = json.loads(json.dumps(overlay, ensure_ascii=False, allow_nan=False))
    else:
        overlay, effective = _exact_overlay(official_dict)
    if effective is None:
        effective = json.loads(
            json.dumps(official_dict, ensure_ascii=False, allow_nan=False)
        )
        changes = dict(overlay["changes"])
        effective["env"]["flatten_obs_space"] = changes["env.flatten_obs_space"][
            "effective"
        ]
        effective["task"]["termination_config"]["max_steps"] = changes[
            "task.termination_config.max_steps"
        ]["effective"]
    return {
        "env_type": "behavior",
        "total_num_envs": 1,
        "auto_reset": False,
        "ignore_terminations": False,
        "use_rel_reward": True,
        "seed": identity["public_seed"],
        "group_size": 1,
        "use_fixed_reset_state_ids": False,
        "max_steps_per_rollout_epoch": identity["max_episode_steps"],
        "max_episode_steps": identity["max_episode_steps"],
        "skip_intermediate_obs_in_chunk": True,
        "num_env_subprocess": 1,
        "direct_omnigibson_env": True,
        "video_cfg": {
            "save_video": False,
            "info_on_video": True,
            "video_base_dir": str(output_dir / "video"),
        },
        "base_config_name": "r1pro_behavior",
        "use_eval_utils_cfg": False,
        "policy_wrapper": None,
        "omni_config_mode": EXACT_OFFICIAL_CONFIG_MODE,
        "omni_config": official_dict,
        "omni_config_semantic_sha256": str(
            meta.get("omni_config_semantic_sha256")
            or _canonical_json_sha256(official_dict)
        ),
        "omni_config_runtime_support": support,
        "omni_config_runtime_support_sha256": str(
            meta.get("omni_config_runtime_support_sha256")
            or _canonical_json_sha256(support)
        ),
        "omni_config_effective_overlay": overlay,
        "omni_config_effective_overlay_sha256": str(
            meta.get("omni_config_effective_overlay_sha256")
            or _canonical_json_sha256(overlay)
        ),
        "omni_config_effective_sha256": str(
            meta.get("omni_config_effective_sha256")
            or _canonical_json_sha256(effective)
        ),
        "action_trace_path": str(output_dir / "behavior_action_trace.jsonl"),
        "action_trace_interval": 1,
    }


def _load_exact_official_config(meta: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if meta.get("omni_config_mode") == EXACT_OFFICIAL_CONFIG_MODE and isinstance(
        meta.get("omni_config"),
        Mapping,
    ):
        return meta

    path_value = (
        meta.get("exact_official_config_path")
        or meta.get("official_omni_config_path")
        or os.environ.get(EXACT_CONFIG_ENV)
    )
    if not path_value:
        return None
    loaded = _read_structured_file(Path(str(path_value)).expanduser().resolve())
    if not isinstance(loaded, Mapping):
        raise ValueError("exact official config file must contain a mapping")
    return loaded


def _default_env_config_path(rlinf_root: Path, meta: Mapping[str, Any]) -> Path:
    path_value = meta.get("rlinf_env_config_path") or os.environ.get(
        RLINF_ENV_CONFIG_ENV
    )
    if path_value:
        return Path(str(path_value)).expanduser().resolve()
    return (
        rlinf_root
        / "examples"
        / "embodiment"
        / "config"
        / "env"
        / "behavior_r1pro.yaml"
    )


def _bootstrap_template_path(
    instance_dir: Path,
    *,
    scene_model: str,
    task_name: str,
    activity_definition_id: int,
) -> Path:
    """Resolve the full instance-0 scene used before applying a TRO delta.

    RLinf's ``ActivityInstanceLoader`` applies ``*_template-tro_state.json``
    only immediately before reset.  OmniGibson therefore needs a complete
    same-task template to construct the object scope first.  The official
    challenge dataset stores that bootstrap template beside the task-specific
    ``*_instances`` directory.
    """

    template_name = (
        f"{scene_model}_task_{task_name}_{activity_definition_id}_0_template.json"
    )
    candidates = (instance_dir / template_name, instance_dir.parent / template_name)
    template_path = next((path for path in candidates if path.is_file()), None)
    if template_path is None:
        raise FileNotFoundError(
            "BEHAVIOR bootstrap scene template not found: "
            + " or ".join(str(path) for path in candidates)
        )
    return template_path


def _resolve_activity_instance_dir(
    activity_dir: Path,
    *,
    scene_model: str,
    task_name: str,
    activity_definition_id: int,
    activity_instance_id: int,
) -> Path:
    """Resolve a task instance directory from either supported CLI shape.

    The public launcher accepts the downloaded challenge dataset root, while
    RLinf's loader requires the task-specific directory that directly contains
    the cached instance JSON files.  An already task-specific directory remains
    valid for callers that supply one explicitly.
    """

    instance_name = (
        f"{scene_model}_task_{task_name}_{activity_definition_id}_"
        f"{activity_instance_id}_template-tro_state.json"
    )
    task_dir_name = f"{scene_model}_task_{task_name}_instances"
    candidates = (
        activity_dir,
        activity_dir / task_dir_name,
        activity_dir / "scenes" / scene_model / "json" / task_dir_name,
    )
    instance_dir = next(
        (path for path in candidates if (path / instance_name).is_file()),
        None,
    )
    if instance_dir is None:
        raise FileNotFoundError(
            "BEHAVIOR activity instance not found: "
            + " or ".join(str(path / instance_name) for path in candidates)
        )
    return instance_dir


def _apply_default_config_identity(
    cfg: Any,
    *,
    identity: Mapping[str, Any],
    output_dir: Path,
    meta: Mapping[str, Any],
) -> Any:
    from omegaconf import OmegaConf

    cfg.env_type = "behavior"
    cfg.total_num_envs = 1
    cfg.auto_reset = False
    cfg.ignore_terminations = False
    cfg.use_fixed_reset_state_ids = False
    cfg.seed = int(identity["public_seed"])
    cfg.direct_omnigibson_env = True
    cfg.num_env_subprocess = 1
    cfg.max_episode_steps = int(identity["max_episode_steps"])
    cfg.max_steps_per_rollout_epoch = int(identity["max_episode_steps"])
    cfg.skip_intermediate_obs_in_chunk = True
    cfg.video_cfg.save_video = False
    cfg.video_cfg.video_base_dir = str(output_dir / "video")
    cfg.omni_config.env.env_wrapper = str(meta.get("env_wrapper") or "rgb")
    cfg.omni_config.env.flatten_obs_space = False
    cfg.omni_config.env.flatten_action_space = False
    cfg.omni_config.env.automatic_reset = False
    cfg.omni_config.task.activity_name = str(identity["task_name"])
    cfg.omni_config.task.activity_definition_id = int(
        identity["activity_definition_id"]
    )
    cfg.omni_config.task.activity_instance_id = int(identity["activity_instance_id"])
    cfg.omni_config.task.online_object_sampling = False
    cfg.omni_config.task.termination_config.max_steps = int(
        identity["max_episode_steps"]
    )
    cfg.omni_config.scene.scene_model = str(identity["scene_model"])

    activity_dir = meta.get("activity_instance_dir") or os.environ.get(
        ACTIVITY_INSTANCE_DIR_ENV
    )
    if activity_dir:
        instance_dir = _resolve_activity_instance_dir(
            Path(str(activity_dir)).expanduser().resolve(),
            scene_model=str(identity["scene_model"]),
            task_name=str(identity["task_name"]),
            activity_definition_id=int(identity["activity_definition_id"]),
            activity_instance_id=int(identity["activity_instance_id"]),
        )
        cfg.omni_config.task.activity_instance_dir = str(instance_dir)
        cfg.omni_config.task.instance_resample_mode = "disabled"
        instance_file_format = str(
            meta.get("activity_instance_file_format")
            or os.environ.get(ACTIVITY_INSTANCE_FORMAT_ENV)
            or "tro_state"
        )
        cfg.omni_config.task.instance_file_format = instance_file_format
        cfg.omni_config.task.use_presampled_robot_pose = bool(
            meta.get("use_presampled_robot_pose", True)
        )
        if instance_file_format == "tro_state":
            # A TRO-state file is a delta, not an OmniGibson scene template.
            # Bootstrap the same task's object scope from the official full
            # instance-0 template; ActivityInstanceLoader applies the selected
            # native instance immediately before the first reset.
            cfg.omni_config.scene.scene_file = str(
                _bootstrap_template_path(
                    instance_dir,
                    scene_model=str(identity["scene_model"]),
                    task_name=str(identity["task_name"]),
                    activity_definition_id=int(identity["activity_definition_id"]),
                )
            )
            cfg.omni_config.scene.scene_instance = None

    for key, default in (
        ("head_resolution", (720, 720)),
        ("wrist_resolution", (480, 480)),
    ):
        if meta.get(key) is not None:
            OmegaConf.update(
                cfg,
                f"omni_config.camera.{key}",
                _resolution(meta.get(key), default),
                merge=False,
            )

    cfg.action_trace_path = str(output_dir / "behavior_action_trace.jsonl")
    cfg.action_trace_interval = 1
    return cfg


def build_behavior_env_config(meta: Mapping[str, Any], output_dir: str | Path) -> Any:
    """Build the RLinf BehaviorEnv config without launching simulation.

    If an exact official config is supplied through ``meta`` or
    ``RPENT_BEHAVIOR_EXACT_OFFICIAL_CONFIG``, this returns an
    ``exact_official_v1`` RLinf config.  Otherwise it loads RLinf's canonical
    ``behavior_r1pro.yaml`` and applies the task/instance identity from RPent.
    """

    from omegaconf import OmegaConf

    output_path = Path(output_dir).expanduser().resolve()
    identity = _task_identity(meta)
    exact_loaded = _load_exact_official_config(meta)
    if exact_loaded is not None:
        if exact_loaded.get(
            "omni_config_mode"
        ) == EXACT_OFFICIAL_CONFIG_MODE and _COMPLETE_EXACT_FIELDS.issubset(
            exact_loaded
        ):
            cfg_dict = dict(exact_loaded)
            official = cfg_dict.get("omni_config")
            if not isinstance(official, Mapping):
                raise ValueError("exact official omni_config must be a mapping")
            _assert_official_identity(official, meta)
            cfg_dict.setdefault("seed", identity["public_seed"])
            cfg_dict.setdefault("max_episode_steps", identity["max_episode_steps"])
            cfg_dict.setdefault(
                "max_steps_per_rollout_epoch",
                identity["max_episode_steps"],
            )
            cfg_dict.setdefault("auto_reset", False)
            cfg_dict.setdefault("ignore_terminations", False)
            cfg_dict.setdefault("use_fixed_reset_state_ids", False)
            cfg_dict.setdefault("skip_intermediate_obs_in_chunk", True)
            cfg_dict.setdefault("num_env_subprocess", 1)
            cfg_dict.setdefault("direct_omnigibson_env", True)
            cfg_dict.setdefault(
                "video_cfg",
                {
                    "save_video": False,
                    "info_on_video": True,
                    "video_base_dir": str(output_path / "video"),
                },
            )
            cfg_dict.setdefault(
                "action_trace_path",
                str(output_path / "behavior_action_trace.jsonl"),
            )
            cfg_dict.setdefault("action_trace_interval", 1)
            return OmegaConf.create(cfg_dict)

        official = exact_loaded.get("omni_config", exact_loaded)
        if not isinstance(official, Mapping):
            raise ValueError("exact official omni_config must be a mapping")
        return OmegaConf.create(
            _exact_config_from_official(official, meta, output_path)
        )

    rlinf_root = ensure_rlinf_import_path()
    config_path = _default_env_config_path(rlinf_root, meta)
    if not config_path.is_file():
        raise FileNotFoundError(f"RLinf BEHAVIOR env config not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    return _apply_default_config_identity(
        cfg,
        identity=identity,
        output_dir=output_path,
        meta=meta,
    )


def _torch_to_numpy(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    return value


def _jsonable(value: Any) -> Any:
    value = _torch_to_numpy(value)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _strict_public_json(value: Any) -> Any:
    value = _torch_to_numpy(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _strict_public_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_public_json(item) for item in value]
    if isinstance(value, bytes):
        return {
            "format": "png",
            "data": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _reset_trace_enabled() -> bool:
    return os.environ.get(RESET_TRACE_ENV) == "1"


def _emit_reset_trace_marker(
    event: str,
    *,
    elapsed_s: float | None = None,
    **fields: Any,
) -> None:
    if not _reset_trace_enabled():
        return
    payload = {
        "schema_version": 1,
        "component": "OfficialBehaviorBackend",
        "event": event,
        **fields,
    }
    if elapsed_s is not None:
        payload["elapsed_s"] = float(elapsed_s)
    try:
        print(
            json.dumps(
                _strict_public_json(payload),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ),
            flush=True,
        )
    except Exception:
        # Tracing must never change reset / timeout / exception semantics.
        pass


def _image_uint8(value: Any) -> np.ndarray:
    arr = np.asarray(_torch_to_numpy(value))
    if arr.ndim != 3 or arr.shape[-1] not in {3, 4}:
        raise ValueError(f"image must be [H,W,3 or 4], got {arr.shape}")
    arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 1.0
        if max_value <= 1.0 + 1e-6:
            arr = arr * 255.0
    arr = np.rint(arr).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _first_batch(value: Any) -> Any:
    arr = np.asarray(_torch_to_numpy(value))
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _task_description(value: Any, default: str) -> str:
    value = _jsonable(value)
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return default
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _extract_raw_observation(raw_obs: Mapping[str, Any]) -> dict[str, Any]:
    main_image = None
    left_image = None
    right_image = None
    proprio = None
    for sensor_data in raw_obs.values():
        if not isinstance(sensor_data, Mapping):
            continue
        for key, value in sensor_data.items():
            if not isinstance(key, str):
                continue
            if "proprio" in key:
                # RLinf exposes proprio as a tensor / ndarray, whereas camera
                # observations are mappings containing an ``rgb`` value.
                proprio = value
            elif not isinstance(value, Mapping):
                continue
            elif "left_realsense_link:Camera:0" in key and "rgb" in value:
                left_image = value["rgb"]
            elif "right_realsense_link:Camera:0" in key and "rgb" in value:
                right_image = value["rgb"]
            elif "zed_link:Camera:0" in key and "rgb" in value:
                main_image = value["rgb"]
    if (
        main_image is None
        or left_image is None
        or right_image is None
        or proprio is None
    ):
        raise ValueError("raw BEHAVIOR observation lacks main/wrist RGB or proprio")
    return {
        "main_images": main_image,
        "wrist_images": np.stack(
            [_image_uint8(left_image), _image_uint8(right_image)],
            axis=0,
        ),
        "states": proprio,
    }


def _normalize_single_observation(
    obs: Mapping[str, Any], *, task_language: str
) -> dict[str, Any]:
    if "main_images" not in obs or "wrist_images" not in obs or "states" not in obs:
        obs = _extract_raw_observation(obs)

    main = _image_uint8(_first_batch(obs["main_images"]))
    wrists_value = _first_batch(obs["wrist_images"])
    wrists = np.asarray(_torch_to_numpy(wrists_value))
    if wrists.ndim == 5 and wrists.shape[0] == 1:
        wrists = wrists[0]
    if wrists.ndim != 4 or wrists.shape[0] != 2:
        raise ValueError(f"wrist_images must be [2,H,W,3], got {wrists.shape}")
    left = _image_uint8(wrists[0])
    right = _image_uint8(wrists[1])
    states = np.asarray(_first_batch(obs["states"]), dtype=np.float32)
    if states.ndim != 1:
        raise ValueError(f"states must be [raw_proprio_dim], got {states.shape}")
    if not np.isfinite(states).all():
        raise ValueError("states contains NaN or infinity")
    return {
        "main_images": main,
        "wrist_images": np.ascontiguousarray(np.stack([left, right], axis=0)),
        "states": np.ascontiguousarray(states.astype(np.float32, copy=False)),
        "task_descriptions": _task_description(
            obs.get("task_descriptions"),
            task_language,
        ),
        "extra_view_images": None,
    }


def _validate_action_chunk(actions: Any) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != ACTION_DIM or arr.shape[0] < 1:
        raise ValueError(f"BEHAVIOR actions must be [T,{ACTION_DIM}], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("BEHAVIOR actions contain NaN or infinity")
    return np.ascontiguousarray(arr)


def _raw_success(info: Any) -> bool:
    done = info.get("done") if isinstance(info, Mapping) else None
    value = done.get("success") if isinstance(done, Mapping) else None
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _receipt_from_info(
    info: Mapping[str, Any], *, env_step: int
) -> dict[str, Any] | None:
    if not _raw_success(info) or type(env_step) is not int or env_step < 0:
        return None
    material = {
        "schema_version": 1,
        "source": 'info["done"]["success"]',
        "env_step": env_step,
        "raw_done": {"success": True},
    }
    return {
        **material,
        "receipt_sha256": official_success_receipt_sha256(material),
    }


def _png_bytes(image: np.ndarray) -> bytes:
    import imageio.v2 as imageio

    buf = io.BytesIO()
    imageio.imwrite(buf, _image_uint8(image), format="png")
    return buf.getvalue()


def _write_frame_files(
    frames: Mapping[str, bytes],
    *,
    output_dir: Path,
    group_id: str,
) -> dict[str, str]:
    capture_dir = output_dir / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for camera, payload in frames.items():
        filename = f"{group_id}_{camera}.png"
        path = capture_dir / filename
        path.write_bytes(payload)
        try:
            paths[str(camera)] = str(path.relative_to(output_dir))
        except ValueError:
            paths[str(camera)] = str(path)
    return paths


class OfficialBehaviorBackend:
    """Duck-typed backend around one official RLinf BehaviorEnv."""

    def __init__(
        self,
        *,
        meta: Mapping[str, Any],
        output_dir: str | Path,
        behavior_env_cls: Any | None = None,
        cfg: Any | None = None,
    ) -> None:
        self.meta = dict(meta)
        self.identity = _task_identity(self.meta)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_obs: dict[str, Any] | None = None
        self._last_info: dict[str, Any] = {}
        self._last_raw_obs: Any = None
        self._closed = False
        self._episode_ended = False
        self._total_env_steps = 0
        self._official_success_latched = False
        self._official_success_receipt: dict[str, Any] | None = None
        self._camera_frames: dict[str, Any] = {}
        self._camera_frame_step = -1
        self._projections: dict[str, Any] = {}
        self._motion_planner = None
        self._gripper_latch = {
            "left": GRIPPER_OPEN_COMMAND,
            "right": GRIPPER_OPEN_COMMAND,
        }
        self.cfg = (
            cfg
            if cfg is not None
            else build_behavior_env_config(self.meta, self.output_dir)
        )
        if behavior_env_cls is None:
            ensure_rlinf_import_path()
            from rlinf.envs.behavior.behavior_env import BehaviorEnv

            behavior_env_cls = BehaviorEnv
        self._env = behavior_env_cls(
            self.cfg,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=SimpleNamespace(group_world_size=1),
            record_metrics=False,
        )

    @property
    def total_env_steps(self) -> int:
        return self._total_env_steps

    @property
    def official_success_latched(self) -> bool:
        return self._official_success_latched

    @property
    def official_success_receipt(self) -> dict[str, Any] | None:
        if self._official_success_receipt is None:
            return None
        return dict(self._official_success_receipt)

    def _wrap_raw_obs(self, raw_obs: Any) -> dict[str, Any]:
        if isinstance(raw_obs, Mapping) and {
            "main_images",
            "wrist_images",
            "states",
        }.issubset(raw_obs):
            return _normalize_single_observation(
                raw_obs,
                task_language=self.identity["task_language"],
            )
        wrapper = getattr(self._env, "_wrap_obs", None)
        if callable(wrapper):
            try:
                wrapped = wrapper([raw_obs])
                return _normalize_single_observation(
                    wrapped,
                    task_language=self.identity["task_language"],
                )
            except Exception:
                pass
        if isinstance(raw_obs, Mapping):
            return _normalize_single_observation(
                raw_obs,
                task_language=self.identity["task_language"],
            )
        raise TypeError("BEHAVIOR raw observation is not a mapping")

    def _note_info(
        self,
        info: Any,
        *,
        telemetry: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        info_dict = dict(_jsonable(info)) if isinstance(info, Mapping) else {}
        runtime = info_dict.get("_rpent")
        if not isinstance(runtime, dict):
            runtime = {}
        runtime["total_env_steps"] = int(self._total_env_steps)
        runtime["global_env_steps"] = int(self._total_env_steps)
        if telemetry is not None:
            for field in (
                "executed_steps",
                "stop_reason",
                "success_step_in_chunk",
            ):
                value = telemetry.get(field)
                if value is not None:
                    info_dict[field] = _strict_public_json(value)
        if _raw_success(info_dict):
            self._official_success_latched = True
            receipt = _receipt_from_info(info_dict, env_step=self._total_env_steps)
            if receipt is not None:
                self._official_success_receipt = receipt
                runtime["official_success_receipt"] = dict(receipt)
        info_dict["_rpent"] = runtime
        self._last_info = info_dict
        return info_dict

    def _reset_raw(self) -> tuple[Any, dict[str, Any]]:
        reset_raw = getattr(self._env, "reset_raw", None)
        env_reset = getattr(self._env, "env_reset", None)
        if callable(reset_raw):
            branch = "reset_raw"
        elif callable(env_reset):
            branch = "env_reset"
        else:
            branch = "reset_fallback"
        started_at = time.monotonic()
        _emit_reset_trace_marker(
            "official_behavior_backend._reset_raw.enter",
            branch=branch,
        )
        try:
            if callable(reset_raw):
                obs, info = reset_raw(env_idx=0)
            elif callable(env_reset):
                observations, infos = env_reset()
                if not isinstance(observations, (list, tuple)) or not observations:
                    raise TypeError(
                        "RLinf BehaviorEnv.env_reset returned no observations"
                    )
                obs = observations[0]
                info = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
            else:
                ret = self._env.reset()
                if isinstance(ret, (tuple, list)) and len(ret) == 2:
                    obs, info = ret
                else:
                    obs, info = ret, {}
            info_out = dict(_jsonable(info)) if isinstance(info, Mapping) else {}
        except Exception as exc:
            _emit_reset_trace_marker(
                "official_behavior_backend._reset_raw.exit",
                branch=branch,
                status="error",
                error_type=type(exc).__name__,
                error=str(exc),
                elapsed_s=time.monotonic() - started_at,
            )
            raise
        _emit_reset_trace_marker(
            "official_behavior_backend._reset_raw.exit",
            branch=branch,
            status="ok",
            info_is_mapping=isinstance(info, Mapping),
            observation_type=type(obs).__name__,
            elapsed_s=time.monotonic() - started_at,
        )
        return obs, info_out

    def _step_one_raw(
        self, action: np.ndarray
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        step_raw = getattr(self._env, "step_raw", None)
        if callable(step_raw):
            obs, reward, terminated, truncated, info = step_raw(action, env_idx=0)
            return (
                obs,
                float(np.asarray(_torch_to_numpy(reward)).reshape(-1)[0]),
                bool(np.asarray(_torch_to_numpy(terminated)).reshape(-1)[0]),
                bool(np.asarray(_torch_to_numpy(truncated)).reshape(-1)[0]),
                dict(_jsonable(info)) if isinstance(info, Mapping) else {},
            )

        env_chunk_step = getattr(self._env, "env_chunk_step", None)
        if callable(env_chunk_step):
            import torch

            chunk_action = torch.as_tensor(
                action.reshape(1, 1, ACTION_DIM), dtype=torch.float32
            )
            raw_obs_list, rewards, terms, truncs, infos = env_chunk_step(chunk_action)
            obs = raw_obs_list[-1][0] if raw_obs_list[-1] is not None else None
            info = infos[-1][0] if infos[-1] else {}
            return (
                obs,
                float(np.asarray(_torch_to_numpy(rewards[-1])).reshape(-1)[0]),
                bool(np.asarray(_torch_to_numpy(terms[-1])).reshape(-1)[0]),
                bool(np.asarray(_torch_to_numpy(truncs[-1])).reshape(-1)[0]),
                dict(_jsonable(info)) if isinstance(info, Mapping) else {},
            )

        chunk_step = getattr(self._env, "chunk_step", None)
        if callable(chunk_step):
            obs_list, rewards, terms, truncs, infos = chunk_step(
                action.reshape(1, 1, ACTION_DIM)
            )
            obs = obs_list[-1] if isinstance(obs_list, (list, tuple)) else obs_list
            info = infos[-1] if isinstance(infos, (list, tuple)) and infos else {}
            if isinstance(info, list) and info:
                info = info[0]
            return (
                obs,
                float(np.asarray(_torch_to_numpy(rewards)).reshape(-1)[-1]),
                bool(np.asarray(_torch_to_numpy(terms)).reshape(-1)[-1]),
                bool(np.asarray(_torch_to_numpy(truncs)).reshape(-1)[-1]),
                dict(_jsonable(info)) if isinstance(info, Mapping) else {},
            )

        raise RuntimeError("RLinf BehaviorEnv exposes no raw step interface")

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        started_at = time.monotonic()
        _emit_reset_trace_marker(
            "official_behavior_backend.reset.enter",
            total_env_steps_before=int(self._total_env_steps),
        )
        try:
            self._total_env_steps = 0
            self._episode_ended = False
            self._camera_frames = {}
            self._camera_frame_step = -1
            self._projections = {}
            self._gripper_latch = {
                "left": GRIPPER_OPEN_COMMAND,
                "right": GRIPPER_OPEN_COMMAND,
            }
            raw_obs, info = self._reset_raw()
            self._last_raw_obs = raw_obs
            self._last_obs = self._wrap_raw_obs(raw_obs)
            info_out = self._note_info(info)
        except Exception as exc:
            _emit_reset_trace_marker(
                "official_behavior_backend.reset.exit",
                status="error",
                error_type=type(exc).__name__,
                error=str(exc),
                elapsed_s=time.monotonic() - started_at,
            )
            raise
        _emit_reset_trace_marker(
            "official_behavior_backend.reset.exit",
            status="ok",
            total_env_steps=int(self._total_env_steps),
            observation_keys=sorted(self._last_obs),
            elapsed_s=time.monotonic() - started_at,
        )
        return self._last_obs, info_out

    def current_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._last_obs is None:
            raise RuntimeError("no BEHAVIOR observation is available before reset")
        return self._last_obs, self._last_info

    def _remember_gripper_commands(self, action: np.ndarray) -> None:
        for hand in ("left", "right"):
            segment = ENV_ACTION_SEGMENTS[f"{hand}_gripper"]
            value = float(np.asarray(action[segment], dtype=np.float32).reshape(-1)[0])
            if np.isfinite(value):
                self._gripper_latch[hand] = value

    def _latest_raw_proprio(self) -> np.ndarray:
        obs, _info = self.current_observation()
        raw = np.asarray(obs.get("states"), dtype=np.float32)
        required = max(segment.stop or 0 for segment in RAW_PROPRIO_SEGMENTS.values())
        if raw.ndim != 1 or raw.shape[0] < required:
            raise ValueError(
                "raw R1Pro proprio must be a vector with at least "
                f"{required} values, got {raw.shape}"
            )
        if not np.isfinite(raw).all():
            raise ValueError("raw R1Pro proprio contains NaN or infinity")
        return raw

    def _hold_action_from_current_proprio(self) -> np.ndarray:
        raw = self._latest_raw_proprio()
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[ENV_ACTION_SEGMENTS["base"]] = 0.0
        for segment_name in ("trunk", "left_arm", "right_arm"):
            action[ENV_ACTION_SEGMENTS[segment_name]] = raw[
                RAW_PROPRIO_SEGMENTS[segment_name]
            ]
        action[ENV_ACTION_SEGMENTS["left_gripper"]] = self._gripper_latch["left"]
        action[ENV_ACTION_SEGMENTS["right_gripper"]] = self._gripper_latch["right"]
        return _validate_action_chunk(action[None, :])[0]

    def _motion_error(
        self,
        name: str,
        kwargs: Mapping[str, Any],
        *,
        stop_reason: str,
        error: str,
    ) -> dict[str, Any]:
        return {
            "status": "failed",
            "name": name,
            "primitive_success": False,
            "task_success": self.official_success_latched,
            "stop_reason": stop_reason,
            "error": error,
            "request": _strict_public_json(dict(kwargs)),
            "info": self._last_info,
        }

    def _gripper_command(
        self,
        name: str,
        kwargs: Mapping[str, Any],
        *,
        command: float,
    ) -> dict[str, Any]:
        request = dict(kwargs)
        hand = request.get("hand")
        if hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")
        if "visual_hand_check" not in request:
            raise ValueError("visual_hand_check is required")
        if self.official_success_latched:
            return {
                "status": "skipped",
                "name": name,
                "primitive_success": False,
                "task_success": True,
                "stop_reason": "already_officially_successful",
                "request": _strict_public_json(request),
                "info": self._last_info,
            }
        if self._episode_ended:
            return self._motion_error(
                name,
                request,
                stop_reason="episode_ended",
                error="BEHAVIOR episode already terminated or truncated",
            )

        try:
            action = self._hold_action_from_current_proprio()
            action[ENV_ACTION_SEGMENTS[f"{hand}_gripper"]] = float(command)
            chunk = np.repeat(action[None, :], GRIPPER_COMMAND_CONTROL_CYCLES, axis=0)
            _obs, reward, terminated, truncated, info = self.chunk_step(
                chunk,
                return_all_frames=False,
            )
        except Exception as exc:
            return self._motion_error(
                name,
                request,
                stop_reason="error",
                error=str(exc),
            )
        executed_steps = int(info.get("executed_steps") or 0)
        stop_reason = str(info.get("stop_reason") or "requested_actions_completed")
        result: dict[str, Any] = {
            "status": "ok" if executed_steps > 0 else "failed",
            "name": name,
            "primitive_success": executed_steps > 0,
            "task_success": self.official_success_latched,
            "stop_reason": stop_reason,
            "hand": hand,
            "gripper_command": float(command),
            "requested_steps": GRIPPER_COMMAND_CONTROL_CYCLES,
            "executed_steps": executed_steps,
            "total_env_steps": int(self.total_env_steps),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "action_shape": [1, ACTION_DIM],
            "action_chunk_shape": [int(chunk.shape[0]), int(chunk.shape[1])],
            "hold_action_source": "raw_proprio_reordered_with_gripper_command_latches",
            "visual_hand_check": _strict_public_json(request["visual_hand_check"]),
            "visual_hand_check_verification": "not_verified",
            "request": _strict_public_json(request),
            "info": info,
            "_observation": self._last_obs,
        }
        if "release_visual_check" in request:
            result["release_visual_check"] = _strict_public_json(
                request["release_visual_check"]
            )
            result["release_visual_check_verification"] = "not_verified"
        if self.official_success_latched:
            result["official_success_receipt"] = self.official_success_receipt
        return result

    def step(
        self,
        action: Any,
    ) -> tuple[dict[str, Any] | None, float, bool, bool, dict[str, Any]]:
        array = np.asarray(action, dtype=np.float32)
        if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
            raise ValueError(f"BEHAVIOR action must be finite [{ACTION_DIM}]")
        return self.chunk_step(array[None, :], return_all_frames=False)

    def chunk_step(
        self,
        actions: Any,
        *,
        return_all_frames: bool = False,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        action_array = _validate_action_chunk(actions)
        last_obs: Any = None
        last_reward = 0.0
        terminated = False
        truncated = False
        last_info: dict[str, Any] = {}
        success_step: int | None = None
        executed_steps = 0
        stop_reason = "requested_actions_completed"
        frames: list[dict[str, Any]] = []

        for step_offset, action in enumerate(action_array):
            raw_obs, reward, step_terminated, step_truncated, info = self._step_one_raw(
                action
            )
            self._remember_gripper_commands(action)
            executed_steps = step_offset + 1
            self._total_env_steps += 1
            last_obs = raw_obs
            last_reward = float(reward)
            last_info = info
            terminated = bool(step_terminated)
            truncated = bool(step_truncated)
            if return_all_frames and raw_obs is not None:
                frames.append(self._wrap_raw_obs(raw_obs))
            if _raw_success(info):
                success_step = step_offset
                stop_reason = "official_task_success"
                break
            if terminated:
                stop_reason = "terminated"
                break
            if truncated:
                stop_reason = "truncated"
                break

        if last_obs is not None:
            self._last_raw_obs = last_obs
            self._last_obs = self._wrap_raw_obs(last_obs)
        if terminated or truncated or success_step is not None:
            self._episode_ended = True
        telemetry = {
            "executed_steps": int(executed_steps),
            "stop_reason": stop_reason,
            "success_step_in_chunk": success_step,
        }
        info_out = self._note_info(last_info, telemetry=telemetry)
        observation: Any = frames if return_all_frames else self._last_obs
        return observation, last_reward, terminated, truncated, info_out

    def get_task_language(self) -> str:
        return str(self.identity["task_language"])

    def healthz(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "runtime": "behavior_rlinf_env",
            "pid": os.getpid(),
            "total_env_steps": self.total_env_steps,
            "official_success_latched": self.official_success_latched,
        }

    def get_env_meta(self) -> dict[str, Any]:
        return dict(self.meta)

    def render_camera(self, camera_name: str = "head", **_kwargs: Any) -> np.ndarray:
        obs, _info = self.current_observation()
        camera = _physical_camera(camera_name)
        if camera == "head":
            return np.asarray(obs["main_images"], dtype=np.uint8)
        index = 0 if camera == "left_wrist" else 1
        return np.asarray(obs["wrist_images"][index], dtype=np.uint8)

    def get_camera_meta(
        self,
        camera_name: str = "head",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        camera = _physical_camera(camera_name)
        frame = self._get_camera_frames()[camera]
        return {
            "camera_name": camera,
            "available": True,
            "rgb_shape": list(frame["rgb"].shape),
            "rgb_dtype": str(frame["rgb"].dtype),
            "calibration_available": True,
            "depth_available": True,
            "intrinsic": frame["intrinsic"],
            "camera_to_world": frame["camera_to_world"],
        }

    def observe(self, camera: str = "head", **kwargs: Any) -> dict[str, Any]:
        from robots.behavior.schemas import (
            FRAME_REVIEW_ASSESSMENTS,
            validate_observe_request,
        )

        request = validate_observe_request(camera=camera, **kwargs)
        camera = _physical_camera(camera)
        if request.get("head_view", "center") != "center":
            return self._motion_error(
                "observe",
                request,
                stop_reason="head_view_unavailable",
                error="R1Pro has no movable head camera; use the current physical view",
            )
        review = request.get("frame_review")
        probe = request.get("depth_probe")
        if review is not None or probe is not None:
            value = review if review is not None else probe
            frame_id = f"behavior-{self.total_env_steps}-{camera}"
            if (
                not isinstance(value, Mapping)
                or value.get("frame_id") != frame_id
                or self._camera_frame_step != self.total_env_steps
            ):
                return self._motion_error(
                    "observe",
                    request,
                    stop_reason="stale_frame",
                    error="review/probe requires the current observed frame",
                )
            if review is not None:
                if review.get("assessment") not in FRAME_REVIEW_ASSESSMENTS:
                    raise ValueError("invalid frame_review assessment")
                return {
                    "status": "ok",
                    "primitive_success": True,
                    "frame_review": dict(review),
                    "verification": "planner_assessment_not_independently_verified",
                    "info": self._last_info,
                }
            if probe.get("assessment") != "target_point_visually_confirmed":
                raise ValueError("invalid depth_probe assessment")
            result = self.pixel_to_world(
                camera=camera, **{k: v for k, v in probe.items() if k != "assessment"}
            )
            result.pop("projection_id", None)
            return {
                **result,
                "depth_probe": dict(probe),
                "verification": "depth_measured_visual_assessment_not_verified",
                "info": self._last_info,
            }
        frames = self._get_camera_frames()
        payloads = {name: _png_bytes(frame["rgb"]) for name, frame in frames.items()}
        depths = {}
        for name, frame in frames.items():
            gray = np.nan_to_num(frame["depth"], nan=0, posinf=0, neginf=0)
            gray = np.rint(np.clip(gray / 5.0, 0, 1) * 255).astype(np.uint8)
            depths[name] = _png_bytes(np.repeat(gray[..., None], 3, axis=-1))
        frame_id = f"behavior-{self.total_env_steps}-{camera}"
        return {
            "status": "ok",
            "camera": camera,
            "paired_hand": request.get("paired_hand"),
            "frame_id": frame_id,
            "step": self.total_env_steps,
            "_image_bytes": payloads["head"],
            "_depth_image_bytes": depths["head"],
            "_image_left_wrist_bytes": payloads["left_wrist"],
            "_depth_left_wrist_bytes": depths["left_wrist"],
            "_image_right_wrist_bytes": payloads["right_wrist"],
            "_depth_right_wrist_bytes": depths["right_wrist"],
            "depth_display_range_m": [0, 5],
            "frames": _write_frame_files(
                payloads,
                output_dir=self.output_dir,
                group_id=frame_id,
            ),
            "info": self._last_info,
        }

    def move_to(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("hand") == "both":
            return self._move_both_hands_to(kwargs)
        return self._move_single_hand_to(kwargs)

    def _move_single_hand_to(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        return self._plan_motion("move_to", kwargs, {kwargs["hand"]: kwargs["target"]})

    def _move_both_hands_to(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        from robots.behavior.schemas import (
            validate_move_both_targets,
            validate_move_both_visual_hand_checks,
        )

        validate_move_both_visual_hand_checks(kwargs.get("visual_hand_checks"))
        return self._plan_motion(
            "move_to", kwargs, validate_move_both_targets(kwargs.get("targets"))
        )

    def _plan_motion(
        self, name: str, request: Mapping[str, Any], targets: dict
    ) -> dict:
        from scipy.spatial.transform import Rotation

        from robots.behavior.motion import BehaviorMotionPlanner, get_planning_state

        started = self.total_env_steps
        try:
            if self._episode_ended:
                return self._motion_error(
                    name,
                    request,
                    stop_reason="episode_ended",
                    error="episode has ended",
                )
            state = self._call_actor(get_planning_state)
            poses = {}
            for hand, target in targets.items():
                live = state["hands"][hand]
                position = np.array(live["position"], dtype=np.float64, copy=True)
                rotation = Rotation.from_quat(live["quaternion_xyzw"])
                orientation = live["quaternion_xyzw"]
                if name == "rotate_wrist":
                    angle = float(request["angle_deg"])
                    if not np.isfinite(angle) or abs(angle) > 180:
                        raise ValueError(
                            "angle_deg must be finite and within [-180,180]"
                        )
                    direction = request.get("direction", "counterclockwise")
                    if direction not in {"clockwise", "counterclockwise"}:
                        raise ValueError("invalid wrist rotation direction")
                    angle *= -1 if direction == "clockwise" else 1
                    orientation = (
                        rotation * Rotation.from_rotvec([0, 0, np.deg2rad(angle)])
                    ).as_quat()
                elif "projection_id" in target:
                    point = self._projection(target["projection_id"])
                    goal = np.asarray(point["world_xyz"])
                    delta = goal - position
                    standoff = float(target.get("standoff_m", 0))
                    if not np.isfinite(standoff) or standoff < 0:
                        raise ValueError("standoff_m must be finite and nonnegative")
                    position = (
                        goal - delta / max(np.linalg.norm(delta), 1e-12) * standoff
                    )
                else:
                    delta = np.asarray(target["delta_xyz"], dtype=np.float64)
                    if delta.shape != (3,) or not np.isfinite(delta).all():
                        raise ValueError("delta_xyz must be finite [3]")
                    if target["frame"] == "eef":
                        delta = rotation.apply(delta)
                    elif target["frame"] != "world":
                        raise ValueError("frame must be world or eef")
                    position += delta
                poses[hand] = {"position": position, "quaternion_xyzw": orientation}
            if self._motion_planner is None:
                self._motion_planner = BehaviorMotionPlanner()
            planned = self._motion_planner.plan(state, poses)
            if not planned["success"]:
                return self._motion_error(
                    name,
                    request,
                    stop_reason=planned["stop_reason"],
                    error=planned["details"],
                )
            positions = planned["positions"]
            times = np.arange(len(positions)) * planned["dt"]
            dt = state["control_dt"]
            if times[-1] > 30 or not np.isfinite(positions).all():
                raise ValueError(
                    "planned trajectory exceeds 30 seconds or is nonfinite"
                )
            sample_times = np.minimum(
                np.arange(int(np.ceil(times[-1] / dt)) + 1) * dt, times[-1]
            )
            positions = np.stack(
                [
                    np.interp(sample_times, times, positions[:, i])
                    for i in range(positions.shape[1])
                ],
                axis=1,
            )
            hold = self._hold_action_from_current_proprio()
            actions = np.repeat(hold[None, :], len(positions), axis=0)
            for i, joint_name in enumerate(planned["joint_names"]):
                hand, _, number = joint_name.partition("_arm_joint")
                actions[
                    :, ENV_ACTION_SEGMENTS[f"{hand}_arm"].start + int(number) - 1
                ] = positions[:, i]
            reason = "trajectory_completed"
            for offset in range(0, len(actions), 32):
                _, _, terminated, truncated, info = self.chunk_step(
                    actions[offset : offset + 32]
                )
                if self.official_success_latched or terminated or truncated:
                    reason = str(info["stop_reason"])
                    break
            # Terminal observations are already captured by chunk_step. Never
            # issue another actor query after official success or truncation.
            final = self._get_motion_state() if not self._episode_ended else None
            errors = {
                hand: float(
                    np.linalg.norm(
                        np.asarray(final["hands"][hand]["position"])
                        - target["position"]
                    )
                )
                for hand, target in poses.items()
                if final is not None
            }
            angles = {
                hand: float(
                    (
                        Rotation.from_quat(
                            final["hands"][hand]["quaternion_xyzw"]
                        ).inv()
                        * Rotation.from_quat(target["quaternion_xyzw"])
                    ).magnitude()
                )
                for hand, target in poses.items()
                if final is not None
            }
            reached = (
                final is not None
                and all(x <= 0.01 for x in errors.values())
                and all(x <= 0.1 for x in angles.values())
            )
            succeeded = self.official_success_latched or (
                reason == "trajectory_completed" and reached
            )
            if reason == "trajectory_completed" and not reached:
                reason = "tracking_error"
            return {
                "status": "ok" if succeeded else "failed",
                "name": name,
                "primitive_success": succeeded,
                "task_success": self.official_success_latched,
                "stop_reason": reason,
                "executed_steps": self.total_env_steps - started,
                "total_env_steps": self.total_env_steps,
                "position_error_m": errors,
                "orientation_error_rad": angles,
                "_observation": self._last_obs,
                "visual_hand_check_verification": "not_verified",
                "request": _strict_public_json(request),
                "info": self._last_info,
            }
        except Exception as exc:
            result = self._motion_error(
                name, request, stop_reason="error", error=str(exc)
            )
            result["executed_steps"] = self.total_env_steps - started
            return result

    def _projection(self, projection_id: str) -> dict:
        if (
            self._camera_frame_step != self.total_env_steps
            or projection_id not in self._projections
        ):
            raise ValueError("projection is not from the current observed frame")
        return self._projections[projection_id]

    def navigate_to(self, **kwargs: Any) -> dict[str, Any]:
        from scipy.spatial.transform import Rotation

        from robots.behavior.motion import get_planning_state, navigation_collision
        from robots.behavior.schemas import validate_relative_navigation_motion

        started = self.total_env_steps
        try:
            if self._episode_ended:
                return self._motion_error(
                    "navigate_to",
                    kwargs,
                    stop_reason="episode_ended",
                    error="episode has ended",
                )
            state = self._call_actor(get_planning_state)
            position = np.asarray(state["base_position"])
            yaw = Rotation.from_quat(state["base_quaternion_xyzw"]).as_euler("xyz")[2]
            target, target_yaw = position.copy(), yaw
            if "relative_motion" in kwargs:
                motion = validate_relative_navigation_motion(kwargs["relative_motion"])
                if motion["kind"] == "translation":
                    distance = motion["distance_m"] * (
                        1 if motion["direction"] == "forward" else -1
                    )
                    target[:2] += distance * np.array([np.cos(yaw), np.sin(yaw)])
                else:
                    target_yaw += np.deg2rad(motion["angle_deg"]) * (
                        1 if motion["direction"] == "left" else -1
                    )
            else:
                check = kwargs.get("navigation_visual_check")
                if (
                    not isinstance(check, Mapping)
                    or check.get("camera") != "head"
                    or check.get("frame_id") != f"behavior-{self.total_env_steps}-head"
                    or check.get("assessment") != "navigation_target_visually_confirmed"
                ):
                    raise ValueError(
                        "navigation_visual_check must confirm the head target"
                    )
                goal = np.asarray(
                    self._projection(kwargs["projection_id"])["world_xyz"]
                )
                standoff = float(kwargs.get("standoff_m", 0.85))
                if not 0.45 <= standoff <= 1.5:
                    raise ValueError("standoff_m must be within [0.45,1.5]")
                delta = goal[:2] - position[:2]
                length = np.linalg.norm(delta)
                target[:2] += delta / max(length, 1e-12) * max(0, length - standoff)
                target_yaw = np.arctan2(delta[1], delta[0])
            obstacle = navigation_collision(state, target)
            if obstacle:
                return self._motion_error(
                    "navigate_to",
                    kwargs,
                    stop_reason="collision",
                    error=f"swept footprint intersects {obstacle}",
                )
            dt = state["control_dt"]
            reason = "duration_limit"
            for _ in range(int(np.ceil(30 / dt))):
                live = state["base_position"]
                yaw = Rotation.from_quat(state["base_quaternion_xyzw"]).as_euler("xyz")[
                    2
                ]
                delta = target[:2] - np.asarray(live)[:2]
                angle = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
                if np.linalg.norm(delta) < 0.01 and abs(angle) < np.deg2rad(1):
                    reason = "target_reached"
                    break
                world_velocity = delta * min(
                    2.0, 0.2 / max(np.linalg.norm(delta), 1e-12)
                )
                local_velocity = (
                    np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
                    @ world_velocity
                )
                action = self._hold_action_from_current_proprio()
                # RLinf base controller scales normalized x/y commands by 0.75 m/s.
                action[:2] = local_velocity / 0.75
                action[2] = np.clip(2 * angle, -0.4, 0.4)
                _, _, terminated, truncated, info = self.chunk_step(action[None, :])
                if self.official_success_latched or terminated or truncated:
                    reason = str(info["stop_reason"])
                    break
                state = self._call_actor(get_planning_state)
                obstacle = navigation_collision(state, target)
                if obstacle:
                    reason = "collision"
                    break
            if not self._episode_ended:
                _, _, terminated, truncated, info = self.chunk_step(
                    self._hold_action_from_current_proprio()[None, :]
                )
                if self.official_success_latched or terminated or truncated:
                    reason = str(info["stop_reason"])
            succeeded = self.official_success_latched or reason == "target_reached"
            return {
                "status": "ok" if succeeded else "failed",
                "name": "navigate_to",
                "primitive_success": succeeded,
                "_observation": self._last_obs,
                "task_success": self.official_success_latched,
                "stop_reason": reason,
                "executed_steps": self.total_env_steps - started,
                "total_env_steps": self.total_env_steps,
                "request": _strict_public_json(kwargs),
                "info": self._last_info,
            }
        except Exception as exc:
            # If a read/planning error followed a base command, release that
            # command through the same monitored action channel before returning.
            brake_error = None
            if self.total_env_steps > started and not self._episode_ended:
                try:
                    self.chunk_step(self._hold_action_from_current_proprio()[None, :])
                except Exception as brake_exc:
                    brake_error = str(brake_exc)
            result = self._motion_error(
                "navigate_to", kwargs, stop_reason="error", error=str(exc)
            )
            result["executed_steps"] = self.total_env_steps - started
            if brake_error is not None:
                result["brake_error"] = brake_error
            return result

    def rotate_wrist(self, **kwargs: Any) -> dict[str, Any]:
        check = kwargs.get("visual_hand_check")
        if (
            not isinstance(check, Mapping)
            or set(check) != {"camera", "frame_id", "selected_hand", "assessment"}
            or check.get("camera") not in PHYSICAL_CAMERAS
            or not isinstance(check.get("frame_id"), str)
            or not check.get("frame_id")
            or check.get("assessment") != "selected_hand_visually_confirmed"
            or check.get("selected_hand") != kwargs.get("hand")
        ):
            raise ValueError("visual_hand_check must identify the selected hand")
        return self._plan_motion("rotate_wrist", kwargs, {kwargs["hand"]: {}})

    def open(self, **kwargs: Any) -> dict[str, Any]:
        return self._gripper_command("open", kwargs, command=GRIPPER_OPEN_COMMAND)

    def close(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs:
            return self._gripper_command("close", kwargs, command=GRIPPER_CLOSE_COMMAND)
        if self._closed:
            return {"status": "ok", "closed": True, "already_closed": True}
        if self._motion_planner is not None:
            self._motion_planner.close()
        closer = getattr(self._env, "close", None)
        if callable(closer):
            closer()
        self._closed = True
        return {"status": "ok", "closed": True}

    def press(self, **kwargs: Any) -> dict[str, Any]:
        """Press along the currently aligned hand axis, at most 2 cm.

        Differential IK keeps orientation fixed. This local contact motion does
        not choose a button or claim official success from contact alone.
        """
        request = dict(kwargs)
        hand = request.get("hand")
        if hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")
        check = request.get("visual_hand_check")
        if not isinstance(check, Mapping) or check.get("selected_hand") != hand:
            raise ValueError("visual_hand_check must identify the selected hand")
        duration = float(request.get("duration_s", 1.0))
        if not np.isfinite(duration) or not 0.0 < duration <= 10.0:
            raise ValueError("duration_s must be finite and in (0, 10]")
        if self._episode_ended or self.official_success_latched:
            return self._motion_error(
                "press", request, stop_reason="episode_ended", error="episode has ended"
            )

        started = self.total_env_steps
        try:
            state = self._get_motion_state()
            initial = state["hands"][hand]
            origin = np.asarray(initial["position"], dtype=np.float64)
            direction = np.asarray(initial["approach_direction"], dtype=np.float64)
            dt = float(state["control_dt"])
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError("invalid environment control timestep")
            target = origin + direction * 0.02
            stop_reason = "duration_limit"
            contacts = []
            travel = 0.0
            for _ in range(int(np.ceil(duration / dt))):
                live = state["hands"][hand]
                position = np.asarray(live["position"], dtype=np.float64)
                travel = float(np.linalg.norm(position - origin))
                contacts = live["contacts"]
                if contacts:
                    stop_reason = "contact"
                    break
                if travel >= 0.02:
                    stop_reason = "travel_limit"
                    break
                error = target - position
                if np.linalg.norm(error) <= 0.001:
                    stop_reason = "target_reached"
                    break
                delta = error * min(1.0, 0.02 * dt / np.linalg.norm(error))
                jacobian = np.asarray(live["jacobian"], dtype=np.float64)
                if jacobian.shape != (6, 7) or not np.isfinite(jacobian).all():
                    raise ValueError("expected a finite [6,7] arm Jacobian")
                twist = np.concatenate((delta, np.zeros(3)))
                dq = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + 1e-4 * np.eye(6), twist
                )
                # Cap joint speed at 0.5 rad/s; preserve unselected joints/latches.
                dq *= min(1.0, 0.5 * dt / max(float(np.max(np.abs(dq))), 1e-12))
                q = np.asarray(live["joint_positions"]) + dq
                if np.any(q < live["joint_lower_limits"]) or np.any(
                    q > live["joint_upper_limits"]
                ):
                    stop_reason = "joint_limit"
                    break
                action = self._hold_action_from_current_proprio()
                action[ENV_ACTION_SEGMENTS[f"{hand}_arm"]] = q
                _, _, terminated, truncated, info = self.chunk_step(action[None, :])
                if self.official_success_latched or terminated or truncated:
                    stop_reason = str(info["stop_reason"])
                    break
                state = self._get_motion_state()
            live = state["hands"][hand]
            travel = float(np.linalg.norm(np.asarray(live["position"]) - origin))
            succeeded = stop_reason in {
                "contact",
                "target_reached",
                "official_task_success",
            }
            return {
                "status": "ok" if succeeded else "failed",
                "name": "press",
                "hand": hand,
                "_observation": self._last_obs,
                "primitive_success": succeeded,
                "task_success": self.official_success_latched,
                "stop_reason": stop_reason,
                "executed_steps": self.total_env_steps - started,
                "total_env_steps": self.total_env_steps,
                "travel_m": travel,
                "travel_limit_m": 0.02,
                "contacts": contacts,
                "contact_verification": "any_non_robot_contact_not_button_verified",
                "visual_hand_check": _strict_public_json(check),
                "visual_hand_check_verification": "not_verified",
                "request": _strict_public_json(request),
                "info": self._last_info,
            }
        except Exception as exc:
            result = self._motion_error(
                "press", request, stop_reason="error", error=str(exc)
            )
            result["executed_steps"] = self.total_env_steps - started
            return result

    def _get_motion_state(self) -> dict[str, Any]:
        from robots.behavior.motion import get_motion_state

        # RLinf owns the OG actor; query it on its existing serial execution lane.
        return self._call_actor(get_motion_state)

    def pixel_to_world(self, **kwargs: Any) -> dict[str, Any]:
        camera = _physical_camera(kwargs.get("camera"))
        expected = f"behavior-{self.total_env_steps}-{camera}"
        if (
            kwargs.get("frame_id") != expected
            or self._camera_frame_step != self.total_env_steps
        ):
            return self._motion_error(
                "pixel_to_world",
                kwargs,
                stop_reason="stale_frame",
                error="observe the current frame before projection",
            )
        frame = self._camera_frames[camera]
        u, v = kwargs["u"], kwargs["v"]
        depth = frame["depth"]
        if (
            type(u) is not int
            or type(v) is not int
            or not 0 <= u < depth.shape[1]
            or not 0 <= v < depth.shape[0]
        ):
            raise ValueError("pixel must be an integer inside the observed image")
        window = kwargs.get("depth_window_px", 7)
        if type(window) is not int or not 1 <= window <= 31:
            raise ValueError("depth_window_px must be in [1,31]")
        radius = window // 2
        samples = depth[
            max(0, v - radius) : v + radius + 1, max(0, u - radius) : u + radius + 1
        ]
        valid = samples[np.isfinite(samples) & (samples > 0)]
        if not valid.size:
            return self._motion_error(
                "pixel_to_world",
                kwargs,
                stop_reason="invalid_depth",
                error="no finite positive depth at this pixel",
            )
        z = float(np.median(valid))
        k = frame["intrinsic"]
        # USD cameras face -Z, +Y up; image rows increase downwards.
        optical = np.array(
            [(u - k[0, 2]) * z / k[0, 0], -(v - k[1, 2]) * z / k[1, 1], -z, 1.0]
        )
        xyz = (frame["camera_to_world"] @ optical)[:3]
        value = {
            "camera": camera,
            "frame_id": expected,
            "u": u,
            "v": v,
            "world_xyz": xyz.tolist(),
            "depth_m": z,
        }
        projection_id = _canonical_json_sha256(value)
        self._projections[projection_id] = value
        return {
            "status": "ok",
            "primitive_success": True,
            "task_success": self.official_success_latched,
            "projection_id": projection_id,
            **value,
        }

    def _get_camera_frames(self) -> dict[str, Any]:
        from robots.behavior.motion import get_camera_observation

        if self._camera_frame_step != self.total_env_steps:
            self._camera_frames = self._call_actor(get_camera_observation)
            self._camera_frame_step = self.total_env_steps
            self._projections = {}
        return self._camera_frames

    def _call_actor(self, function):
        import ray

        pool = self._env.pool
        index = self._env.pool_offset
        return ray.get(
            pool.env_processes[index % pool.num_env_subprocess].__ray_call__.remote(
                function, index // pool.num_env_subprocess
            )
        )


def _physical_camera(value: Any) -> str:
    camera = str(value or "head")
    aliases = {
        "main": "head",
        "zed": "head",
        "left": "left_wrist",
        "right": "right_wrist",
    }
    camera = aliases.get(camera, camera)
    if camera not in PHYSICAL_CAMERAS:
        raise ValueError("camera must be head, left_wrist, or right_wrist")
    return camera


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "PHYSICAL_CAMERAS",
    "OfficialBehaviorBackend",
    "build_behavior_env_config",
    "discover_rlinf_root",
    "ensure_rlinf_import_path",
]
