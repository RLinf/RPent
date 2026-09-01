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

ACTION_DIM = 23
ACTION_HORIZON = 32
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


def _module_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_rlinf_roots() -> tuple[Path, ...]:
    explicit = os.environ.get(RLINF_ROOT_ENV)
    roots: list[Path] = []
    if explicit:
        roots.append(Path(explicit).expanduser())
    projects = _module_repo_root().parent
    roots.extend(
        [
            projects / "RLinf",
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = root.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            deduped.append(resolved)
    return tuple(deduped)


def discover_rlinf_root() -> Path:
    """Return the RLinf checkout that contains the official BehaviorEnv."""

    for root in _candidate_rlinf_roots():
        if (root / "rlinf" / "envs" / "behavior" / "behavior_env.py").is_file():
            return root
    searched = ", ".join(str(path) for path in _candidate_rlinf_roots())
    raise FileNotFoundError(
        "could not locate RLinf behavior_env.py; set "
        f"{RLINF_ROOT_ENV} to the RLinf checkout. searched: {searched}"
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
        instance_dir = Path(str(activity_dir)).expanduser().resolve()
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
    if not _raw_success(info):
        return None
    material = {
        "schema_version": 1,
        "source": 'info["done"]["success"]',
        "env_step": int(env_step),
        "raw_done": {"success": True},
    }
    return {
        **material,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                material,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
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
        monitor: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        info_dict = dict(_jsonable(info)) if isinstance(info, Mapping) else {}
        runtime = info_dict.get("_rpent")
        if not isinstance(runtime, dict):
            runtime = {}
        runtime["total_env_steps"] = int(self._total_env_steps)
        runtime["global_env_steps"] = int(self._total_env_steps)
        if monitor is not None:
            runtime["pi0_nav_pick_monitor"] = dict(_strict_public_json(monitor))
        if _raw_success(info_dict):
            self._official_success_latched = True
            receipt = _receipt_from_info(info_dict, env_step=self._total_env_steps)
            if receipt is not None:
                self._official_success_receipt = receipt
                runtime["official_success_receipt"] = dict(receipt)
                if isinstance(runtime.get("pi0_nav_pick_monitor"), dict):
                    runtime["pi0_nav_pick_monitor"]["official_success_receipt"] = dict(
                        receipt
                    )
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

    def pi0_nav_pick_chunk_step(
        self,
        actions: Any,
        *,
        chunk_index: int,
    ) -> tuple[dict[str, Any] | None, float, bool, bool, dict[str, Any]]:
        action_array = _validate_action_chunk(actions)
        last_obs: Any = None
        last_reward = 0.0
        terminated = False
        truncated = False
        last_info: dict[str, Any] = {}
        success_step: int | None = None
        executed_steps = 0
        stop_reason = "requested_actions_completed"

        for step_offset, action in enumerate(action_array):
            raw_obs, reward, step_terminated, step_truncated, info = self._step_one_raw(
                action
            )
            executed_steps = step_offset + 1
            self._total_env_steps += 1
            last_obs = raw_obs
            last_reward = float(reward)
            last_info = info
            terminated = bool(step_terminated)
            truncated = bool(step_truncated)
            if _raw_success(info):
                success_step = step_offset
                terminated = True
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
        if terminated or truncated:
            self._episode_ended = True
        monitor = {
            "chunk_index": int(chunk_index),
            "requested_steps": int(action_array.shape[0]),
            "executed_steps": int(executed_steps),
            "stop_reason": stop_reason,
            "success_step_in_chunk": success_step,
            "total_env_steps": int(self._total_env_steps),
        }
        info_out = self._note_info(last_info, monitor=monitor)
        return self._last_obs, last_reward, terminated, truncated, info_out

    def get_task_language(self) -> str:
        return str(self.identity["task_language"])

    def healthz(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "runtime": "behavior_official_env_backend",
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
        image = self.render_camera(camera)
        return {
            "camera": camera,
            "available": False,
            "rgb_shape": list(image.shape),
            "reason": (
                "RLinf BehaviorEnv RPC adapter exposes RGB/proprio only; "
                "calibration/depth are not exported"
            ),
        }

    def observe(self, camera: str = "head", **_kwargs: Any) -> dict[str, Any]:
        camera = _physical_camera(camera)
        image = self.render_camera(camera)
        payload = _png_bytes(image)
        frame_id = f"behavior-{self.total_env_steps}-{camera}"
        frame_payload = (
            {"_image_cam_bytes": payload}
            if camera == "head"
            else {"_image_wrist_bytes": payload}
        )
        return {
            "status": "ok",
            "camera": camera,
            "frame_id": frame_id,
            "step": self.total_env_steps,
            "_image_bytes": payload,
            **frame_payload,
            "frames": _write_frame_files(
                {camera: payload},
                output_dir=self.output_dir,
                group_id=frame_id,
            ),
            "info": self._last_info,
        }

    def get_prepared_motion_status(
        self,
        *,
        prepared_plan_id: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "status": "unknown",
            "prepared_plan_id": str(prepared_plan_id),
            "motion_available": (
                self._last_obs is not None
                and not self._closed
                and not self._episode_ended
                and not self._official_success_latched
            ),
            "prepared": None,
        }

    def finalize_paused_runtime(
        self,
        vla_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": "ok",
            "task_success": self.official_success_latched,
            "official_success_source": 'info["done"]["success"]',
            "official_success_receipt": self.official_success_receipt,
            "vla_status": _strict_public_json(vla_status),
            "total_env_steps": int(self.total_env_steps),
        }

    def _motion_unavailable(
        self, name: str, kwargs: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "status": "failed",
            "name": name,
            "primitive_success": False,
            "task_success": self.official_success_latched,
            "stop_reason": "motion_unavailable",
            "error": (
                f"{name} requires a reviewed motion adapter; this "
                "backend only supports reset/current_observation/pi0 chunk "
                "stepping and observation"
            ),
            "motion_available": False,
            "request": _strict_public_json(dict(kwargs)),
            "info": self._last_info,
        }

    def move_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._motion_unavailable("move_to", kwargs)

    def move_both_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._motion_unavailable("move_both_to", kwargs)

    def navigate_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._motion_unavailable("navigate_to", kwargs)

    def rotate_wrist(self, **kwargs: Any) -> dict[str, Any]:
        return self._motion_unavailable("rotate_wrist", kwargs)

    def open(self, **kwargs: Any) -> dict[str, Any]:
        return self._motion_unavailable("open", kwargs)

    def close(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs:
            return self._motion_unavailable("close", kwargs)
        if self._closed:
            return {"status": "ok", "closed": True, "already_closed": True}
        closer = getattr(self._env, "close", None)
        if callable(closer):
            closer()
        self._closed = True
        return {"status": "ok", "closed": True}

    def press(self, **kwargs: Any) -> dict[str, Any]:
        return self._motion_unavailable("press", kwargs)

    def save_robot_state_checkpoint(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "failed",
            "primitive_success": False,
            "task_success": self.official_success_latched,
            "stop_reason": "checkpoint_unavailable",
            "error": "official RLinf backend does not expose RPent robot checkpoints",
            "request": _strict_public_json(dict(kwargs)),
        }

    def pixel_to_world(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "failed",
            "primitive_success": False,
            "task_success": self.official_success_latched,
            "stop_reason": "calibration_unavailable",
            "error": "RGB-only RLinf observation does not expose depth/camera calibration",
            "request": _strict_public_json(dict(kwargs)),
        }


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


def create_backend(
    meta: Mapping[str, Any], output_dir: str | Path
) -> OfficialBehaviorBackend:
    """Factory used by ``RPENT_BEHAVIOR_ENV_BACKEND_FACTORY``."""

    return OfficialBehaviorBackend(meta=meta, output_dir=output_dir)


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "PHYSICAL_CAMERAS",
    "OfficialBehaviorBackend",
    "build_behavior_env_config",
    "create_backend",
    "discover_rlinf_root",
    "ensure_rlinf_import_path",
]
