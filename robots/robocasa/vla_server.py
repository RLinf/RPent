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

"""RoboCasa VLA server — loads RLDX model and exposes inference calls via RPC."""

import argparse
import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from rpent.robots.components.vla_facade_base import BaseVLAFacade
from rpent.utils.logging import get_logger

logger = get_logger("vla_server")

VLA_PROTOCOL_VERSION = 1
MAX_ACTION_HORIZON = 512
MAX_SESSION_ID_LENGTH = 192
ACTION_FIELDS = {
    "action.end_effector_position": 3,
    "action.end_effector_rotation": 3,
    "action.gripper_close": 1,
    "action.base_motion": 4,
    "action.control_mode": 1,
}
STATE_FIELDS = {
    "state.gripper_qpos": 2,
    "state.base_position": 3,
    "state.base_rotation": 4,
    "state.end_effector_position_relative": 3,
    "state.end_effector_rotation_relative": 4,
}
VIDEO_FIELDS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
ANNOTATION_FIELD = "annotation.human.task_description"
_SESSION_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(model_path):
    if not isinstance(model_path, (str, os.PathLike)) or not str(model_path).strip():
        raise TypeError("model_path must be a non-empty path or model id")
    requested = str(model_path)
    path = Path(requested).expanduser()
    if not path.exists():
        return {
            "kind": "remote",
            "requested": requested,
            "fingerprint": f"remote:{requested}",
        }
    if not path.is_dir():
        raise ValueError(f"model_path must be a checkpoint directory, got {path}")
    path = path.resolve()
    config_path = path / "config.json"
    index_path = path / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config is missing: {config_path}")
    files = {"config.json": _sha256_file(config_path)}
    total_size = None
    if index_path.is_file():
        files[index_path.name] = _sha256_file(index_path)
        try:
            with open(index_path, encoding="utf-8") as stream:
                index = json.load(stream)
            total_size = index.get("metadata", {}).get("total_size")
        except (OSError, TypeError, ValueError):
            total_size = None
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return {
        "kind": "local",
        "requested": requested,
        "resolved_path": str(path),
        "fingerprint": digest.hexdigest(),
        "files": files,
        "total_size": total_size,
    }


def _session_id(value):
    if not isinstance(value, str) or not value:
        raise TypeError("session_id must be a non-empty string")
    if len(value) > MAX_SESSION_ID_LENGTH or _SESSION_RE.fullmatch(value) is None:
        raise ValueError(
            "session_id must be at most 192 characters and contain only "
            "letters, digits, '_', '-', '.', or ':'"
        )
    return value


def _numeric_array(value, name, *, shape):
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


class RoboCasaVLAFacade(BaseVLAFacade):
    """Loads RLDX model and exposes inference-only RPC methods."""

    def __init__(self, model_path):
        super().__init__()
        self._runtime_id = uuid.uuid4().hex
        self._model_identity = _model_identity(model_path)
        from rldx.data.embodiment_tags import EmbodimentTag
        from rldx.eval.rollout_policy import create_rldx_sim_policy

        self.policy = create_rldx_sim_policy(
            model_path,
            EmbodimentTag.GENERAL_EMBODIMENT,
            "",
            None,
        )
        loaded_identity = _model_identity(model_path)
        if loaded_identity != self._model_identity:
            raise RuntimeError("checkpoint identity changed while loading the VLA")
        mod = self.policy.get_modality_config()
        self._vdi = np.asarray(mod["video"].delta_indices)
        if (
            self._vdi.ndim != 1
            or self._vdi.size == 0
            or not np.issubdtype(self._vdi.dtype, np.integer)
            or np.any(np.diff(self._vdi) <= 0)
            or int(self._vdi[-1]) != 0
        ):
            raise RuntimeError(f"invalid model video delta indices: {self._vdi!r}")
        self._vdi = self._vdi.astype(np.int64, copy=False)
        self._hist_maxlen = int(self._vdi.max() - self._vdi.min()) + 2
        self._warmed_up = False
        self._warmup()
        print(
            f"[vla_server] policy loaded; video_delta_indices={self._vdi.tolist()} "
            f"hist_maxlen={self._hist_maxlen}",
            flush=True,
        )

    def _register_rpc(self):
        super()._register_rpc()
        self._rpc["vla.get_modality_config"] = self.get_modality_config
        self._rpc["vla.predict"] = self.predict
        self._rpc["vla.reset_session"] = self.reset_session
        self._rpc["vla.get_runtime_info"] = self.get_runtime_info
        self._readonly_methods.update(
            {
                "vla.get_modality_config",
                "vla.get_runtime_info",
            }
        )

    def get_runtime_info(self):
        return {
            "protocol_version": VLA_PROTOCOL_VERSION,
            "runtime_id": self._runtime_id,
            "backend": "rldx",
            "model": self._model_identity,
            "warmed_up": self._warmed_up,
            "action_schema": {
                "name": "rldx.robocasa.action.v1",
                "batch_size": 1,
                "max_horizon": MAX_ACTION_HORIZON,
                "flat_dim": sum(ACTION_FIELDS.values()),
                "fields": [
                    {"name": name, "size": size} for name, size in ACTION_FIELDS.items()
                ],
            },
            "observation_schema": {
                "name": "rldx.robocasa.observation.v1",
                "batch_size": 1,
                "video_frames": int(self._vdi.size),
                "video_height": 256,
                "video_width": 256,
                "state_fields": dict(STATE_FIELDS),
                "video_fields": list(VIDEO_FIELDS),
                "annotation_field": ANNOTATION_FIELD,
            },
            "modality": self.get_modality_config(),
        }

    def get_modality_config(self):
        return {
            "video_delta_indices": self._vdi.tolist(),
            "hist_maxlen": self._hist_maxlen,
        }

    def _validate_observation(self, obs_dict):
        if not isinstance(obs_dict, Mapping):
            raise TypeError("obs_dict must be a mapping")
        for name, size in STATE_FIELDS.items():
            if name not in obs_dict:
                raise ValueError(f"obs_dict is missing {name!r}")
            _numeric_array(obs_dict[name], name, shape=(1, 1, size))
        video_shape = (1, int(self._vdi.size), 256, 256, 3)
        for name in VIDEO_FIELDS:
            if name not in obs_dict:
                raise ValueError(f"obs_dict is missing {name!r}")
            video = np.asarray(obs_dict[name])
            if video.shape != video_shape or video.dtype != np.uint8:
                raise ValueError(
                    f"{name} must be uint8 with shape {video_shape}, "
                    f"got dtype={video.dtype}, shape={video.shape}"
                )
        annotation = obs_dict.get(ANNOTATION_FIELD)
        if (
            not isinstance(annotation, (list, tuple))
            or len(annotation) != 1
            or not isinstance(annotation[0], str)
            or not annotation[0].strip()
        ):
            raise ValueError(f"{ANNOTATION_FIELD} must contain one non-empty string")

    @staticmethod
    def _validate_options(options):
        if not isinstance(options, Mapping):
            raise TypeError("options must be a mapping")
        unknown = set(options).difference({"session_ids", "reset_memory"})
        if unknown:
            raise ValueError(f"unsupported VLA options: {sorted(unknown)}")
        sessions = options.get("session_ids")
        resets = options.get("reset_memory")
        if not isinstance(sessions, (list, tuple)) or len(sessions) != 1:
            raise ValueError("options.session_ids must contain exactly one session id")
        if not isinstance(resets, (list, tuple)) or len(resets) != 1:
            raise ValueError("options.reset_memory must contain exactly one boolean")
        _session_id(sessions[0])
        if not isinstance(resets[0], (bool, np.bool_)):
            raise TypeError("options.reset_memory[0] must be boolean")

    @staticmethod
    def _validate_actions(actions):
        if not isinstance(actions, Mapping):
            raise TypeError("RLDX policy must return an action mapping")
        normalized = dict(actions)
        horizon = None
        for name, size in ACTION_FIELDS.items():
            if name not in actions:
                raise ValueError(f"RLDX action output is missing {name!r}")
            array = np.asarray(actions[name])
            if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != size:
                raise ValueError(
                    f"{name} must have shape (1, horizon, {size}), got {array.shape}"
                )
            if array.shape[1] <= 0 or array.shape[1] > MAX_ACTION_HORIZON:
                raise ValueError(f"{name} has invalid horizon {array.shape[1]}")
            if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
                array.dtype, np.bool_
            ):
                raise TypeError(f"{name} must contain numeric values")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
            if horizon is None:
                horizon = array.shape[1]
            elif array.shape[1] != horizon:
                raise ValueError("RLDX action fields have inconsistent horizons")
            normalized[name] = array
        return normalized, horizon

    def predict(self, obs_dict, options):
        # policy.get_action returns dict[str, np.ndarray] because RLDX's
        # PolicyRuntime._decode already .cpu().numpy()s torch internally, and
        # _NumpyEncoder (http_rpc) tags numpy arrays at JSON time. If you ever
        # bypass _decode (e.g. call the model forward directly), you must
        # .cpu().numpy() the result here — _NumpyEncoder raises on torch.Tensor.
        self._validate_observation(obs_dict)
        self._validate_options(options)
        actions, info = self.policy.get_action(obs_dict, options=dict(options))
        actions, _ = self._validate_actions(actions)
        return actions

    def _warmup(self):
        frames = int(self._vdi.size)
        obs = {
            name: np.zeros((1, 1, size), dtype=np.float32)
            for name, size in STATE_FIELDS.items()
        }
        obs["state.base_rotation"][0, 0, 3] = 1.0
        obs["state.end_effector_rotation_relative"][0, 0, 3] = 1.0
        obs.update(
            {
                name: np.zeros((1, frames, 256, 256, 3), dtype=np.uint8)
                for name in VIDEO_FIELDS
            }
        )
        obs[ANNOTATION_FIELD] = ["warm up the robocasa policy"]
        session_id = f"warmup:{self._runtime_id}"
        try:
            self.predict(
                obs,
                {"session_ids": [session_id], "reset_memory": [True]},
            )
        finally:
            self.policy.reset({"session_ids": [session_id]})
        self._warmed_up = True

    def reset_session(self, session_id):
        session_id = _session_id(session_id)
        self.policy.reset({"session_ids": [session_id]})
        return {"ok": True, "runtime_id": self._runtime_id, "session_id": session_id}


def main():
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        os.environ.setdefault("RLDX_ATTN_IMPL", "sdpa")

    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    p.add_argument("--model-path", required=True, help="RLDX checkpoint path")
    args = p.parse_args()

    if args.cuda_device is not None:
        # New RLDX create_rldx_sim_policy hardcodes device=0 internally and
        # does not accept a device argument. Map physical GPU to cuda:0
        # via CUDA_VISIBLE_DEVICES instead.
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is already set; overriding with --cuda-device=%s",
                prev,
                args.cuda_device,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    facade = RoboCasaVLAFacade(args.model_path)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
