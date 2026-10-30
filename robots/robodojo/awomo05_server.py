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

"""RPent RPC service for the official Awomo05 model and episode history."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from robots.robodojo.access import ARX_X5_JOINT_FIELDS, validate_rgb
from robots.robodojo.awomo05_client import REPLAN_STEPS
from rpent.robots.components.vla_facade_base import BaseVLAFacade


def model_paths(model_path: str | Path | None) -> dict[str, Path]:
    """Resolve and validate the official Awomo05 resource layout."""
    if not model_path:
        raise ValueError("Set --vla-model-path / AWOMO05_MODEL_PATH or --vla-endpoint")
    root = Path(model_path).expanduser().resolve()
    policy = root / "policy"
    xpolicy = policy / "XPolicyLab"
    adapter = xpolicy / "policy" / "Awomo05"
    weights = root / "checkpoints"
    paths = {
        "policy_dir": policy,
        "xpolicylab_dir": xpolicy,
        "config_path": adapter / "deploy.yml",
        "model_config": adapter / "model_config.yaml",
        "weights_dir": weights,
        "ckpt": weights / "model.pt",
        "dataset_stats_path": weights / "dataset_stats.json",
        "flux2_base_path": weights
        / "flux2-klein-base-4b"
        / "flux-2-klein-base-4b.safetensors",
        "ae_path": weights / "flux2-dev" / "ae.safetensors",
        "qwen_vl_path": weights / "Qwen3-VL-4B-Instruct",
    }
    for path in (
        *paths.values(),
        adapter / "model.py",
        policy / "env_cfg" / "arx_x5.yml",
        policy / "env_cfg" / "robot" / "_robot_info.json",
    ):
        if not path.exists():
            raise FileNotFoundError(f"Awomo05 resource missing: {path}")
    return paths


class Awomo05Facade(BaseVLAFacade):
    """Serve native Awomo05 history for one RPC session at a time."""

    def __init__(self, model) -> None:
        self._model = model
        self._owner: str | None = None
        self._next_step = 0
        self._episode_ready = False
        super().__init__(enable_sessions=True)

    def _register_rpc(self) -> None:
        self._rpc.update(
            {
                "vla.reset_session": self.reset_session,
                "vla.update_obs": self.update_obs,
                "vla.get_action": self.get_action,
                "vla.end_episode": self.end_episode,
            }
        )

    def reset_session(self, *, session_id: str) -> dict:
        """Acquire and reset the native model history for this episode."""
        if self._owner is not None and self._owner != session_id:
            raise RuntimeError("Policy history is owned by another session")
        self._owner = session_id
        self._episode_ready = False
        self._model.reset()
        self._next_step = 0
        self._episode_ready = True
        return {"ok": True}

    def _require_episode(self, session_id: str) -> None:
        if self._owner != session_id:
            raise RuntimeError("Call vla.reset_session to acquire policy history")
        if not self._episode_ready:
            raise RuntimeError(
                "Policy state is uncertain; reset the episode before use"
            )

    def _call_model(self, method: str, *args):
        try:
            return getattr(self._model, method)(*args)
        except Exception:
            # Native history may have advanced before the inference failed.
            self._episode_ready = False
            raise

    def update_obs(self, obs: dict, step: int, *, session_id: str) -> dict:
        """Validate observations before updating model history."""
        self._require_episode(session_id)
        if type(step) is not int or step != self._next_step:
            raise ValueError(f"Expected observation step {self._next_step}, got {step}")
        state = obs["state"]
        for key, size in ARX_X5_JOINT_FIELDS.items():
            value = np.asarray(state.get(key))
            if (
                value.shape != (size,)
                or not np.issubdtype(value.dtype, np.number)
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"Awomo05 {key} must contain {size} finite values")
        validate_rgb(obs)
        instruction = obs["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a nonempty task string")
        self._call_model("update_obs", obs)
        self._next_step += 1
        return {"ok": True}

    def get_action(self, *, session_id: str) -> list[dict]:
        """Infer a native joint action chunk from the current history."""
        self._require_episode(session_id)
        if self._next_step == 0:
            raise RuntimeError("An observation is required before inference")
        return self._call_model("get_action")

    def _on_session_drop(self, session_id: str) -> None:
        if self._owner == session_id:
            self._owner = None
            self._episode_ready = False

    def end_episode(self, *, session_id: str) -> dict:
        """Release this episode; the next owner must reset the native model."""
        self._on_session_drop(session_id)
        return {"ok": True}

    def close(self) -> None:
        """Release the process-owned native model."""
        self._model = None
        self._owner = None
        self._episode_ready = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()

    paths = model_paths(args.model_path)
    sys.path[:0] = [
        str(paths.pop("policy_dir")),
        str(paths.pop("xpolicylab_dir")),
    ]
    import yaml
    from XPolicyLab.policy.Awomo05.model import RAW_FPS, Model, _opt_float

    with paths.pop("config_path").open() as stream:
        cfg = yaml.safe_load(stream)
    cfg.update({key: str(value) for key, value in paths.items()})
    cfg.update(
        env_cfg_type="arx_x5",
        action_type="joint",
        input_color_order="bgr",
        replan_steps=REPLAN_STEPS,
        seed=0,
    )
    model = Model(cfg)
    history_cfg = model.run_cfg.data.train.get("history") or {}
    history_fps = _opt_float(model.model_cfg.get("history_fps")) or float(
        history_cfg.get("fps", RAW_FPS)
    )
    if model.history_grid is None or history_fps != 25:
        raise ValueError("Awomo05 requires enabled history at 25 Hz")
    if model.replan_steps != REPLAN_STEPS:
        raise ValueError(
            f"Awomo05 requires {REPLAN_STEPS} executed steps per action chunk"
        )
    facade = Awomo05Facade(model)
    try:
        facade.serve(
            transport="socket",
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
            session_sweep_s=60.0,
        )
    finally:
        facade.close()


if __name__ == "__main__":
    main()
