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

"""Run the official Awomo05 model with read-only metadata for endpoint validation.

Uses XPolicyLab's existing --config_path/--overrides arguments. Put its checkout
and parent on PYTHONPATH; run this module in the independent model environment.
"""

from __future__ import annotations

import asyncio
from typing import Any

if __package__:
    from .awomo05_contract import POLICY_METHODS, validate_model_meta
else:
    from awomo05_contract import POLICY_METHODS, validate_model_meta


class ModelWithMetadata:
    """Delegate all inference and state operations to the official model."""

    def __init__(self, model: Any, *, history_fps: float) -> None:
        self._model = model
        self._meta = validate_model_meta(
            {
                "policy_name": "Awomo05",
                "env_cfg_type": model.env_cfg_type,
                "action_type": model.action_type,
                "action_dim": model.expected_action_dim,
                "action_layout": "native_joint_dict",
                "camera_order": list(model.cameras),
                "input_color_order": model.input_color_order,
                "control_hz": history_fps,
                "history_enabled": model.history_grid is not None,
                "action_horizon": model.action_horizon,
                "replan_steps": model.replan_steps,
                "methods": [
                    name
                    for name in POLICY_METHODS
                    if callable(getattr(model, name, None))
                ],
            }
        )

    def get_model_meta(self) -> dict:
        return dict(self._meta)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def main() -> None:
    # Optional model/server dependencies stay entirely in this interpreter.
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from setup_policy_server import parse_args_and_config
    from XPolicyLab.policy.Awomo05.model import RAW_FPS, Model, _opt_float

    cfg = parse_args_and_config()
    if cfg.get("policy_name") != "Awomo05" or cfg.get("protocol") != "ws":
        raise ValueError("Use the official Awomo05 deploy.yml with protocol=ws")
    native = Model(cfg)
    history_cfg = native.run_cfg.data.train.get("history") or {}
    # Match the official model's optional-value parsing and history-grid rate.
    history_fps = _opt_float(native.model_cfg.get("history_fps")) or float(
        history_cfg.get("fps", RAW_FPS)
    )
    model = ModelWithMetadata(native, history_fps=history_fps)
    server = PolicyServer(
        model, PolicyServerConfig(host=cfg["host"], port=int(cfg["port"]))
    )
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
