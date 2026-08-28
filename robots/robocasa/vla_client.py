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

"""RoboCasa VLA client — thin RPC layer over the VLA server."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from rpent.robots.components.vla_client_base import BaseVLAClient

VLA_PROTOCOL_VERSION = 1
EXPECTED_ACTION_FIELDS = {
    "action.end_effector_position": 3,
    "action.end_effector_rotation": 3,
    "action.gripper_close": 1,
    "action.base_motion": 4,
    "action.control_mode": 1,
}


class RoboCasaVLAClient(BaseVLAClient):
    _TIMEOUT_S = BaseVLAClient._TIMEOUT_S

    def __init__(
        self,
        client,
        *,
        expected_model_fingerprint: str | None = None,
        require_verified_model: bool = False,
    ):
        super().__init__(client)
        self.runtime_info = self.get_runtime_info()
        self._validate_runtime_info(
            self.runtime_info,
            expected_model_fingerprint=expected_model_fingerprint,
            require_verified_model=require_verified_model,
        )
        self.runtime_id = self.runtime_info["runtime_id"]
        self.action_schema = self.runtime_info["action_schema"]
        self.model_identity = self.runtime_info["model"]

    def get_runtime_info(self) -> dict:
        return self._client.call(
            "vla.get_runtime_info", timeout_s=self._TIMEOUT_S["default"]
        )

    @staticmethod
    def _validate_runtime_info(
        info,
        *,
        expected_model_fingerprint=None,
        require_verified_model=False,
    ):
        if not isinstance(info, Mapping):
            raise TypeError("VLA runtime handshake must return a mapping")
        if info.get("protocol_version") != VLA_PROTOCOL_VERSION:
            raise RuntimeError(
                f"unsupported VLA protocol_version={info.get('protocol_version')!r}; "
                f"expected {VLA_PROTOCOL_VERSION}"
            )
        if not isinstance(info.get("runtime_id"), str) or not info["runtime_id"]:
            raise RuntimeError("VLA runtime handshake is missing runtime_id")
        if info.get("backend") != "rldx":
            raise RuntimeError(f"unsupported VLA backend {info.get('backend')!r}")
        if info.get("warmed_up") is not True:
            raise RuntimeError("VLA server has not completed model warmup")
        model = info.get("model")
        if not isinstance(model, Mapping) or not isinstance(
            model.get("fingerprint"), str
        ):
            raise RuntimeError("VLA runtime handshake is missing model identity")
        if require_verified_model and model.get("verified") is not True:
            raise RuntimeError("VLA runtime handshake has no verified model identity")
        if (
            expected_model_fingerprint is not None
            and model.get("fingerprint") != expected_model_fingerprint
        ):
            raise RuntimeError("VLA runtime model fingerprint mismatch")
        schema = info.get("action_schema")
        if not isinstance(schema, Mapping):
            raise RuntimeError("VLA runtime handshake is missing action_schema")
        if schema.get("batch_size") != 1 or schema.get("flat_dim") != sum(
            EXPECTED_ACTION_FIELDS.values()
        ):
            raise RuntimeError(f"incompatible VLA action schema: {schema!r}")
        fields = schema.get("fields")
        if not isinstance(fields, list):
            raise RuntimeError("VLA action schema fields must be a list")
        actual_fields = {
            field.get("name"): field.get("size")
            for field in fields
            if isinstance(field, Mapping)
        }
        if actual_fields != EXPECTED_ACTION_FIELDS:
            raise RuntimeError(
                f"incompatible VLA action fields: expected={EXPECTED_ACTION_FIELDS!r}, "
                f"actual={actual_fields!r}"
            )
        horizon = schema.get("max_horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise RuntimeError(f"invalid VLA max_horizon={horizon!r}")

    def get_modality_config(self) -> dict:
        config = self._client.call(
            "vla.get_modality_config", timeout_s=self._TIMEOUT_S["default"]
        )
        if not isinstance(config, Mapping):
            raise TypeError("VLA modality config must be a mapping")
        indices = np.asarray(config.get("video_delta_indices"))
        hist_maxlen = config.get("hist_maxlen")
        if (
            indices.ndim != 1
            or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(np.diff(indices) <= 0)
            or int(indices[-1]) != 0
        ):
            raise RuntimeError(f"invalid VLA video_delta_indices={indices!r}")
        expected_hist = int(indices.max() - indices.min()) + 2
        if (
            isinstance(hist_maxlen, bool)
            or not isinstance(hist_maxlen, int)
            or hist_maxlen != expected_hist
        ):
            raise RuntimeError(
                f"invalid VLA hist_maxlen={hist_maxlen!r}; expected {expected_hist}"
            )
        return dict(config)

    def predict(self, obs_dict: dict, options: dict) -> dict:
        """Run inference; returns raw actions dict.

        Actions are numpy arrays.
        """
        actions = self._client.call(
            "vla.predict",
            args=(obs_dict, options),
            timeout_s=self._TIMEOUT_S["predict"],
        )
        if not isinstance(actions, Mapping):
            raise TypeError("VLA predict response must be an action mapping")
        normalized = dict(actions)
        horizon = None
        max_horizon = int(self.action_schema["max_horizon"])
        for name, size in EXPECTED_ACTION_FIELDS.items():
            if name not in actions:
                raise RuntimeError(f"VLA predict response is missing {name!r}")
            array = np.asarray(actions[name])
            if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != size:
                raise ValueError(
                    f"{name} must have shape (1, horizon, {size}), got {array.shape}"
                )
            if not 0 < array.shape[1] <= max_horizon:
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
                raise ValueError("VLA action fields have inconsistent horizons")
            normalized[name] = array
        return normalized

    def reset_session(self, session_id: str) -> dict:
        if not isinstance(session_id, str) or not session_id:
            raise TypeError("session_id must be a non-empty string")
        result = self._client.call(
            "vla.reset_session",
            args=(session_id,),
            timeout_s=self._TIMEOUT_S["default"],
        )
        if (
            not isinstance(result, Mapping)
            or result.get("ok") is not True
            or result.get("runtime_id") != self.runtime_id
            or result.get("session_id") != session_id
        ):
            raise RuntimeError(f"invalid VLA reset_session response: {result!r}")
        return dict(result)
