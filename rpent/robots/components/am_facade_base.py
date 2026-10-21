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

"""Shared action-model RPC registration and VLA/WAM service contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from rpent.robots.components.am_contracts import (
    WAMModelMeta,
    WAMPrediction,
    WAMRequest,
)
from rpent.utils.rpc import RpcFacade
from rpent.utils.rpc.rpc_facade import DEFAULT_SESSION_TIMEOUT_S

# --- Shared RPC registration ---


class BaseAMFacade(RpcFacade):
    """Register prediction while reusing RPC locking and session ownership."""

    PREDICT_METHOD: str

    def __init__(
        self,
        *,
        enable_sessions: bool = False,
        session_timeout_s: float = DEFAULT_SESSION_TIMEOUT_S,
    ) -> None:
        super().__init__(
            enable_sessions=enable_sessions, session_timeout_s=session_timeout_s
        )
        self._register_rpc()

    def _register_rpc(self) -> None:
        self._rpc[self.PREDICT_METHOD] = self.predict

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Run model inference using the subclass's request and result format."""
        raise NotImplementedError


# --- VLA: model-owned observation and action formats ---


class BaseVLAFacade(BaseAMFacade):
    """Preserve VLA prediction and model-specific RPC/session extensions.

    Subclasses implement ``predict`` and may extend ``_register_rpc``.
    Session-aware services supply their own reset and session-drop hooks.
    """

    PREDICT_METHOD = "vla.predict"


# --- WAM: paired adapters, physical observations and native actions ---


WAMAdapter = tuple[
    Callable[[dict[str, Any]], dict[str, Any]],
    Callable[[Any], WAMPrediction],
]


@dataclass(frozen=True)
class WAMAdapterSpec:
    """Register paired transforms and a checkpoint metadata factory."""

    encode: Callable[..., dict[str, Any]]
    decode: Callable[..., WAMPrediction]
    model_meta: Callable[..., WAMModelMeta]


class BaseWAMFacade(BaseAMFacade):
    """Own the common ``wam.*`` routes, independent of model and simulator."""

    PREDICT_METHOD = "wam.predict"

    def __init__(
        self,
        model_meta: WAMModelMeta,
        *,
        adapter: WAMAdapter,
        session_timeout_s: float = DEFAULT_SESSION_TIMEOUT_S,
    ) -> None:
        self._model_meta = model_meta
        self._adapter = adapter
        super().__init__(
            enable_sessions=model_meta["uses_sessions"],
            session_timeout_s=session_timeout_s,
        )

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc.update(
            {
                "wam.get_model_meta": self.get_model_meta,
                "wam.reset": self.reset,
            }
        )
        self._readonly_methods.add("wam.get_model_meta")

    def get_model_meta(self, *, session_id: str | None = None) -> WAMModelMeta:
        """Return this checkpoint's input and execution contract."""
        return self._model_meta

    def predict(
        self, request: WAMRequest, *, session_id: str | None = None
    ) -> WAMPrediction:
        """Check incoming observations, then encode, infer and decode directly."""
        control = self._model_meta["control"]
        if any(request[key] != control[key] for key in ("embodiment", "action_space")):
            raise ValueError("WAM request does not match the worker's control contract")
        if (
            not isinstance(request["instruction"], str)
            or not request["instruction"].strip()
        ):
            raise ValueError("instruction must be a non-empty string")
        for role in self._model_meta["camera_roles"]:
            image = np.asarray(request["images"].get(role))
            if (
                image.ndim != 3
                or image.shape[-1] != 3
                or 0 in image.shape
                or image.dtype != np.uint8
            ):
                raise ValueError(
                    f"images.{role} must have non-empty [H, W, 3] uint8 RGB shape"
                )
        for name, size in self._model_meta["state_schema"].items():
            state = np.asarray(request["state"].get(name))
            if (
                state.shape != (size,)
                or state.dtype.kind not in "iuf"
                or not np.isfinite(state).all()
            ):
                raise ValueError(
                    f"state.{name} must have shape ({size},) without NaN or Inf"
                )
        model_input = self._adapter[0](request)
        result = self.predict_native(model_input, session_id=session_id)
        return self._adapter[1](result)

    def predict_native(
        self, request: dict[str, Any], *, session_id: str | None = None
    ) -> Any:
        """Infer from encoded input; a configured adapter decodes the native result."""
        raise NotImplementedError

    def reset(self, *, session_id: str | None = None) -> dict[str, bool]:
        """Release episode history; stateless workers need no reset hook."""
        self.reset_native(session_id=session_id)
        return {"ok": True}

    def reset_native(self, *, session_id: str | None = None) -> None:
        """Stateful workers must discard this session's cached model state."""
        if self._model_meta["uses_sessions"]:
            raise NotImplementedError(
                "session-based WAM workers must implement reset_native"
            )

    def _on_session_drop(self, session_id: str) -> None:
        self.reset_native(session_id=session_id)

    def close(self) -> None:
        """Release remaining model histories during server shutdown."""
        for session_id in list(self._sessions):
            self.drop_session(session_id)
        super().close()
