"""RoboCasa VLA client — thin RPC layer over the VLA server."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rpent.utils.rpc import RpcClient

_TIMEOUT_S = {
    "default": 30.0,
    "predict": 120.0,
}


class RoboCasaVLAClient:
    def __init__(self, client: RpcClient):
        self._client = client

    def get_modality_config(self) -> dict:
        return self._client.call("env.get_modality_config", timeout_s=_TIMEOUT_S["default"])

    def predict(self, obs_dict: dict, options: dict) -> dict:
        """Run inference; returns raw actions dict.

        Actions are numpy arrays.
        """
        return self._client.call("env.predict", args=(obs_dict, options),
                                 timeout_s=_TIMEOUT_S["predict"])

    def reset_session(self, session_id: str) -> dict:
        return self._client.call("env.reset_session", args=(session_id,),
                                 timeout_s=_TIMEOUT_S["default"])
