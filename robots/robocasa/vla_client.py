"""RoboCasa VLA client — thin RPC layer over the VLA server."""
from __future__ import annotations

from rpent.tools.vla_client_base import BaseVLAClient


class RoboCasaVLAClient(BaseVLAClient):
    _TIMEOUT_S = {
        **BaseVLAClient._TIMEOUT_S,
    }

    def __init__(self, client):
        super().__init__(client)

    def get_modality_config(self) -> dict:
        return self._client.call("vla.get_modality_config", timeout_s=self._TIMEOUT_S["default"])

    def predict(self, obs_dict: dict, options: dict) -> dict:
        """Run inference; returns raw actions dict.

        Actions are numpy arrays. The caller must NOT set
        ``options["session_ids"]`` — the server injects the caller's private
        session id so RLDX memory/RTC state is isolated per client.
        """
        return self._client.call("vla.predict", args=(obs_dict, options),
                                 timeout_s=self._TIMEOUT_S["predict"])

    def reset_session(self) -> dict:
        """Reset RLDX internal state (memory/RTC) for this client's session.

        The session id is the client's private one (injected by the RPC
        facade); the caller does not pass it. The session stays live for
        subsequent calls.
        """
        return self._client.call("env.reset_session",
                                 timeout_s=self._TIMEOUT_S["default"])
