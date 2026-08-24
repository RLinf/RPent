"""RoboCasa VLA client — thin RPC layer over the VLA server."""
from __future__ import annotations

from rpent.tools.vla_client_base import BaseVLAClient


class RoboCasaVLAClient(BaseVLAClient):
    """RoboCasa VLA client. ``predict`` inherited from :class:`BaseVLAClient`.

    The session id is the client's private one (injected by the RPC
    facade); the caller does not pass it. ``predict`` does not set
    ``options["session_ids"]`` — the server injects it.
    """

    def get_video_delta_indices(self):
        return self._client.call(
            "vla.get_video_delta_indices",
            timeout_s=self._TIMEOUT_S["default"],
        )

    def reset_session(self) -> dict:
        """Reset RLDX internal state (memory/RTC) for this client's session.

        The session stays live for subsequent calls.
        """
        return self._client.call(
            "vla.reset_session",
            timeout_s=self._TIMEOUT_S["default"],
        )
