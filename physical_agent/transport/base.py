"""Small transport interface for the interactive driver boundary."""
from __future__ import annotations

from typing import Protocol


class TransportClient(Protocol):
    """Client-side interface used by tool frontends."""

    def request(
        self,
        method: str,
        params: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict:
        """Send one transport request and return a JSON-serializable result."""

    def close(self) -> None:
        """Release any client-side transport resources."""
