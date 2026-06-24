"""Driver client protocol for env/model RPC."""
from __future__ import annotations

from typing import Any, Protocol


class RpcClient(Protocol):
    """Transport from the agent process to the driver process."""

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """Invoke a remote method and return its result."""

    def close(self) -> None:
        """Release any client-side transport resources."""
