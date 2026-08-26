"""Unified VLA client base class.
Design reference: ``docs/source-zh/rst_source/development/add_vla.rst``.
"""
from __future__ import annotations


class BaseVLAClient:
    """Unified VLA client base class.
    """

    _TIMEOUT_S: dict[str, float] = {"default": 30.0, "predict": 120.0}

    def __init__(self, client):
        self._client = client

    def predict(self, obs, options=None):
        """Request a single VLA action chunk.

        Args:
            obs: observation data.
            options: optional dict.

        Returns:
            actions.
        """
        return self._client.call("vla.predict", args=(obs, options),
                                timeout_s=self._TIMEOUT_S["predict"])
