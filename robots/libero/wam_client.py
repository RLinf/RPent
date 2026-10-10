# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""LIBERO observation and controller adapter shared by compatible WAM workers."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots.libero.control import LIBERO_OSC
from robots.libero.observation import physical_observation
from rpent.robots.components.wam_client_base import BaseWAMClient
from rpent.robots.components.wam_rpc_protocol import WAM_PROTOCOL_VERSION, WAMPrediction
from rpent.utils.rpc import RpcClient

ACTION_SPACE = LIBERO_OSC.action_space
ACTION_SCHEMA = LIBERO_OSC.action_schema


def libero_request(raw_obs: dict[str, Any]) -> dict[str, Any]:
    """Attach the WAM execution contract to model-independent observations."""
    return {
        **physical_observation(raw_obs),
        "protocol_version": WAM_PROTOCOL_VERSION,
        "embodiment": LIBERO_OSC.embodiment,
        "action_space": ACTION_SPACE,
    }


class LiberoWAMClient:
    """Adapt raw LIBERO observations to the model-independent WAM client."""

    def __init__(
        self, client: RpcClient, *, expected_backend: str | None = None
    ) -> None:
        self.wam = BaseWAMClient(client, expected_backend=expected_backend)

    def validate_libero(self) -> None:
        """Reject incompatible controllers before starting a control loop."""
        LIBERO_OSC.require(self.wam.get_capabilities())

    def predict(self, env_obs: dict, options: dict | None = None) -> np.ndarray:
        """Return executable actions for the existing LIBERO chunk executor."""
        return self.predict_result(env_obs, options).actions

    def predict_result(
        self, env_obs: dict, options: dict | None = None
    ) -> WAMPrediction:
        """Return actions and optional predictions separately from simulator state."""
        if options and options != {"mode": "eval"}:
            raise ValueError("LIBERO WAM supports only evaluation options")
        self.validate_libero()
        return self.wam.predict(libero_request(env_obs))

    def reset(self) -> None:
        """Clear model history when the caller begins a new episode."""
        self.wam.reset()

    def close(self) -> None:
        """Release model session resources."""
        self.wam.close()
