# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""RoboTwin qpos observation and controller adapter for compatible WAM workers."""

from functools import cached_property
from typing import Any

import numpy as np

from robots.robotwin.control import (
    ROBOTWIN_CONTROLLERS,
    ROBOTWIN_QPOS,
)
from robots.robotwin.observation import physical_observation
from rpent.robots.components.wam_client_base import BaseWAMClient
from rpent.robots.components.wam_control_spec import WAMControlSpec
from rpent.robots.components.wam_rpc_protocol import WAMPrediction, WAMRequest
from rpent.utils.rpc import RpcClient

ACTION_SPACE = ROBOTWIN_QPOS["action_space"]
ACTION_SCHEMA = ROBOTWIN_QPOS["action_schema"]


def robotwin_request(
    observation: dict[str, Any], controller: WAMControlSpec = ROBOTWIN_QPOS
) -> WAMRequest:
    """Attach the selected execution contract to native RoboTwin observations."""
    return {
        **physical_observation(observation),
        "embodiment": controller["embodiment"],
        "action_space": controller["action_space"],
    }


class RoboTwinWAMClient:
    """Return negotiated qpos14/EEF16 chunks for RoboTwin's native executor."""

    @cached_property
    def controller(self) -> WAMControlSpec:
        caps = self.wam.get_capabilities()
        try:
            controller = ROBOTWIN_CONTROLLERS[caps["control"]["action_space"]]
        except KeyError:
            raise ValueError(
                f"Unsupported RoboTwin action space: {caps['control']['action_space']}"
            ) from None
        self.wam.validate_contract(controller)
        return controller

    @property
    def action_type(self) -> str:
        return self.controller["action_type"]

    def __init__(
        self, client: RpcClient, *, expected_backend: str | None = None
    ) -> None:
        self.wam = BaseWAMClient(client, expected_backend=expected_backend)

    def validate_robotwin(self) -> None:
        """Reject incompatible model controllers before executing actions."""
        self.controller

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        """Return executable actions in the selected platform control format."""
        return self.predict_result(observation)["actions"]

    def predict_result(self, observation: dict[str, Any]) -> WAMPrediction:
        """Return a validated action chunk and its model metadata."""
        return self.wam.predict(robotwin_request(observation, self.controller))

    def reset(self) -> None:
        """Clear model history before starting a new episode."""
        self.wam.reset()

    def close(self) -> None:
        """Release the RPC client and its session resources."""
        self.wam.close()
