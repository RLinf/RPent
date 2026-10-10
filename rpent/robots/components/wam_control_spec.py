# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Data type for platform-owned WAM action semantics, without simulator imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rpent.robots.components.wam_rpc_protocol import WAMCapabilities


@dataclass(frozen=True)
class WAMControlSpec:
    """Describe executable actions shared by platform clients and model adapters.

    This is a control specification, not a controller implementation.
    ``action_space`` identifies the platform's versioned action semantics;
    ``action_schema`` records field order. ``action_type`` selects a native
    environment execution mode when the platform exposes more than one.
    """

    embodiment: str
    action_space: str
    action_schema: tuple[str, ...]
    action_type: str | None = None

    def capabilities(self, **requirements: Any) -> WAMCapabilities:
        """Combine this execution contract with checkpoint input requirements."""
        from rpent.robots.components.wam_rpc_protocol import WAMCapabilities

        return WAMCapabilities(
            supported_embodiments=(self.embodiment,),
            action_space=self.action_space,
            action_schema=self.action_schema,
            **requirements,
        )

    def require(self, capabilities: WAMCapabilities) -> None:
        """Reject a worker whose output cannot be executed by this controller."""
        capabilities.require_execution(
            self.embodiment, self.action_space, action_schema=self.action_schema
        )
