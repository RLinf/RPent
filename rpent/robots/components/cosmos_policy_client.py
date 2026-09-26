# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Client for Cosmos Policy action-model bridges."""

from __future__ import annotations

from rpent.robots.components.action_model_client_base import BaseActionModelClient


class CosmosPolicyClient(BaseActionModelClient):
    """Validate and call a Cosmos Policy bridge."""

    def __init__(self, client, *, timeout_s: float = 300.0) -> None:
        super().__init__(client, expected_backend="cosmos_policy", timeout_s=timeout_s)
