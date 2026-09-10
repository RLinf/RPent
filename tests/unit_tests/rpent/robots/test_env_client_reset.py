# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contracts for the shared client-side reset commit."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rpent.robots.components.env_client_base import (
    BaseEnvClient,
    ExplorationResetOutcome,
)


def test_reset_outcome_is_typed_and_frozen() -> None:
    observation = {"frame": 1}
    details = {"seed": 7}

    outcome = ExplorationResetOutcome(observation=observation, details=details)

    assert outcome.observation is observation
    assert outcome.details is details
    with pytest.raises(FrozenInstanceError):
        outcome.observation = {"frame": 2}


def test_common_reset_commit_updates_cache_and_returns_independent_details() -> None:
    client = BaseEnvClient.__new__(BaseEnvClient)
    client.last_obs = {"old": True}
    client.last_info = {"nested": {"value": "old"}}
    details = {"nested": {"value": "new"}}
    outcome = ExplorationResetOutcome(observation={"new": True}, details=details)

    returned = client._commit_reset_outcome(
        outcome,
        cache_updates={"last_info": details, "terminated": False},
    )

    assert client.last_obs is outcome.observation
    assert client.last_info == details
    assert client.last_info is not details
    assert client.terminated is False
    returned["nested"]["value"] = "caller mutation"
    assert client.last_info["nested"]["value"] == "new"
    assert details["nested"]["value"] == "new"


def test_common_reset_commit_copies_before_changing_cache() -> None:
    class CopyFailure:
        def __deepcopy__(self, memo):
            del memo
            raise RuntimeError("decoder copy failed")

    client = BaseEnvClient.__new__(BaseEnvClient)
    previous_observation = {"old": True}
    previous_info = {"old": True}
    client.last_obs = previous_observation
    client.last_info = previous_info
    outcome = ExplorationResetOutcome(
        observation={"new": True}, details={"bad": CopyFailure()}
    )

    with pytest.raises(RuntimeError, match="decoder copy failed"):
        client._commit_reset_outcome(
            outcome,
            cache_updates={"last_info": {"new": True}},
        )

    assert client.last_obs is previous_observation
    assert client.last_info is previous_info
