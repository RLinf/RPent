"""Offline tests for the Franka Pi0.5 VLA client wire format."""

from __future__ import annotations

import numpy as np

from robots.franka.vla_client import VLAClient


class FakeRpcClient:
    """Record ``predict`` payloads and return a deterministic action chunk."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        del args, timeout_s
        if method == "predict":
            self.payloads.append(dict(kwargs or {}))
            return {"actions": np.zeros((1, 4, 7), dtype=np.float32)}
        return {}


def _env_obs(**overrides):
    obs = {
        "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
        "states": np.zeros(7, dtype=np.float32),
        "task_descriptions": "grasp the cube",
    }
    obs.update(overrides)
    return obs


def test_single_extra_view_is_serialized_as_dict():
    client = VLAClient(FakeRpcClient())
    client.predict_action_batch(
        _env_obs(extra_view_images=np.zeros((1, 8, 8, 3), dtype=np.uint8)),
        mode="eval",
    )
    extra = client._client.payloads[0]["images"]["extra"]
    assert isinstance(extra, dict)
    assert extra["format"] == "png"


def test_multiple_extra_views_are_serialized_as_list():
    client = VLAClient(FakeRpcClient())
    client.predict_action_batch(
        _env_obs(extra_view_images=np.zeros((2, 8, 8, 3), dtype=np.uint8)),
        mode="eval",
    )
    extra = client._client.payloads[0]["images"]["extra"]
    assert isinstance(extra, list)
    assert len(extra) == 2


def test_no_extra_view_omits_extra_key():
    client = VLAClient(FakeRpcClient())
    client.predict_action_batch(_env_obs(), mode="eval")
    assert "extra" not in client._client.payloads[0]["images"]
