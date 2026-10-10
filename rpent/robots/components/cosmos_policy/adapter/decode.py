# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Cosmos Policy native predictions to platform execution actions."""

from typing import Any

import numpy as np

from rpent.robots.components.wam_rpc_protocol import WAMPrediction


def decode_libero(
    result: dict[str, Any], *, predict_future: bool = False
) -> WAMPrediction:
    """The official get_action already unnormalizes LIBERO actions."""
    return WAMPrediction(
        actions=np.asarray(result["actions"], dtype=np.float32),
        future_observation=result.get("future_image_predictions")
        if predict_future
        else None,
        value=result.get("value_prediction") if predict_future else None,
    )
