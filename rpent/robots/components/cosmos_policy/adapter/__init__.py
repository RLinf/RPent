# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Matched Cosmos Policy platform encoders and decoders."""

from functools import partial

from robots.libero.control import LIBERO_OSC
from rpent.robots.components.cosmos_policy import USES_SESSIONS
from rpent.robots.components.cosmos_policy.adapter.decode import decode_libero
from rpent.robots.components.cosmos_policy.adapter.encode import (
    STATE_SCHEMA,
    encode_libero,
)
from rpent.robots.components.wam_facade_base import (
    WAMAdapter,
    WAMAdapterSpec,
)
from rpent.robots.components.wam_rpc_protocol import WAMCapabilities

ACTION_SCHEMA = LIBERO_OSC["action_schema"]
ADAPTERS = {
    "libero": WAMAdapterSpec(
        encode_libero,
        decode_libero,
        partial(
            WAMCapabilities,
            control=LIBERO_OSC,
            camera_roles=("primary", "wrist"),
            state_schema=STATE_SCHEMA,
        ),
    ),
}


def make_adapter(
    platform: str, checkpoint: str, *, predict_future: bool = False
) -> tuple[WAMCapabilities, WAMAdapter]:
    """Select a supported checkpoint contract and its paired transforms."""
    if platform not in ADAPTERS:
        raise ValueError(f"unsupported Cosmos Policy platform: {platform!r}")
    spec = ADAPTERS[platform]
    capabilities = spec.capabilities(
        backend="cosmos_policy",
        checkpoint=checkpoint,
        chunk_size=16,
        uses_sessions=USES_SESSIONS,
        returns_future_observation=predict_future,
        returns_value=predict_future,
        metadata={},
    )
    return capabilities, (
        spec.encode,
        partial(spec.decode, predict_future=predict_future),
    )
