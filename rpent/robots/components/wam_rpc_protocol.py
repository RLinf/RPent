# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""WAM RPC messages, capability declarations, and boundary validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

WAM_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class WAMCapabilities:
    """One checkpoint's input requirements and simulator-ready action contract.

    ``action_space`` identifies controller semantics, including field order,
    units, frame, scaling, gripper convention, and control rate. Embodiments
    sharing an endpoint must all implement this same execution contract.
    """

    backend: str
    checkpoint: str
    supported_embodiments: tuple[str, ...]
    action_space: str
    action_schema: tuple[str, ...]
    camera_roles: tuple[str, ...]
    state_schema: dict[str, int]
    chunk_size: int | None = None
    uses_sessions: bool = False
    returns_future_observation: bool = False
    returns_value: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def action_dim(self) -> int:
        """Number of ordered action fields accepted by the controller."""
        return len(self.action_schema)

    @classmethod
    def from_wire(cls, payload: Any) -> WAMCapabilities:
        """Validate and decode a capabilities RPC response."""
        _require_version(payload)
        state_schema = _named_mapping(payload.get("state_schema"), "state_schema")
        for name, size in state_schema.items():
            _positive_int(size, f"state_schema.{name}")
        chunk_size = payload.get("chunk_size")
        if chunk_size is not None:
            _positive_int(chunk_size, "chunk_size")
        flags = {}
        for name in ("uses_sessions", "returns_future_observation", "returns_value"):
            value = payload.get(name, False)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
            flags[name] = value
        return cls(
            backend=_required_text(payload, "backend"),
            checkpoint=_required_text(payload, "checkpoint"),
            supported_embodiments=_string_tuple(
                payload.get("supported_embodiments"), "supported_embodiments"
            ),
            action_space=_required_text(payload, "action_space"),
            action_schema=_string_tuple(payload.get("action_schema"), "action_schema"),
            camera_roles=_string_tuple(payload.get("camera_roles"), "camera_roles"),
            state_schema=dict(state_schema),
            chunk_size=chunk_size,
            metadata=dict(_named_mapping(payload.get("metadata", {}), "metadata")),
            **flags,
        )

    def to_wire(self) -> dict[str, Any]:
        """Encode capabilities using transport-supported values."""
        return {"protocol_version": WAM_PROTOCOL_VERSION, **asdict(self)}

    def require_execution(
        self,
        embodiment: str,
        action_space: str,
        *,
        action_schema: tuple[str, ...] | None = None,
        action_dim: int | None = None,
    ) -> None:
        """Reject a controller the checkpoint does not explicitly support."""
        if embodiment not in self.supported_embodiments:
            raise ValueError(
                f"backend {self.backend!r} checkpoint {self.checkpoint!r} does not "
                f"support embodiment {embodiment!r}; supported="
                f"{list(self.supported_embodiments)!r}"
            )
        if action_space != self.action_space:
            raise ValueError(
                f"backend action_space={self.action_space!r} does not match "
                f"required {action_space!r}"
            )
        if action_dim is not None and self.action_dim != action_dim:
            raise ValueError(
                f"backend action_dim={self.action_dim} does not match required {action_dim}"
            )
        if action_schema is not None and self.action_schema != action_schema:
            raise ValueError(
                "backend action_schema does not match the controller field order"
            )

    def validate_request(self, request: dict[str, Any]) -> None:
        """Check normalized observations against this checkpoint's requirements."""
        self.require_execution(request["embodiment"], request["action_space"])
        for role in self.camera_roles:
            if role not in request["images"]:
                raise ValueError(f"request.images is missing required camera {role!r}")
        for name, size in self.state_schema.items():
            value = request["state"].get(name)
            if value is None or value.shape != (size,):
                raise ValueError(
                    f"request.state.{name} must be a vector of length {size}"
                )

    def validate_prediction(self, prediction: WAMPrediction) -> None:
        """Check normalized output before returning or executing any actions."""
        if prediction.actions.shape[1] != self.action_dim:
            raise ValueError(
                f"prediction action dimension {prediction.actions.shape[1]} "
                f"does not match advertised action_dim {self.action_dim}"
            )
        if (
            self.chunk_size is not None
            and prediction.actions.shape[0] != self.chunk_size
        ):
            raise ValueError(f"prediction must contain {self.chunk_size} actions")
        if (
            prediction.future_observation is not None
            and not self.returns_future_observation
        ):
            raise ValueError("backend returned an unadvertised future_observation")
        if prediction.value is not None and not self.returns_value:
            raise ValueError("backend returned an unadvertised value")

    def prediction_metadata(self, embodiment: str) -> dict[str, Any]:
        """Identify the execution contract used for a prediction."""
        return {
            "backend": self.backend,
            "checkpoint": self.checkpoint,
            "embodiment": embodiment,
            "action_space": self.action_space,
            "action_schema": list(self.action_schema),
        }


@dataclass
class WAMPrediction:
    """Executable action chunk and optional model predictions, never env state."""

    actions: np.ndarray
    future_observation: Any | None = None
    value: float | np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: Any) -> WAMPrediction:
        """Validate and decode a prediction RPC response."""
        if not isinstance(payload, Mapping) or "actions" not in payload:
            raise ValueError("prediction must be a mapping containing actions")
        actions = _numeric_array(payload["actions"], "prediction.actions")
        if actions.ndim != 2 or 0 in actions.shape:
            raise ValueError(
                f"prediction.actions must have non-empty [T, A] shape, got {actions.shape}"
            )
        value = payload.get("value")
        if value is not None:
            value = _numeric_array(value, "prediction.value", dtype=None)
            if value.ndim == 0:
                value = float(value)
        return cls(
            actions=actions,
            future_observation=payload.get("future_observation"),
            value=value,
            metadata=dict(_named_mapping(payload.get("metadata", {}), "metadata")),
        )

    def to_wire(self) -> dict[str, Any]:
        """Encode the prediction without copying large observation arrays."""
        return {
            "actions": self.actions,
            "future_observation": self.future_observation,
            "value": self.value,
            "metadata": dict(self.metadata),
        }


def normalize_wam_request(payload: Any) -> dict[str, Any]:
    """Validate named sensor fields without model-specific preprocessing."""
    _require_version(payload)
    images = {}
    for name, value in _named_mapping(payload.get("images"), "images").items():
        image = np.asarray(value)
        if image.ndim != 3 or image.shape[2] != 3 or 0 in image.shape:
            raise ValueError(f"images.{name} must have non-empty [H, W, 3] shape")
        if image.dtype != np.uint8:
            raise ValueError(f"images.{name} must be uint8 RGB data")
        images[name] = image
    state = {}
    for name, value in _named_mapping(payload.get("state"), "state").items():
        vector = _numeric_array(value, f"state.{name}")
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(f"state.{name} must be a non-empty vector")
        state[name] = vector
    return {
        "protocol_version": WAM_PROTOCOL_VERSION,
        "images": images,
        "state": state,
        "instruction": _required_text(payload, "instruction"),
        "embodiment": _required_text(payload, "embodiment"),
        "action_space": _required_text(payload, "action_space"),
        "metadata": dict(_named_mapping(payload.get("metadata", {}), "metadata")),
    }


def _require_version(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("WAM payload must be a mapping")
    version = payload.get("protocol_version")
    if type(version) is not int or version != WAM_PROTOCOL_VERSION:
        raise ValueError(
            f"unsupported WAM protocol_version {version!r}; expected {WAM_PROTOCOL_VERSION}"
        )


def _required_text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _named_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not key.strip() for key in value
    ):
        raise ValueError(f"{name} must be a mapping with non-empty string keys")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty string sequence")
    result = tuple(item.strip() for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _positive_int(value: Any, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _numeric_array(
    value: Any, name: str, *, dtype: type | None = np.float32
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric data")
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array
