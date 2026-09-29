# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Wire contracts shared by world-action-model clients and bridges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


class ActionModelProtocolError(ValueError):
    """Raised when an action-model request or response is malformed."""


class ActionModelCompatibilityError(ActionModelProtocolError):
    """Raised when a backend cannot control the requested embodiment."""


@dataclass(frozen=True)
class ActionModelCapabilities:
    """Capabilities advertised by one action-model bridge."""

    backend: str
    checkpoint: str
    supported_embodiments: tuple[str, ...]
    action_dim: int
    camera_roles: tuple[str, ...]
    action_schema: tuple[str, ...] = ()
    proprio_schema: tuple[str, ...] = ()
    returns_future_observation: bool = False
    returns_value: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: Any) -> "ActionModelCapabilities":
        """Validate and decode a capabilities RPC response."""
        if not isinstance(payload, Mapping):
            raise ActionModelProtocolError("capabilities must be a mapping")
        backend = _required_text(payload, "backend")
        checkpoint = _required_text(payload, "checkpoint")
        embodiments = _string_tuple(
            payload.get("supported_embodiments"), "supported_embodiments"
        )
        camera_roles = _string_tuple(payload.get("camera_roles"), "camera_roles")
        try:
            action_dim = int(payload["action_dim"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionModelProtocolError(
                "capabilities.action_dim must be an integer"
            ) from exc
        if action_dim <= 0:
            raise ActionModelProtocolError("capabilities.action_dim must be positive")
        action_schema = _optional_string_tuple(
            payload.get("action_schema"), "action_schema"
        )
        if action_schema and len(action_schema) != action_dim:
            raise ActionModelProtocolError(
                "capabilities.action_schema length must equal action_dim"
            )
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ActionModelProtocolError("capabilities.metadata must be a mapping")
        return cls(
            backend=backend,
            checkpoint=checkpoint,
            supported_embodiments=embodiments,
            action_dim=action_dim,
            camera_roles=camera_roles,
            action_schema=action_schema,
            proprio_schema=_optional_string_tuple(
                payload.get("proprio_schema"), "proprio_schema"
            ),
            returns_future_observation=bool(
                payload.get("returns_future_observation", False)
            ),
            returns_value=bool(payload.get("returns_value", False)),
            metadata=dict(metadata),
        )

    def to_wire(self) -> dict[str, Any]:
        """Encode capabilities using transport-supported values."""
        return {
            "backend": self.backend,
            "checkpoint": self.checkpoint,
            "supported_embodiments": list(self.supported_embodiments),
            "action_dim": self.action_dim,
            "camera_roles": list(self.camera_roles),
            "action_schema": list(self.action_schema),
            "proprio_schema": list(self.proprio_schema),
            "returns_future_observation": self.returns_future_observation,
            "returns_value": self.returns_value,
            "metadata": dict(self.metadata),
        }

    def require_embodiment(
        self,
        embodiment: str,
        *,
        action_dim: int | None = None,
        action_schema: tuple[str, ...] | None = None,
    ) -> None:
        """Reject an execution contract not explicitly advertised by the bridge."""
        if embodiment not in self.supported_embodiments:
            raise ActionModelCompatibilityError(
                f"backend {self.backend!r} checkpoint {self.checkpoint!r} does not "
                f"support embodiment {embodiment!r}; supported="
                f"{list(self.supported_embodiments)!r}"
            )
        if action_dim is not None and self.action_dim != action_dim:
            raise ActionModelCompatibilityError(
                f"backend action_dim={self.action_dim} does not match required {action_dim}"
            )
        if action_schema is not None and self.action_schema != action_schema:
            raise ActionModelCompatibilityError(
                f"backend action_schema={list(self.action_schema)!r} does not match "
                f"required {list(action_schema)!r}"
            )


@dataclass
class ActionModelPrediction:
    """Normalized output of one action-model inference call."""

    actions: np.ndarray
    future_observation: Any | None = None
    value: float | np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: Any) -> "ActionModelPrediction":
        """Validate and decode a prediction RPC response."""
        if not isinstance(payload, Mapping):
            raise ActionModelProtocolError("prediction must be a mapping")
        if "actions" not in payload:
            raise ActionModelProtocolError("prediction omitted actions")
        try:
            actions = np.asarray(payload["actions"], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ActionModelProtocolError(
                "prediction.actions must be numeric"
            ) from exc
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
            raise ActionModelProtocolError(
                f"prediction.actions must have non-empty [T, A] shape, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ActionModelProtocolError("prediction.actions contains NaN or Inf")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ActionModelProtocolError("prediction.metadata must be a mapping")
        value = payload.get("value")
        if value is not None:
            value_array = np.asarray(value)
            if (
                not np.issubdtype(value_array.dtype, np.number)
                or not np.isfinite(value_array).all()
            ):
                raise ActionModelProtocolError(
                    "prediction.value must be finite numeric data"
                )
            value = float(value_array) if value_array.ndim == 0 else value_array
        return cls(
            actions=actions,
            future_observation=payload.get("future_observation"),
            value=value,
            metadata=dict(metadata),
        )

    def to_wire(self) -> dict[str, Any]:
        """Encode the prediction using transport-supported values."""
        return {
            "actions": self.actions,
            "future_observation": self.future_observation,
            "value": self.value,
            "metadata": dict(self.metadata),
        }


def normalize_action_model_request(payload: Any) -> dict[str, Any]:
    """Validate a unified action-model request without backend preprocessing."""
    if not isinstance(payload, Mapping):
        raise ActionModelProtocolError("action-model request must be a mapping")
    instruction = _required_text(payload, "instruction")
    embodiment = _required_text(payload, "embodiment")
    images = payload.get("images")
    if not isinstance(images, Mapping):
        raise ActionModelProtocolError("request.images must be a mapping")
    primary = _normalize_rgb(images.get("primary"), "images.primary")
    wrist = images.get("wrist")
    if wrist is not None:
        wrist = _normalize_rgb(wrist, "images.wrist")
    extra = images.get("extra", [])
    if not isinstance(extra, (list, tuple)):
        raise ActionModelProtocolError("request.images.extra must be a sequence")
    normalized_extra = [
        _normalize_rgb(image, f"images.extra[{index}]")
        for index, image in enumerate(extra)
    ]
    try:
        proprio = np.asarray(payload.get("proprio"), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ActionModelProtocolError("request.proprio must be numeric") from exc
    if proprio.ndim != 1 or proprio.size == 0:
        raise ActionModelProtocolError(
            f"request.proprio must be non-empty and one-dimensional, got {proprio.shape}"
        )
    if not np.isfinite(proprio).all():
        raise ActionModelProtocolError("request.proprio contains NaN or Inf")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ActionModelProtocolError("request.metadata must be a mapping")
    return {
        "images": {"primary": primary, "wrist": wrist, "extra": normalized_extra},
        "proprio": proprio,
        "instruction": instruction,
        "embodiment": embodiment,
        "metadata": dict(metadata),
    }


def _required_text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ActionModelProtocolError(f"{name} must be a non-empty string")
    return value.strip()


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    result = _optional_string_tuple(value, name)
    if not result:
        raise ActionModelProtocolError(f"capabilities.{name} must not be empty")
    return result


def _optional_string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ActionModelProtocolError(f"capabilities.{name} must be a string sequence")
    return tuple(item.strip() for item in value)


def _normalize_rgb(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ActionModelProtocolError(
            f"{name} must have [H, W, 3] shape, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ActionModelProtocolError(f"{name} must be numeric")
    if not np.isfinite(array).all():
        raise ActionModelProtocolError(f"{name} contains NaN or Inf")
    return array.astype(np.uint8, copy=False)
