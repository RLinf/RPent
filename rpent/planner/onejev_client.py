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
"""Lightweight HTTP client for an external OneJev System One service."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from rpent.planner.onejev_types import DecisionContext

DEFAULT_ONEJEV_MODEL = "OneJev-9B"


@dataclass(frozen=True)
class ChoiceScores:
    """Validated scores and diagnostic metadata for a candidate choice."""

    probabilities: dict[str, float]
    confidence: float
    usage: dict[str, int]
    request_id: str | None
    metadata: dict[str, Any]


class OneJevClient:
    """Call OneJev without installing its model package in the RPent process."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str = DEFAULT_ONEJEV_MODEL,
        timeout_s: float = 120.0,
    ) -> None:
        """Validate the endpoint and own a synchronous HTTP connection pool."""
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("OneJev requires an http(s) service URL via --base-url")
        if url.query or url.fragment or url.username or url.password:
            raise ValueError(
                "OneJev URL must not contain credentials, query or fragment"
            )
        if url.path not in {"", "/"}:
            raise ValueError("OneJev --base-url must be the server root, without /v1")
        if not model:
            raise ValueError("OneJev requires a served model name")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("OneJev timeout must be finite and positive")
        self.model = model
        self.base_url = str(url).rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_s,
            trust_env=False,
        )

    def close(self) -> None:
        """Close client connections; the external model server stays running."""
        self._http.close()

    def check_service(self, *, timeout_s: float = 30.0) -> None:
        """Check readiness and verify the exact requested model is served."""
        deadline = time.monotonic() + timeout_s
        health = self._http.get("/health", timeout=timeout_s)
        health.raise_for_status()
        health_body = health.json()
        if not isinstance(health_body, dict) or health_body.get("status") != "ok":
            raise ValueError("OneJev service is not ready")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise httpx.TimeoutException("OneJev service check timed out")
        models = self._http.get("/v1/models", timeout=remaining)
        models.raise_for_status()
        model_body = models.json()
        items = model_body.get("models") if isinstance(model_body, dict) else None
        if not isinstance(items, list) or not all(
            isinstance(item, dict) and isinstance(item.get("name"), str)
            for item in items
        ):
            raise ValueError("OneJev model list must contain named model objects")
        names = {item["name"] for item in items}
        if self.model not in names:
            raise ValueError(
                f"OneJev model {self.model!r} is not served; available: {sorted(names)}"
            )

    def score(
        self,
        context: DecisionContext,
        *,
        timeout_s: float,
        on_request: Callable[[dict[str, Any]], None] | None = None,
        on_response: Callable[[dict[str, Any]], None] | None = None,
    ) -> ChoiceScores:
        """Score a fixed candidate set and reject malformed or foreign scores.

        Trace callbacks receive the wire request before HTTP and the raw
        response before validation. A failed callback stops the operation.
        """
        candidates = context.candidates
        ids = [action.id for action in candidates]
        if not 1 <= len(ids) <= 255 or len(set(ids)) != len(ids):
            raise ValueError("OneJev needs 1–255 candidates with unique IDs")
        request = {
            "model": self.model,
            "state": context.state,
            "media": context.media,
            "questions": {
                "next_action": {
                    "type": "choice",
                    "instructions": context.instructions,
                    "criteria": {
                        action.id: {
                            "description": action.description,
                            "tool": action.tool_name,
                            "arguments": action.arguments,
                        }
                        for action in candidates
                    },
                }
            },
        }
        if on_request is not None:
            on_request(request)
        response = self._http.post("/v1/systemone", timeout=timeout_s, json=request)
        try:
            body = response.json()
        except ValueError:
            body = response.text
        if on_response is not None:
            on_response(
                {
                    "http_status": response.status_code,
                    "request_id": response.headers.get("x-typesafe-request-id"),
                    "output": body,
                }
            )
        response.raise_for_status()
        if not isinstance(body, dict):
            raise ValueError("OneJev response must be a JSON object")
        if body.get("model") != self.model:
            raise ValueError("OneJev returned a different model")
        answers = body.get("answers")
        if not isinstance(answers, dict) or set(answers) != {"next_action"}:
            raise ValueError("OneJev returned an unexpected question set")
        answer = answers["next_action"]
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError("OneJev next_action must be a choice answer")
        raw = answer.get("probabilities")
        if not isinstance(raw, dict) or set(raw) != set(ids):
            raise ValueError("OneJev probabilities must match every candidate ID")
        probabilities = {}
        for candidate_id in ids:
            value = raw[candidate_id]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("OneJev probabilities must be numbers")
            probability = float(value)
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("OneJev probabilities must be finite and in [0, 1]")
            probabilities[candidate_id] = probability
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.001):
            raise ValueError("OneJev probabilities do not sum to one")
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in probabilities:
            raise ValueError("OneJev returned an unknown choice")
        if probabilities[choice] + 0.000001 < max(probabilities.values()):
            raise ValueError("OneJev choice disagrees with its probabilities")
        confidence = answer.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("OneJev confidence must be a number")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("OneJev confidence must be finite and in [0, 1]")
        usage = body.get("usage", {})
        if not isinstance(usage, dict):
            raise ValueError("OneJev usage must be an object")
        counts = {}
        for key in ("input_tokens", "output_tokens"):
            value = usage.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("OneJev token usage must be nonnegative integers")
            counts[key] = value
        metadata = body.get("qev", {})
        if not isinstance(metadata, dict):
            raise ValueError("OneJev metadata must be an object")
        return ChoiceScores(
            probabilities=probabilities,
            confidence=float(confidence),
            usage=counts,
            request_id=response.headers.get("x-typesafe-request-id"),
            metadata=metadata,
        )
