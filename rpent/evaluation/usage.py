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

"""Per-trial token accounting from model requests or authoritative SDK totals."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from json import dumps
from threading import Lock
from typing import Any, Iterator
from uuid import uuid4

from rpent.utils.rpc.rpc_client import unconfirmed_rpc_observer


class TokenLedger:
    """Collect model-native usage without converting between tokenizers or modalities."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str | None, str, str], dict[str, Any]] = {}
        self._lock = Lock()
        self._unconfirmed_model_requests: list[dict[str, str]] = []

    def record_usage(
        self,
        *,
        request_id: str,
        model: str,
        module: str,
        stage: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        coverage: str | None = None,
        granularity: str = "request",
        status: str = "completed",
        provider: str | None = None,
        modality: str | None = None,
        tokenizer: str | None = None,
        raw_usage: Mapping[str, Any] | None = None,
    ) -> None:
        """Record an attempt or replace its previous cumulative snapshot.

        Args:
            request_id: Stable identifier within the module and provider. Use a
                distinct ID for each retry or independently reported session.
            model: Actual model identifier when available, otherwise "unknown".
            module: Caller such as planner, perception or action_model.
            stage: Trial stage such as planning or recovery. Use online_unattributed
                when a session total spans stages.
            input_tokens: Inclusive normalized input. None denotes unavailable.
            output_tokens: Inclusive normalized output. None denotes unavailable.
            coverage: complete, partial, unknown or not_applicable. Defaults to
                coverage of the observed input and output fields.
            granularity: request for one model request or session for an SDK total.
            status: Attempt outcome, independent of whether usage is available.
            provider: Provider or SDK reporting the usage.
            modality: Model-native token modality when it can be distinguished.
            tokenizer: Model-native tokenizer identity/version when known.
            raw_usage: JSON-compatible source counters and provider-specific breakdowns.
        """
        if not request_id or not model or not module or not stage:
            raise ValueError(
                "token entries require request_id, model, module and stage"
            )
        if granularity not in {"request", "session"}:
            raise ValueError("token granularity must be request or session")
        counters = (input_tokens, output_tokens)
        if any(v is not None and (type(v) is not int or v < 0) for v in counters):
            raise ValueError("token counts must be nonnegative integers or None")
        if coverage is None:
            coverage = ("unknown", "partial", "complete")[
                sum(v is not None for v in counters)
            ]
        if coverage not in {"complete", "partial", "unknown", "not_applicable"}:
            raise ValueError(f"invalid token coverage {coverage!r}")
        if coverage == "complete" and None in counters:
            raise ValueError("complete token coverage requires both counters")
        if coverage in {"not_applicable", "unknown"} and any(
            v is not None for v in counters
        ):
            raise ValueError(f"{coverage} token coverage cannot have counters")
        # Reject invalid external payloads at ingestion instead of losing the
        # whole episode when the recorder publishes its JSON artifact.
        raw = deepcopy(dict(raw_usage or {}))
        if raw:
            dumps(raw, allow_nan=False)
        entry = {
            "request_id": request_id,
            "model": model,
            "module": module,
            "stage": stage,
            "granularity": granularity,
            "status": status,
            "provider": provider,
            "modality": modality,
            "tokenizer": tokenizer,
            "coverage": coverage,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "raw_usage": raw,
        }
        with self._lock:
            self._entries[provider, module, request_id] = entry

    @contextmanager
    def unmetered_call(self, *, model: str, module: str, stage: str) -> Iterator[None]:
        """Record one opaque model invocation, including failed and interrupted attempts."""
        request_id = uuid4().hex
        status = "failed"

        def unconfirmed(method: str, cause: str) -> None:
            with self._lock:
                self._unconfirmed_model_requests.append(
                    {
                        "request_id": request_id,
                        "module": module,
                        "model": model,
                        "rpc_method": method,
                        "cause": cause,
                    }
                )

        token = unconfirmed_rpc_observer.set(unconfirmed)
        try:
            yield
            status = "completed"
        finally:
            unconfirmed_rpc_observer.reset(token)
            self.record_usage(
                request_id=request_id,
                model=model,
                module=module,
                stage=stage,
                status=status,
            )

    @property
    def unconfirmed_model_requests(self) -> list[dict[str, str]]:
        """Return submitted model RPCs without evidence that the worker stopped."""
        with self._lock:
            return deepcopy(self._unconfirmed_model_requests)

    def observe_planner(
        self,
        planner: str,
        model: str | None,
        stats: Mapping[str, Any],
        *,
        failed: bool = False,
    ) -> None:
        """Import a single planner's fallback session counters if direct evidence is absent.

        Built-in model clients record source events whenever available. The Flash
        planner replays a stored program and its explicit zero applies to planning.
        """
        with self._lock:
            if any(e["module"] == "planner" for e in self._entries.values()):
                return
        supplied = {}
        inp = out = None
        if planner == "flash":
            inp = out = 0
            model = "none"
        else:
            supplied = {
                k: v
                for k, v in stats.items()
                if k.startswith("total_") and "tokens" in k
            }
            measured = stats.get("usage_observed")
            if measured is True or (measured is None and any(supplied.values())):
                inp, out = (
                    supplied.get("total_input_tokens"),
                    supplied.get("total_output_tokens"),
                )
                if planner == "claude_code" and inp is not None:
                    inp += (supplied.get("total_cache_read_input_tokens") or 0) + (
                        supplied.get("total_cache_creation_input_tokens") or 0
                    )
        self.record_usage(
            request_id="planner-session",
            model=model or "unknown",
            provider=planner,
            module="planner",
            stage="online_unattributed",
            granularity="session",
            input_tokens=inp,
            output_tokens=out,
            status="failed" if failed else "completed",
            # A legacy session summary cannot establish per-request coverage.
            coverage="partial"
            if planner != "flash" and (inp is not None or out is not None)
            else None,
            raw_usage=supplied,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready ledger; an empty ledger has unknown coverage."""
        with self._lock:
            entries = deepcopy(list(self._entries.values()))
        coverage = {e["coverage"] for e in entries}
        applicable = coverage - {"not_applicable"}
        if not applicable:
            overall = "not_applicable" if coverage else "unknown"
        elif len(applicable) == 1:
            overall = applicable.pop()
        else:
            overall = "partial"
        return {"coverage": overall, "entries": entries}
