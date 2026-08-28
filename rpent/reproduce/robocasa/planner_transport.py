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

"""Planner authentication and transport contracts for RoboCasa reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

API_KEY_AUTH = "api-key"
CHATGPT_SUBSCRIPTION_AUTH = "chatgpt-subscription"
AUTH_MODES = (API_KEY_AUTH, CHATGPT_SUBSCRIPTION_AUTH)

CHATGPT_BROKER_PROFILE = "chatgpt_subscription_broker"
CHATGPT_BROKER_PROTOCOL = "root_oauth_injection_v1"
CHATGPT_ENDPOINT_IDENTITY = "openai_chatgpt_subscription"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})

# Keep transient provider failures inside the benchmark deadline instead of
# prematurely consuming cell-level infrastructure retries. The deadline
# supervisor remains the authoritative upper bound for every planner process.
CODEX_REQUEST_MAX_RETRIES = 12
CODEX_STREAM_MAX_RETRIES = 120
CODEX_STREAM_IDLE_TIMEOUT_MS = 330_000


def codex_provider_retry_policy() -> dict[str, int]:
    """Return the frozen, credential-free Codex provider retry identity."""
    return {
        "request_max_retries": CODEX_REQUEST_MAX_RETRIES,
        "stream_max_retries": CODEX_STREAM_MAX_RETRIES,
        "stream_idle_timeout_ms": CODEX_STREAM_IDLE_TIMEOUT_MS,
    }


def normalize_responses_base_url(base_url: str) -> str:
    """Validate and normalize one Responses-compatible API base URL."""
    raw = base_url.strip().rstrip("/")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base URL must be an absolute http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain credentials, query, or fragment")
    if parsed.scheme == "http" and parsed.hostname not in LOOPBACK_HOSTS:
        raise ValueError("non-loopback planner base URL must use https")
    path = parsed.path.rstrip("/")
    if path.endswith("/responses"):
        raise ValueError("base URL must identify the API base, not /responses")
    if not path.endswith("/v1"):
        path = f"{path}/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def normalize_broker_health_url(raw_url: str) -> str:
    """Accept only a loopback HTTP broker health endpoint."""
    raw = raw_url.strip().rstrip("/")
    parsed = urlsplit(raw)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in LOOPBACK_HOSTS
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"/health", "/lease-health"}
    ):
        raise ValueError(
            "broker health URL must be loopback HTTP ending in /health or /lease-health"
        )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


@dataclass(frozen=True)
class PlannerTransport:
    """Secret-free transport identity plus a root-only client credential path."""

    auth_mode: str
    credential_file: Path
    request_base_url: str
    endpoint_identity: str
    provider_id: str
    provider_name: str
    broker_health_url: str | None = None
    broker_protocol: str | None = None

    @classmethod
    def api_key(cls, *, credential_file: Path, base_url: str) -> "PlannerTransport":
        normalized = normalize_responses_base_url(base_url)
        return cls(
            auth_mode=API_KEY_AUTH,
            credential_file=Path(credential_file),
            request_base_url=normalized,
            endpoint_identity=f"responses_api:{normalized}",
            provider_id="rpent_responses_api",
            provider_name="RPent Responses API",
        )

    @classmethod
    def chatgpt_subscription(
        cls,
        *,
        credential_file: Path,
        broker_base_url: str,
        broker_health_url: str,
    ) -> "PlannerTransport":
        normalized_base = normalize_responses_base_url(broker_base_url)
        parsed = urlsplit(normalized_base)
        if parsed.scheme != "http" or parsed.hostname not in LOOPBACK_HOSTS:
            raise ValueError("ChatGPT subscription broker must use loopback HTTP")
        normalized_health = normalize_broker_health_url(broker_health_url)
        if urlsplit(normalized_health).netloc != parsed.netloc:
            raise ValueError("broker base and health URLs must use the same listener")
        return cls(
            auth_mode=CHATGPT_SUBSCRIPTION_AUTH,
            credential_file=Path(credential_file),
            request_base_url=normalized_base,
            endpoint_identity=CHATGPT_ENDPOINT_IDENTITY,
            provider_id="rpent_chatgpt_broker",
            provider_name="RPent ChatGPT Subscription Broker",
            broker_health_url=normalized_health,
            broker_protocol=CHATGPT_BROKER_PROTOCOL,
        )

    @property
    def uses_broker(self) -> bool:
        return self.auth_mode == CHATGPT_SUBSCRIPTION_AUTH

    def manifest_identity(self) -> dict[str, str | bool | None]:
        """Return the stable, credential-free transport identity."""
        return {
            "auth_mode": self.auth_mode,
            "provider": self.provider_id,
            "endpoint_identity": self.endpoint_identity,
            "credential_broker": self.uses_broker,
            "credential_broker_protocol": self.broker_protocol,
        }
