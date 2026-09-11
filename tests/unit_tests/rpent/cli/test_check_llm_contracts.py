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

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from rpent.cli import check_llm as check_cli
from rpent.planner.check import (
    CHECK_STATUSES,
    STATUS_AUTH_FAILED,
    STATUS_MISSING_API_KEY,
    STATUS_OK,
    LlmCheckRequest,
    LlmCheckResult,
)


def _stub_check(
    monkeypatch: pytest.MonkeyPatch, result: LlmCheckResult
) -> dict[str, Any]:
    """Replace the shared check with a stub and capture its request."""
    captured: dict[str, Any] = {}

    def fake_check(request: LlmCheckRequest) -> LlmCheckResult:
        captured["request"] = request
        return result

    monkeypatch.setattr(check_cli, "check_llm", fake_check)
    return captured


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["rpent-check-llm", *argv])
    return check_cli.main()


def test_cli_forwards_every_flag_to_the_shared_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured = _stub_check(
        monkeypatch,
        LlmCheckResult(
            ok=True,
            status=STATUS_OK,
            planner="api",
            model="anthropic:m",
            reply="ok",
            latency_s=0.42,
        ),
    )

    exit_code = _run(
        monkeypatch,
        "--planner",
        "api",
        "--model",
        "anthropic:m",
        "--base-url",
        "https://gateway.example",
        "--timeout-s",
        "7",
    )

    assert exit_code == 0
    request = captured["request"]
    assert request.planner == "api"
    assert request.model == "anthropic:m"
    assert request.base_url == "https://gateway.example"
    assert request.timeout_s == 7


def test_cli_defaults_match_the_run_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _stub_check(
        monkeypatch,
        LlmCheckResult(ok=True, status=STATUS_OK, planner="api", reply="ok"),
    )

    _run(monkeypatch)

    request = captured["request"]
    assert request.planner == "api"
    assert request.model is None
    assert request.base_url is None
    # Deliberately unset so the check applies its own short default.
    assert request.timeout_s is None


def test_cli_exits_zero_on_success_and_one_on_any_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_check(
        monkeypatch,
        LlmCheckResult(ok=True, status=STATUS_OK, planner="api", reply="ok"),
    )
    assert _run(monkeypatch) == 0

    _stub_check(
        monkeypatch,
        LlmCheckResult(ok=False, status=STATUS_AUTH_FAILED, planner="api"),
    )
    assert _run(monkeypatch) == 1


def test_cli_success_report_shows_reply_and_latency(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_check(
        monkeypatch,
        LlmCheckResult(
            ok=True,
            status=STATUS_OK,
            planner="api",
            model="anthropic:m",
            credential_env="ANTHROPIC_API_KEY",
            credential_present=True,
            reply="ok",
            latency_s=0.42,
        ),
    )

    _run(monkeypatch)
    out = capsys.readouterr().out

    assert "OK" in out
    assert "0.42s" in out
    assert "ANTHROPIC_API_KEY (set)" in out


def test_cli_failure_report_carries_status_detail_and_a_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_check(
        monkeypatch,
        LlmCheckResult(
            ok=False,
            status=STATUS_MISSING_API_KEY,
            planner="api",
            model="anthropic:m",
            credential_env="ANTHROPIC_API_KEY",
            credential_present=False,
            detail="ANTHROPIC_API_KEY is not set for provider 'anthropic'.",
        ),
    )

    _run(monkeypatch)
    out = capsys.readouterr().out

    assert "FAILED" in out
    assert STATUS_MISSING_API_KEY in out
    assert "ANTHROPIC_API_KEY (not set)" in out
    # Remediation prose is rendered by the CLI, not the shared layer.
    assert "Set ANTHROPIC_API_KEY" in out


def test_cli_json_output_is_parseable_and_matches_the_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = LlmCheckResult(
        ok=False,
        status=STATUS_AUTH_FAILED,
        planner="api",
        model="anthropic:m",
        credential_env="ANTHROPIC_API_KEY",
        credential_present=True,
        detail="ModelHTTPError: status_code: 401",
        latency_s=0.31,
    )
    _stub_check(monkeypatch, result)

    exit_code = _run(monkeypatch, "--json")
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload == result.as_dict()
    # --json is the machine surface: no prose leaks into it.
    assert "hint" not in payload


@pytest.mark.parametrize(
    ("planner", "env_var"),
    [("claude_code", "ANTHROPIC_BASE_URL"), ("codex", "CODEX_BASE_URL")],
)
def test_cli_rejects_base_url_for_the_backends_that_ignore_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    planner: str,
    env_var: str,
) -> None:
    """A flag the run would drop must not be honoured by the check.

    ``build_planner`` forwards ``base_url`` to the api model alone, so
    accepting it here would test an endpoint the run never uses.
    """
    _stub_check(monkeypatch, LlmCheckResult(ok=True, status=STATUS_OK, planner=planner))

    with pytest.raises(SystemExit):
        _run(monkeypatch, "--planner", planner, "--base-url", "https://gateway.example")

    assert env_var in capsys.readouterr().err


def test_cli_keeps_base_url_for_the_api_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _stub_check(
        monkeypatch, LlmCheckResult(ok=True, status=STATUS_OK, planner="api")
    )

    _run(
        monkeypatch,
        "--planner",
        "api",
        "--model",
        "anthropic:m",
        "--base-url",
        "https://gateway.example",
    )

    assert captured["request"].base_url == "https://gateway.example"


def test_cli_rejects_an_unknown_planner_at_the_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["rpent-check-llm", "--planner", "telepathy"])

    with pytest.raises(SystemExit) as excinfo:
        check_cli.main()

    assert excinfo.value.code == 2


def test_every_failure_status_has_remediation_text() -> None:
    for status in CHECK_STATUSES:
        if status == STATUS_OK:
            continue
        hint = check_cli._remediation(
            LlmCheckResult(ok=False, status=status, planner="api")
        )
        assert hint, f"no remediation text for {status}"


def test_multiline_detail_stays_aligned_in_the_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_check(
        monkeypatch,
        LlmCheckResult(
            ok=False,
            status=STATUS_AUTH_FAILED,
            planner="api",
            detail="first line\nsecond line",
        ),
    )

    _run(monkeypatch)
    lines = capsys.readouterr().out.splitlines()
    continuation = [ln for ln in lines if ln.strip() == "second line"]

    assert continuation, "the wrapped detail line is missing"
    assert continuation[0].startswith("    "), "continuation is not indented"
