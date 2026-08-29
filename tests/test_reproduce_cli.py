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

from types import SimpleNamespace

import pytest

from rpent.cli import reproduce
from rpent.cli.reproduce import (
    DEFAULT_MEMORY_REPO_ID,
    _gpus,
    _memory_root,
    _parser,
    _planner_transport,
    _resolve_run_cells,
)
from rpent.reproduce.robocasa.executor import PINNED_CODEX_PATH

API_BASE_URL = "https://planner.example/v1"


def test_reproduction_parser_freezes_preliminary_dual_gpu_run(tmp_path):
    args = _parser().parse_args(
        [
            "robocasa",
            "run",
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--memory-dir",
            str(tmp_path / "memory"),
            "--results-root",
            str(tmp_path / "results"),
            "--rollout-root",
            str(tmp_path / "rollouts"),
            "--planner-auth-mode",
            "api-key",
            "--api-key-file",
            str(tmp_path / "secret"),
            "--base-url",
            API_BASE_URL,
            "--preliminary-local-runtime",
            "--selection",
            "smoke-v1",
        ]
    )
    assert args.preliminary_local_runtime is True
    assert args.gpus == (0, 1)
    assert args.planner_profile == "codex-gpt55-xhigh"
    assert args.selection == "smoke-v1"
    assert args.codex_bin == PINNED_CODEX_PATH


@pytest.mark.parametrize("raw", ["", "0,0", "-1", "gpu0"])
def test_gpu_parser_rejects_invalid_values(raw):
    with pytest.raises(Exception):
        _gpus(raw)


def test_gpu_parser_accepts_two_unique_devices():
    assert _gpus("0, 1") == (0, 1)


def test_huggingface_memory_is_materialized_without_cache_symlinks(
    tmp_path, monkeypatch
):
    revision = "a" * 40
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = kwargs["local_dir"]
        memory = local_dir / "robocasa" / "harness_vla_v1"
        memory.mkdir(parents=True)
        (memory / "manifest.json").write_text("{}\n")
        return str(local_dir)

    monkeypatch.setenv("RPENT_ROBOCASA_MEMORY_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    args = _parser().parse_args(
        [
            "robocasa",
            "run",
            "--runtime-root",
            "/runtime",
            "--memory-revision",
            revision,
            "--results-root",
            "/results",
            "--rollout-root",
            "/rollouts",
            "--planner-auth-mode",
            "api-key",
            "--api-key-file",
            "/secret",
            "--base-url",
            API_BASE_URL,
        ]
    )

    root = _memory_root(args)

    assert root.name == "harness_vla_v1"
    assert calls[0]["repo_id"] == DEFAULT_MEMORY_REPO_ID
    assert calls[0]["revision"] == revision
    assert calls[0]["local_dir"].is_dir()
    assert not calls[0]["local_dir"].is_symlink()


def test_huggingface_memory_requires_immutable_revision(tmp_path, monkeypatch):
    monkeypatch.setenv("RPENT_ROBOCASA_MEMORY_CACHE", str(tmp_path / "cache"))
    args = _parser().parse_args(
        [
            "robocasa",
            "run",
            "--runtime-root",
            "/runtime",
            "--memory-repo-id",
            "org/memory",
            "--memory-revision",
            "main",
            "--results-root",
            "/results",
            "--rollout-root",
            "/rollouts",
            "--planner-auth-mode",
            "api-key",
            "--api-key-file",
            "/secret",
            "--base-url",
            API_BASE_URL,
        ]
    )
    with pytest.raises(ValueError, match="immutable"):
        _memory_root(args)


def _parse_run_selection(*selection_args: str):
    return _parser().parse_args(
        [
            "robocasa",
            "run",
            "--runtime-root",
            "/runtime",
            "--memory-dir",
            "/memory",
            "--results-root",
            "/results",
            "--rollout-root",
            "/rollouts",
            "--planner-auth-mode",
            "api-key",
            "--api-key-file",
            "/secret",
            "--base-url",
            API_BASE_URL,
            *selection_args,
        ]
    )


def test_run_can_select_exactly_one_frozen_cell():
    args = _parse_run_selection(
        "--split",
        "atomic",
        "--task",
        "OpenDrawer",
        "--seed",
        "1",
    )

    cells = _resolve_run_cells(args)

    assert len(cells) == 1
    assert cells[0].split.value == "atomic"
    assert cells[0].task == "OpenDrawer"
    assert cells[0].seed == 1


@pytest.mark.parametrize(
    "selection_args",
    [
        ("--split", "atomic"),
        ("--task", "OpenDrawer", "--seed", "1"),
        (
            "--selection",
            "smoke-v1",
            "--split",
            "atomic",
            "--task",
            "OpenDrawer",
            "--seed",
            "1",
        ),
        (
            "--split",
            "composite_seen",
            "--task",
            "OpenDrawer",
            "--seed",
            "1",
        ),
        (
            "--split",
            "atomic",
            "--task",
            "OpenDrawer",
            "--seed",
            "11",
        ),
    ],
)
def test_single_cell_selection_fails_closed(selection_args):
    args = _parse_run_selection(*selection_args)

    with pytest.raises(ValueError):
        _resolve_run_cells(args)


def test_run_without_selection_defaults_to_full_matrix():
    assert len(_resolve_run_cells(_parse_run_selection())) == 340


def test_publication_gate_is_explicit_for_validate_and_summarize(tmp_path):
    validate = _parser().parse_args(
        [
            "robocasa",
            "validate",
            "--results-root",
            str(tmp_path),
            "--require-publication-ready",
        ]
    )
    summarize = _parser().parse_args(
        [
            "robocasa",
            "summarize",
            "--results-root",
            str(tmp_path),
            "--require-publication-ready",
        ]
    )
    assert validate.require_publication_ready is True
    assert summarize.require_publication_ready is True


def test_run_delegates_preflight_to_the_locked_runner(tmp_path, monkeypatch):
    config = SimpleNamespace(results_root=tmp_path / "results")
    events = []

    monkeypatch.setattr(reproduce, "_executor_config", lambda _args: config)
    monkeypatch.setattr(reproduce, "_resolve_run_cells", lambda _args: ())

    def doctor(_config, *, verify_checkpoint, verify_isolation):
        events.append(("doctor", verify_checkpoint, verify_isolation))
        return {"ok": True}

    def run_cells(_config, _cells, **kwargs):
        events.append(("runner",))
        assert kwargs["preflight"]() == {"ok": True}
        return {"complete": True}

    monkeypatch.setattr(reproduce, "doctor", doctor)
    monkeypatch.setattr(reproduce, "run_cells", run_cells)

    rc = reproduce.main(
        [
            "robocasa",
            "run",
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--memory-dir",
            str(tmp_path / "memory"),
            "--results-root",
            str(tmp_path / "results"),
            "--rollout-root",
            str(tmp_path / "rollouts"),
            "--planner-auth-mode",
            "api-key",
            "--api-key-file",
            str(tmp_path / "secret"),
            "--base-url",
            API_BASE_URL,
        ]
    )

    assert rc == 0
    assert events == [("runner",), ("doctor", True, True)]


def test_doctor_delegates_mutating_checks_to_the_results_lock(tmp_path, monkeypatch):
    config = SimpleNamespace(results_root=tmp_path / "results")
    events = []
    monkeypatch.setattr(reproduce, "_executor_config", lambda _args: config)

    def doctor(_config, *, verify_checkpoint, verify_isolation):
        events.append(("doctor", verify_checkpoint, verify_isolation))
        return {"ok": True}

    def locked(results_root, callback):
        events.append(("lock", results_root))
        return callback()

    monkeypatch.setattr(reproduce, "doctor", doctor)
    monkeypatch.setattr(reproduce, "run_locked_preflight", locked)

    rc = reproduce.main(
        [
            "robocasa",
            "doctor",
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--memory-dir",
            str(tmp_path / "memory"),
            "--results-root",
            str(tmp_path / "results"),
            "--rollout-root",
            str(tmp_path / "rollouts"),
            "--planner-auth-mode",
            "api-key",
            "--api-key-file",
            str(tmp_path / "secret"),
            "--base-url",
            API_BASE_URL,
            "--verify-checkpoint",
            "--verify-isolation",
        ]
    )

    assert rc == 0
    assert events == [
        ("lock", config.results_root),
        ("doctor", True, True),
    ]


def _parse_transport_args(*transport_args: str):
    return _parser().parse_args(
        [
            "robocasa",
            "run",
            "--runtime-root",
            "/runtime",
            "--memory-dir",
            "/memory",
            "--results-root",
            "/results",
            "--rollout-root",
            "/rollouts",
            *transport_args,
        ]
    )


def test_subscription_auth_builds_loopback_broker_transport():
    args = _parse_transport_args(
        "--planner-auth-mode",
        "chatgpt-subscription",
        "--broker-credential-file",
        "/broker-capability",
        "--broker-base-url",
        "http://127.0.0.1:8765/v1",
        "--broker-health-url",
        "http://127.0.0.1:8765/health",
    )

    transport = _planner_transport(args)

    assert transport.auth_mode == "chatgpt-subscription"
    assert transport.credential_file.as_posix() == "/broker-capability"
    assert transport.request_base_url == "http://127.0.0.1:8765/v1"
    assert transport.broker_health_url == "http://127.0.0.1:8765/health"
    assert transport.provider_id == "rpent_chatgpt_broker"


@pytest.mark.parametrize(
    "transport_args",
    [
        ("--planner-auth-mode", "api-key", "--api-key-file", "/secret"),
        ("--planner-auth-mode", "api-key", "--base-url", API_BASE_URL),
        (
            "--planner-auth-mode",
            "chatgpt-subscription",
            "--broker-base-url",
            "http://127.0.0.1:8765/v1",
            "--broker-health-url",
            "http://127.0.0.1:8765/health",
        ),
        (
            "--planner-auth-mode",
            "chatgpt-subscription",
            "--broker-credential-file",
            "/broker-capability",
            "--broker-health-url",
            "http://127.0.0.1:8765/health",
        ),
        (
            "--planner-auth-mode",
            "chatgpt-subscription",
            "--broker-credential-file",
            "/broker-capability",
            "--broker-base-url",
            "http://127.0.0.1:8765/v1",
        ),
    ],
)
def test_planner_auth_rejects_missing_mode_specific_options(transport_args):
    args = _parse_transport_args(*transport_args)

    with pytest.raises(ValueError, match="requires"):
        _planner_transport(args)


@pytest.mark.parametrize(
    "transport_args",
    [
        (
            "--planner-auth-mode",
            "api-key",
            "--api-key-file",
            "/secret",
            "--base-url",
            API_BASE_URL,
            "--broker-health-url",
            "http://127.0.0.1:8765/health",
        ),
        (
            "--planner-auth-mode",
            "chatgpt-subscription",
            "--broker-credential-file",
            "/broker-capability",
            "--broker-base-url",
            "http://127.0.0.1:8765/v1",
            "--broker-health-url",
            "http://127.0.0.1:8765/health",
            "--api-key-file",
            "/secret",
        ),
    ],
)
def test_planner_auth_rejects_cross_mode_options(transport_args):
    args = _parse_transport_args(*transport_args)

    with pytest.raises(ValueError, match="cannot be combined"):
        _planner_transport(args)


def test_parser_requires_explicit_planner_auth_mode(monkeypatch):
    monkeypatch.delenv("RPENT_ROBOCASA_PLANNER_AUTH_MODE", raising=False)

    with pytest.raises(SystemExit):
        _parse_transport_args(
            "--api-key-file",
            "/secret",
            "--base-url",
            API_BASE_URL,
        )
