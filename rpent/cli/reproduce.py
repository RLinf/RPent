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

"""Standalone, fail-closed reproduction CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

from rpent.reproduce.robocasa.executor import (
    PINNED_CODEX_PATH,
    ExecutorConfig,
    RuntimePaths,
    doctor,
)
from rpent.reproduce.robocasa.memory import (
    pack_memory,
    validate_memory_pack,
)
from rpent.reproduce.robocasa.planner_transport import (
    API_KEY_AUTH,
    AUTH_MODES,
    CHATGPT_SUBSCRIPTION_AUTH,
    PlannerTransport,
)
from rpent.reproduce.robocasa.profiles import PROFILES, get_profile
from rpent.reproduce.robocasa.protocol import Split, cell_for
from rpent.reproduce.robocasa.runner import (
    PreflightFailed,
    run_cells,
    run_locked_preflight,
    select_cells,
)
from rpent.reproduce.robocasa.validator import summarize, validate_cell

DEFAULT_MEMORY_REPO_ID = "RLinf/RPent-memory"


def _path(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def _gpus(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU ids must be comma-separated integers"
        ) from exc
    if (
        not values
        or len(values) != len(set(values))
        or any(value < 0 for value in values)
    ):
        raise argparse.ArgumentTypeError("GPU ids must be unique non-negative integers")
    return values


def _memory_root(args: argparse.Namespace) -> Path:
    if args.memory_dir is not None:
        if args.memory_repo_id is not None or args.memory_revision is not None:
            raise ValueError(
                "--memory-dir cannot be combined with Hugging Face options"
            )
        return args.memory_dir
    repo_id = args.memory_repo_id or DEFAULT_MEMORY_REPO_ID
    args.memory_repo_id = repo_id
    if args.memory_revision is None:
        raise ValueError(
            "provide --memory-dir or an immutable --memory-revision; "
            f"--memory-repo-id defaults to {DEFAULT_MEMORY_REPO_ID}"
        )
    revision = args.memory_revision
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(
            "--memory-revision must be an immutable 40-character commit SHA"
        )
    from huggingface_hub import snapshot_download

    cache_root = _path(
        os.environ.get(
            "RPENT_ROBOCASA_MEMORY_CACHE",
            str(Path.home() / ".cache" / "rpent" / "robocasa-memory"),
        )
    )
    _prepare_root(cache_root, "materialized memory cache", 0o700)
    cache_key = hashlib.sha256(f"{repo_id}\0{revision}".encode("utf-8")).hexdigest()
    materialized = cache_root / cache_key
    _prepare_root(materialized, "materialized memory snapshot", 0o700)

    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=["robocasa/harness_vla_v1/**"],
            local_dir=materialized,
        )
    )
    if snapshot.resolve(strict=True) != materialized:
        raise ValueError(
            "Hugging Face returned an unexpected materialized snapshot path"
        )
    return snapshot / "robocasa" / "harness_vla_v1"


def _prepare_root(path: Path, label: str, mode: int) -> Path:
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} must be a real directory: {path}")
        if metadata.st_uid != os.geteuid():
            raise ValueError(f"{label} must be owned by the reproduction user: {path}")
        if stat.S_IMODE(metadata.st_mode) != mode:
            raise ValueError(f"{label} must have exact mode {mode:04o}: {path}")
        return path
    path.mkdir(parents=True, mode=mode)
    path.chmod(mode)
    return path


def _planner_transport(args: argparse.Namespace) -> PlannerTransport:
    api_values = (args.api_key_file, args.base_url)
    broker_values = (
        args.broker_credential_file,
        args.broker_base_url,
        args.broker_health_url,
    )
    if args.planner_auth_mode == API_KEY_AUTH:
        if any(value is not None for value in broker_values):
            raise ValueError(
                "API-key auth cannot be combined with ChatGPT broker options"
            )
        if any(value is None for value in api_values):
            raise ValueError("API-key auth requires --api-key-file and --base-url")
        return PlannerTransport.api_key(
            credential_file=args.api_key_file,
            base_url=args.base_url,
        )
    if args.planner_auth_mode == CHATGPT_SUBSCRIPTION_AUTH:
        if any(value is not None for value in api_values):
            raise ValueError(
                "ChatGPT subscription auth cannot be combined with API-key options"
            )
        if any(value is None for value in broker_values):
            raise ValueError(
                "ChatGPT subscription auth requires --broker-credential-file, "
                "--broker-base-url, and --broker-health-url"
            )
        return PlannerTransport.chatgpt_subscription(
            credential_file=args.broker_credential_file,
            broker_base_url=args.broker_base_url,
            broker_health_url=args.broker_health_url,
        )
    raise ValueError(f"unsupported planner auth mode: {args.planner_auth_mode!r}")


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime-root",
        type=_path,
        default=(
            _path(os.environ["RPENT_ROBOCASA_RUNTIME_ROOT"])
            if os.environ.get("RPENT_ROBOCASA_RUNTIME_ROOT")
            else None
        ),
        required=not bool(os.environ.get("RPENT_ROBOCASA_RUNTIME_ROOT")),
    )
    parser.add_argument(
        "--memory-dir",
        type=_path,
        help="local memory pack override for offline or development runs",
    )
    parser.add_argument(
        "--memory-repo-id",
        default=os.environ.get("RPENT_ROBOCASA_MEMORY_REPO_ID"),
        help=(
            "Hugging Face dataset containing robocasa/harness_vla_v1 "
            f"(default: {DEFAULT_MEMORY_REPO_ID})"
        ),
    )
    parser.add_argument(
        "--memory-revision",
        default=os.environ.get("RPENT_ROBOCASA_MEMORY_REVISION"),
        help="immutable lowercase 40-character Hugging Face commit SHA",
    )
    parser.add_argument("--results-root", type=_path, required=True)
    parser.add_argument("--rollout-root", type=_path, required=True)
    parser.add_argument(
        "--planner-profile",
        choices=tuple(PROFILES),
        default="codex-gpt55-xhigh",
    )
    auth_mode = os.environ.get("RPENT_ROBOCASA_PLANNER_AUTH_MODE")
    parser.add_argument(
        "--planner-auth-mode",
        choices=AUTH_MODES,
        default=auth_mode,
        required=auth_mode is None,
    )
    parser.add_argument(
        "--api-key-file",
        type=_path,
        default=(
            _path(os.environ["CODEX_API_KEY_FILE"])
            if os.environ.get("CODEX_API_KEY_FILE")
            else None
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CODEX_BASE_URL"),
    )
    parser.add_argument(
        "--broker-credential-file",
        type=_path,
        default=(
            _path(os.environ["CODEX_BROKER_CREDENTIAL_FILE"])
            if os.environ.get("CODEX_BROKER_CREDENTIAL_FILE")
            else None
        ),
    )
    parser.add_argument(
        "--broker-base-url",
        default=os.environ.get("CODEX_BROKER_BASE_URL"),
    )
    parser.add_argument(
        "--broker-health-url",
        default=os.environ.get("CODEX_BROKER_HEALTH_URL"),
    )
    parser.add_argument("--keep-workdirs", action="store_true")
    parser.add_argument(
        "--preliminary-local-runtime",
        action="store_true",
        help=(
            "explicitly acknowledge that --runtime-root is a local hybrid snapshot "
            "for preliminary parity testing, not a release runtime"
        ),
    )
    parser.add_argument("--codex-bin", type=_path, default=PINNED_CODEX_PATH)


def _executor_config(args: argparse.Namespace) -> ExecutorConfig:
    memory = _memory_root(args)
    problems = validate_memory_pack(memory)
    if problems:
        raise ValueError("invalid memory pack: " + "; ".join(problems))
    results_root = _prepare_root(args.results_root, "results root", 0o700)
    rollout_root = _prepare_root(args.rollout_root, "rollout root", 0o711)
    if (
        results_root == rollout_root
        or results_root in rollout_root.parents
        or rollout_root in results_root.parents
    ):
        raise ValueError("--results-root and --rollout-root must not overlap")
    local_memory = args.memory_dir is not None
    planner_transport = _planner_transport(args)
    return ExecutorConfig(
        runtime=RuntimePaths.discover(args.runtime_root),
        results_root=results_root,
        rollout_root=rollout_root,
        memory_root=memory,
        planner_profile=get_profile(args.planner_profile),
        planner_transport=planner_transport,
        codex_bin=args.codex_bin,
        keep_workdirs=args.keep_workdirs,
        preliminary_local_runtime=args.preliminary_local_runtime,
        memory_source=("local" if local_memory else args.memory_repo_id),
        memory_revision=(None if local_memory else args.memory_revision),
    )


def _resolve_run_cells(args: argparse.Namespace):
    """Resolve either a named run or exactly one frozen protocol cell."""
    supplied = (
        args.split is not None,
        args.task is not None,
        args.seed is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError("single-cell runs require --split, --task, and --seed")
    if all(supplied):
        if args.selection is not None:
            raise ValueError(
                "--selection cannot be combined with --split, --task, and --seed"
            )
        return (cell_for(args.split, args.task, args.seed),)
    return select_cells(args.selection or "full")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rpent-reproduce")
    robots = parser.add_subparsers(dest="robot", required=True)
    robocasa = robots.add_parser("robocasa")
    commands = robocasa.add_subparsers(dest="command", required=True)

    pack = commands.add_parser("memory-pack")
    pack.add_argument("--migration-root", type=_path, required=True)
    pack.add_argument("--output", type=_path, required=True)
    memory_validate = commands.add_parser("memory-validate")
    memory_validate.add_argument("--memory-dir", type=_path, required=True)

    doctor_parser = commands.add_parser("doctor")
    _add_runtime_args(doctor_parser)
    doctor_parser.add_argument("--verify-checkpoint", action="store_true")
    doctor_parser.add_argument("--verify-isolation", action="store_true")

    run = commands.add_parser("run")
    _add_runtime_args(run)
    run.add_argument(
        "--selection",
        choices=(
            "full",
            "smoke-v1",
            "atomic",
            "composite_seen",
            "composite_unseen",
        ),
        help="named protocol selection (defaults to full when no cell is provided)",
    )
    run.add_argument("--split", choices=tuple(item.value for item in Split))
    run.add_argument("--task")
    run.add_argument("--seed", type=int)
    run.add_argument("--gpus", type=_gpus, default=(0, 1))
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--retry-backoff-seconds", type=float, default=60.0)

    validate = commands.add_parser("validate")
    validate.add_argument("--results-root", type=_path, required=True)
    validate.add_argument("--split", choices=tuple(item.value for item in Split))
    validate.add_argument("--task")
    validate.add_argument("--seed", type=int)
    validate.add_argument("--require-publication-ready", action="store_true")
    summary = commands.add_parser("summarize")
    summary.add_argument("--results-root", type=_path, required=True)
    summary.add_argument("--output", type=_path)
    summary.add_argument("--require-publication-ready", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "memory-pack":
            value = pack_memory(args.migration_root, args.output)
            print(json.dumps(value, ensure_ascii=False, indent=2))
            return 0
        if args.command == "memory-validate":
            problems = validate_memory_pack(args.memory_dir)
            print(json.dumps({"valid": not problems, "problems": problems}, indent=2))
            return int(bool(problems))
        if args.command == "doctor":
            config = _executor_config(args)
            value = run_locked_preflight(
                config.results_root,
                lambda: doctor(
                    config,
                    verify_checkpoint=args.verify_checkpoint,
                    verify_isolation=args.verify_isolation,
                ),
            )
            print(json.dumps(value, ensure_ascii=False, indent=2))
            return int(not value["ok"])
        if args.command == "run":
            cells = _resolve_run_cells(args)
            config = _executor_config(args)
            try:
                value = run_cells(
                    config,
                    cells,
                    gpus=args.gpus,
                    max_attempts=args.max_attempts,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                    preflight=lambda: doctor(
                        config,
                        verify_checkpoint=True,
                        verify_isolation=True,
                    ),
                )
            except PreflightFailed as exc:
                print(json.dumps(exc.report, ensure_ascii=False, indent=2))
                return 78
            print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
            return 0 if value["complete"] else 76
        if args.command == "validate":
            supplied = (
                args.split is not None,
                args.task is not None,
                args.seed is not None,
            )
            if any(supplied) and not all(supplied):
                parser.error("cell validation requires --split, --task, and --seed")
            if all(supplied):
                if args.require_publication_ready:
                    parser.error(
                        "--require-publication-ready applies only to full-run validation"
                    )
                validation = validate_cell(
                    args.results_root,
                    cell_for(args.split, args.task, args.seed),
                )
                value = {
                    "canonical": validation.result.canonical,
                    "problems": validation.problems,
                }
                print(json.dumps(value, ensure_ascii=False, indent=2))
                return int(not validation.result.canonical)
            value = summarize(args.results_root)
            print(json.dumps(value, ensure_ascii=False, indent=2))
            gate = "publication_ready" if args.require_publication_ready else "complete"
            return int(not value[gate])
        value = summarize(args.results_root)
        rendered = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        gate = "publication_ready" if args.require_publication_ready else "complete"
        return int(not value[gate])
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
