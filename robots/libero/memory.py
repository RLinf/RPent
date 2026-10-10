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

"""Model selection and complete, revision-isolated LIBERO memory downloads."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from rpent.evaluation import write_json_atomic
from rpent.memory.source import selected_source
from rpent.memory.source import source_record as build_source_record
from rpent.planner.base import resolve_model
from rpent.robots.robot_spec import RunConfig
from rpent.utils.config import get_memory_dir
from rpent.utils.logging import get_logger

logger = get_logger("memory")
GPT5 = "GPT_5.5_xhigh"
ASTRA = "GPT_6_astra_low"
MEMORY_VERSIONS = ("auto", GPT5, ASTRA)
DEFAULT_REPO = "RLinf/RPent-memory"


def validate_options(args: argparse.Namespace) -> None:
    """Check memory options without starting services or downloading files."""
    revision = getattr(args, "memory_revision", None)
    if revision is not None:
        if not revision.strip():
            raise ValueError("--memory-revision cannot be empty")
        if (
            getattr(args, "explore", False)
            or getattr(args, "memory_profile", None) == "local"
        ):
            raise ValueError("--memory-revision requires --memory-profile hf")
    if getattr(args, "memory_version", "auto") != "auto" and (
        getattr(args, "explore", False)
        or getattr(args, "memory_profile", None) == "local"
    ):
        raise ValueError(
            "--memory-version requires --memory-profile hf; local memory and exploration use --memory-dir"
        )
    if (
        getattr(args, "planner", None) == "flash"
        and getattr(args, "memory_version", "auto") == ASTRA
    ):
        raise ValueError(f"{ASTRA} has no Flash/Task Card replay assets; choose {GPT5}")


def prepare_memory(args: argparse.Namespace, config: RunConfig) -> None:
    """Bind the selected corpus and record its source before each task starts."""
    validate_options(args)
    profile = getattr(args, "memory_profile", None) or (
        "local" if getattr(args, "explore", False) else "hf"
    )
    if profile == "local":
        destination = Path(config.output_dir) / "memory_source.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            destination,
            {
                "schema_version": 1,
                "profile": "local",
                "local_dir": str(
                    Path(config.prompt_vars["memory_dir"]).expanduser().resolve()
                ),
                "resolved_commit": None,
            },
        )
        return
    selection = selected_source(args, "libero")
    version = select_version(
        getattr(args, "memory_version", "auto"), model=args.model, planner=args.planner
    )
    repo = None
    revision = getattr(args, "memory_revision", None) or "main"
    if selection is not None:
        repo, revision, recorded_version = selection
        if recorded_version is not None:
            version = recorded_version
    root = sync_version(
        version=version,
        revision=revision,
        repo_id=repo,
        cache_dir=get_memory_dir("libero") / ".versions",
        source_record=Path(config.output_dir) / "memory_source.json",
    )
    if args.planner == "flash":
        replay_directory(root)
    config.prompt_vars["memory_dir"] = root
    config.prompt_vars["memory_version"] = version


def select_version(
    version: str = "auto", *, model: str | None = None, planner: str = "api"
) -> str:
    """Select a corpus without changing the running model or reasoning effort."""
    if version not in MEMORY_VERSIONS:
        raise ValueError(f"Unknown memory version: {version!r}")
    if version != "auto":
        return version
    if planner == "flash":
        return GPT5
    resolved = resolve_model(planner, model)
    name = resolved.rsplit(":", 1)[-1].lower() if resolved else ""
    selected = {"gpt-5.5": GPT5, "gpt-6-astra": ASTRA}.get(name)
    if selected:
        return selected
    logger.warning(
        "No memory mapping for model %r (%s); using %s. "
        "Use --memory-version to select explicitly.",
        resolved,
        planner,
        GPT5,
    )
    return GPT5


def replay_directory(root: Path) -> Path:
    """Locate generated Flash plans or the published Task Card replay assets."""
    for name in ("flash", "task_card"):
        directory = root / name
        if directory.is_dir() and any(directory.glob("*_plan.json")):
            return directory
    raise ValueError(
        f"Memory {root} has no Flash/Task Card replay assets. "
        f"Select --memory-version {GPT5} or provide a local replay corpus."
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_relative(name: str) -> bool:
    if not isinstance(name, str):
        return False
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and name == path.as_posix()
    )


def _matches_files(root: Path, files: dict[str, str]) -> bool:
    try:
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        return (
            bool(files)
            and actual == set(files)
            and all(
                _safe_relative(name) and _hash(root / name) == digest
                for name, digest in files.items()
            )
        )
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return False


def _verified(root: Path) -> bool:
    try:
        receipt = json.loads((root.parent / f"{root.name}.receipt.json").read_text())
        return receipt.get("prefix") == f"libero/{root.name}/" and _matches_files(
            root, receipt["files"]
        )
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return False


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _is_commit(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def sync_version(
    *,
    version: str,
    cache_dir: Path,
    repo_id: str | None = None,
    revision: str = "main",
    output_dir: Path | None = None,
    source_record: Path | None = None,
) -> Path:
    """Download one complete corpus; reuse only its verified cache on outage.

    Published file hashes come from ``libero/manifest.json``. A receipt is
    written after every selected file has downloaded, then checked on every
    reuse. Ref pointers are scoped to the repository, requested revision and
    version. Historical unversioned layouts require their matching client.
    """
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    if version not in (GPT5, ASTRA):
        raise ValueError("sync_version requires a resolved memory version")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("memory revision cannot be empty")
    pinned_commit = revision.lower() if _is_commit(revision) else None
    repo_id = repo_id or os.environ.get("RPENT_MEMORY_HF_REPO", DEFAULT_REPO)
    key = hashlib.sha256(repo_id.encode()).hexdigest()[:20]
    base = Path(cache_dir).resolve() / key
    if source_record is not None:
        source_record = Path(source_record).expanduser().resolve()
        protected = [Path(cache_dir).resolve()]
        if output_dir is not None:
            protected.append(Path(output_dir).resolve())
        if any(source_record.is_relative_to(directory) for directory in protected):
            raise ValueError(
                "source record must be outside the memory cache and output corpus"
            )
    base.mkdir(parents=True, exist_ok=True)
    ref_key = hashlib.sha256(f"{revision}:{version}".encode()).hexdigest()
    ref = base / f"{ref_key}.json"
    with (base / "sync.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Resolve the requested ref before consulting its cache. Never fall
        # back to another version, or to a different explicitly pinned commit.
        offline = os.environ.get("HF_HUB_OFFLINE", "").upper() in (
            "1",
            "YES",
            "TRUE",
            "ON",
        )
        try:
            if offline:
                raise ConnectionError("HF_HUB_OFFLINE")
            info = HfApi().repo_info(
                repo_id, repo_type="dataset", revision=pinned_commit or revision
            )
        except Exception as exc:
            try:
                if pinned_commit is not None:
                    # An immutable request addresses its snapshot directly,
                    # including when it was originally fetched through a branch.
                    sha = pinned_commit
                else:
                    pointer = json.loads(ref.read_text())
                    if (
                        pointer.get("repo") != repo_id
                        or pointer.get("version") != version
                    ):
                        raise ValueError("cache pointer identity mismatch")
                    sha = pointer["commit"]
                if not _is_commit(sha):
                    raise ValueError("invalid cached commit")
                sha = sha.lower()
                root = base / "snapshots" / sha / version
                if not _verified(root):
                    raise ValueError("incomplete or modified cache")
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                AttributeError,
            ) as cache_exc:
                raise RuntimeError(
                    f"Cannot resolve {repo_id}@{revision}: no complete cache for {version}"
                ) from cache_exc
            logger.warning(
                "Memory Hub unavailable (%s); using verified %s",
                type(exc).__name__,
                root,
            )
        else:
            sha = info.sha
            if not _is_commit(sha) or (
                pinned_commit is not None and sha.lower() != pinned_commit
            ):
                raise ValueError(
                    "Hub response does not match the requested memory commit"
                )
            sha = sha.lower()
            root = base / "snapshots" / sha / version
            if not _verified(root):
                names = [entry.rfilename for entry in info.siblings]
                prefix = f"libero/{version}/"
                selected = [name for name in names if name.startswith(prefix)]
                if not selected:
                    raise ValueError(
                        f"{repo_id}@{sha} has no versioned {version} corpus. "
                        "Download the current versioned dataset; historical "
                        "unversioned layouts require the matching historical client."
                    )
                manifest_path = hf_hub_download(
                    repo_id,
                    "libero/manifest.json",
                    repo_type="dataset",
                    revision=sha,
                )
                manifest = json.loads(Path(manifest_path).read_text())
                expected = manifest["versions"][version]["files"]
                relative = [name.removeprefix(prefix) for name in selected]
                if any(not _safe_relative(name) for name in relative):
                    raise ValueError("Unsafe memory file path")
                if set(relative) != set(expected):
                    raise ValueError(
                        "Memory manifest does not match the selected file list"
                    )
                if "MEMORY.md" not in relative:
                    raise ValueError("Memory corpus is missing MEMORY.md")
                snapshot = Path(
                    snapshot_download(
                        repo_id,
                        repo_type="dataset",
                        revision=sha,
                        allow_patterns=selected,
                    )
                )
                root.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=root.parent) as staging:
                    corpus = Path(staging) / version
                    hashes = {}
                    for name in relative:
                        dest = corpus / name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(snapshot / prefix / name, dest)
                        hashes[name] = _hash(dest)
                    if hashes != expected:
                        raise ValueError("Memory file checksum mismatch")
                    if root.exists():
                        shutil.rmtree(root)
                    corpus.replace(root)
                    _write_json(
                        root.parent / f"{version}.receipt.json",
                        {"prefix": prefix, "files": hashes},
                    )
            _write_json(ref, {"commit": sha, "version": version, "repo": repo_id})
        if source_record is not None or output_dir is not None:
            receipt_path = root.parent / f"{version}.receipt.json"
            receipt_bytes = receipt_path.read_bytes()
            receipt = json.loads(receipt_bytes)
        if source_record is not None:
            provenance = build_source_record(
                repo_id, sha, "libero", memory_version=version
            )
    logger.info("memory: %s @ %s, root=%s", version, sha, root)
    if output_dir is not None:
        destination = Path(output_dir).resolve()
        if destination != root:
            if destination.exists():
                raise ValueError(
                    f"Output directory already exists: {destination}; choose a new directory"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
                copy = Path(staging) / "corpus"
                shutil.copytree(root, copy)
                if not _matches_files(copy, receipt["files"]):
                    raise ValueError("Memory contents changed during export")
                copy.replace(destination)
        root = destination
    if source_record is not None:
        source_record.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(source_record, provenance)
    return root


def main(argv: list[str] | None = None) -> int:
    """Download a LIBERO corpus without starting robot services."""
    parser = argparse.ArgumentParser(prog="python -m robots.libero.memory")
    commands = parser.add_subparsers(dest="command", required=True)
    sync = commands.add_parser(
        "sync", help="Download one model-specific LIBERO corpus."
    )
    sync.add_argument("--memory-version", choices=MEMORY_VERSIONS, default="auto")
    sync.add_argument("--model", default=None)
    sync.add_argument(
        "--planner",
        choices=["api", "codex", "claude_code", "flash"],
        default="api",
        help="Planner backend (default: api); codex uses CODEX_MODEL when --model is omitted.",
    )
    sync.add_argument("--revision", default="main", help="Hub commit, tag or branch.")
    sync.add_argument(
        "--source-record",
        type=Path,
        default=None,
        help="Write resolved source and file hashes outside the cache and output corpus.",
    )
    sync.add_argument(
        "--output-dir", type=Path, help="Copy into a new local corpus directory."
    )
    args = parser.parse_args(argv)
    version = select_version(
        args.memory_version, model=args.model, planner=args.planner
    )
    root = sync_version(
        version=version,
        revision=args.revision,
        cache_dir=get_memory_dir("libero") / ".versions",
        output_dir=args.output_dir,
        source_record=args.source_record,
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
