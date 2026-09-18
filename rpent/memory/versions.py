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

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from rpent.utils.logging import get_logger

logger = get_logger("memory")
GPT5 = "GPT_5.5_xhigh"
ASTRA = "GPT_6_astra_low"
MEMORY_VERSIONS = ("auto", GPT5, ASTRA)
DEFAULT_REPO = "RLinf/RPent-memory"


def resolve_model(planner: str, model: str | None) -> str | None:
    """Resolve the model using the planner's explicit/environment precedence."""
    return model or (os.environ.get("CODEX_MODEL") if planner == "codex" else None)


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
    """Locate replay assets in this corpus, including the legacy directory name."""
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
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _verified(root: Path) -> bool:
    try:
        files = json.loads((root.parent / f"{root.name}.receipt.json").read_text())
        return bool(files) and all(
            _safe_relative(name) and _hash(root / name) == digest
            for name, digest in files.items()
        )
    except (OSError, ValueError, AttributeError):
        return False


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sync_version(
    *,
    version: str,
    cache_dir: Path,
    repo_id: str = DEFAULT_REPO,
    revision: str = "main",
    output_dir: Path | None = None,
) -> Path:
    """Download one complete corpus; reuse only its verified cache on outage.

    New-layout hashes come from ``libero/manifest.json``. Legacy layouts are
    accepted only for GPT-5.5. A receipt is written after every selected file
    has downloaded, then checked on every reuse. Ref pointers are scoped to
    the repository, requested revision and version.
    """
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    if version not in (GPT5, ASTRA):
        raise ValueError("sync_version requires a resolved memory version")
    repo_id = os.environ.get("RPENT_MEMORY_HF_REPO", repo_id)
    key = hashlib.sha256(repo_id.encode()).hexdigest()[:20]
    base = Path(cache_dir).resolve() / key
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
            info = HfApi().repo_info(repo_id, repo_type="dataset", revision=revision)
        except Exception as exc:
            try:
                sha = json.loads(ref.read_text())["commit"]
                root = base / "snapshots" / sha / version
                if not _verified(root):
                    raise ValueError("incomplete or modified cache")
            except (OSError, ValueError, KeyError) as cache_exc:
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
            root = base / "snapshots" / sha / version
            if not _verified(root):
                names = [entry.rfilename for entry in info.siblings]
                prefix = f"libero/{version}/"
                selected = [name for name in names if name.startswith(prefix)]
                expected = None
                if selected:
                    manifest_path = hf_hub_download(
                        repo_id,
                        "libero/manifest.json",
                        repo_type="dataset",
                        revision=sha,
                    )
                    manifest = json.loads(Path(manifest_path).read_text())
                    expected = manifest["versions"][version]["files"]
                elif version == GPT5 and "libero/MEMORY.md" in names:
                    prefix = "libero/"
                    selected = [
                        name
                        for name in names
                        if name == "libero/MEMORY.md"
                        or any(
                            name.startswith(f"libero/{folder}/")
                            for folder in (
                                "global",
                                "suite",
                                "task_only",
                                "task_card",
                                "flash",
                            )
                        )
                    ]
                else:
                    raise ValueError(f"{repo_id}@{sha} has no {version} corpus")
                relative = [name.removeprefix(prefix) for name in selected]
                if any(not _safe_relative(name) for name in relative):
                    raise ValueError("Unsafe memory file path")
                if expected is not None and set(relative) != set(expected):
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
                    if expected is not None and hashes != expected:
                        raise ValueError("Memory file checksum mismatch")
                    if root.exists():
                        shutil.rmtree(root)
                    corpus.replace(root)
                    _write_json(root.parent / f"{version}.receipt.json", hashes)
            _write_json(ref, {"commit": sha, "version": version, "repo": repo_id})
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
                copy.replace(destination)
        root = destination
    return root
