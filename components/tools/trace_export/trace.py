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

"""Trace v1: the recorded input of the trace exporter.

An episode directory holds three things:

- ``episode.json``: identity, model, verdict, and optional ``robo_methods``;
- ``turns.jsonl``: one record per planner turn, in order;
- ``images/``: the PNG files the turns reference.

A turn carries the model-visible input exactly as it was rendered for the
planner, so the exporter never renders a prompt of its own. Every object must
contain only the keys listed here: an unknown field needs a causal review
before it can reach an exported row, so it is rejected instead of ignored.
"""

from __future__ import annotations

import hashlib
import io
import json
import keyword
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

TRACE_SCHEMA = "rpent.code_policy_trace.v1"
EPISODE_FILE = "episode.json"
TURNS_FILE = "turns.jsonl"
IMAGE_DIR = "images"

EPISODE_KEYS = ("schema", "episode_id", "robot", "model", "verdict")
OPTIONAL_EPISODE_KEYS = ("robo_methods",)
VERDICT_KEYS = ("accepted", "truth_source", "stop_reason")
TURN_KEYS = (
    "turn",
    "system",
    "user",
    "images",
    "code",
    "executed_code",
    "accepted",
    "interrupted",
    "timed_out",
    "namespace_reset",
)
IMAGE_KEYS = ("path", "png_sha256", "rgb_sha256", "width", "height")


class TraceError(ValueError):
    """Raised when a recorded trace cannot be exported safely."""


def require(condition: object, message: str) -> None:
    """Raise ``TraceError`` with ``message`` unless ``condition`` holds."""
    if not condition:
        raise TraceError(message)


def sha256_hex(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _reject_constant(name: str) -> None:
    raise ValueError(f"non-finite number {name} is not allowed")


def _loads(text: str, what: str) -> Any:
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except ValueError as exc:
        raise TraceError(f"{what} is not valid JSON: {exc}") from exc


def _exact_keys(record: Any, keys: tuple[str, ...], what: str) -> dict[str, Any]:
    require(isinstance(record, dict), f"{what} must be an object")
    require(set(record) == set(keys), f"{what} keys must be exactly {sorted(keys)}")
    return record


def _episode_keys(record: Any) -> dict[str, Any]:
    require(isinstance(record, dict), "episode must be an object")
    missing = sorted(set(EPISODE_KEYS) - set(record))
    extra = sorted(set(record) - set(EPISODE_KEYS) - set(OPTIONAL_EPISODE_KEYS))
    require(not missing, f"episode missing required keys: {missing}")
    require(not extra, f"episode contains unknown keys: {extra}")
    return record


def validate_robo_methods(
    methods: Any,
    what: str = "robo_methods",
    *,
    require_list: bool = False,
) -> tuple[str, ...]:
    """Return validated public ``robo`` method names.

    Args:
        methods: Collection or JSON list of method names to validate.
        what: Label used in validation error messages.
        require_list: When true, require ``methods`` to be a JSON list.

    Returns:
        Validated public method names in input order.
    """
    if require_list:
        require(
            isinstance(methods, list) and bool(methods),
            f"{what} must be a non-empty list",
        )
    else:
        require(
            isinstance(methods, Collection)
            and not isinstance(methods, (str, bytes))
            and bool(methods),
            f"{what} must be non-empty",
        )
    items = list(methods)
    for name in items:
        require(
            isinstance(name, str)
            and name.isidentifier()
            and not name.startswith("_")
            and not keyword.iskeyword(name),
            f"{what} contains invalid public method name: {name!r}",
        )
    require(len(set(items)) == len(items), f"{what} contains duplicate method names")
    return tuple(items)


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _check_images(refs: Any, turn: int) -> list[dict[str, Any]]:
    require(isinstance(refs, list), f"turn {turn}: images must be a list")
    for ref in refs:
        _exact_keys(ref, IMAGE_KEYS, f"turn {turn}: image")
        require(_nonempty_str(ref["path"]), f"turn {turn}: image path is empty")
        require(
            all(type(ref[k]) is int and ref[k] > 0 for k in ("width", "height")),
            f"turn {turn}: image size must be positive integers",
        )
    return refs


def _check_user(parts: Any, image_count: int, turn: int) -> None:
    require(isinstance(parts, list) and parts, f"turn {turn}: user input is empty")
    used = []
    for part in parts:
        require(isinstance(part, dict), f"turn {turn}: user part must be an object")
        if part.get("type") == "text":
            require(
                set(part) == {"type", "text"} and isinstance(part["text"], str),
                f"turn {turn}: malformed text part",
            )
        else:
            index = part.get("image")
            require(
                part.get("type") == "image"
                and set(part) == {"type", "image"}
                and type(index) is int
                and 0 <= index < image_count,
                f"turn {turn}: malformed image part",
            )
            used.append(index)
    require(
        sorted(used) == list(range(image_count)),
        f"turn {turn}: every listed image must appear exactly once",
    )


def _check_turn(record: Any, index: int) -> dict[str, Any]:
    turn = _exact_keys(record, TURN_KEYS, f"turn {index}")
    require(type(turn["turn"]) is int and turn["turn"] == index, f"turn gap at {index}")
    require(_nonempty_str(turn["system"]), f"turn {index}: system prompt is empty")
    require(_nonempty_str(turn["code"]), f"turn {index}: code is empty")
    require(
        turn["code"] == turn["executed_code"],
        f"turn {index}: executed source differs from accepted source",
    )
    require(turn["accepted"] is True, f"turn {index}: cell was not accepted")
    for key in ("interrupted", "timed_out", "namespace_reset"):
        require(
            turn[key] is False,
            f"turn {index}: {key}; the namespace prefix is unknown, isolate the episode",
        )
    refs = _check_images(turn["images"], index)
    _check_user(turn["user"], len(refs), index)
    return turn


def load_episode(root: Path | str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read and structurally validate one episode directory.

    Args:
        root: Episode directory holding ``episode.json`` and ``turns.jsonl``.

    Returns:
        The episode record and its turn records, in turn order.
    """
    root = Path(root)
    require(root.is_dir(), f"trace directory does not exist: {root}")
    try:
        episode_text = (root / EPISODE_FILE).read_text(encoding="utf-8")
        turns_text = (root / TURNS_FILE).read_text(encoding="utf-8")
    except OSError as exc:
        raise TraceError(f"cannot read trace files: {exc}") from exc
    episode = _episode_keys(_loads(episode_text, EPISODE_FILE))
    require(
        episode["schema"] == TRACE_SCHEMA, f"unsupported schema {episode['schema']!r}"
    )
    for key in ("episode_id", "robot", "model"):
        require(_nonempty_str(episode[key]), f"episode {key} is empty")
    _exact_keys(episode["verdict"], VERDICT_KEYS, "verdict")
    if "robo_methods" in episode:
        episode["robo_methods"] = list(
            validate_robo_methods(
                episode["robo_methods"], "episode robo_methods", require_list=True
            )
        )
    lines = [line for line in turns_text.splitlines() if line.strip()]
    require(lines, "trace has no turns")
    turns = [
        _check_turn(_loads(line, f"turn {i}"), i) for i, line in enumerate(lines, 1)
    ]
    return episode, turns


def discover_episodes(sources: Sequence[Path | str] | Path | str) -> list[Path]:
    """Resolve direct episode directories and search parent directories.

    Args:
        sources: Episode directory, parent directory, or sequence of them.

    Returns:
        Unique resolved episode candidate directories in discovery order.
    """
    raw = [sources] if isinstance(sources, (Path, str)) else list(sources)
    require(raw, "no source episodes")
    discovered: list[Path] = []
    for source in raw:
        root = Path(source).expanduser().resolve()
        require(root.is_dir(), f"source directory does not exist: {root}")
        if (root / EPISODE_FILE).is_file() or (root / TURNS_FILE).is_file():
            discovered.append(root)
            continue
        found = sorted(
            {
                path.parent.resolve()
                for filename in (EPISODE_FILE, TURNS_FILE)
                for path in root.rglob(filename)
                if path.is_file()
            }
        )
        require(found, f"no source episodes found under {root}")
        discovered.extend(found)
    return list(dict.fromkeys(discovered))


def read_image(root: Path | str, ref: dict[str, Any]) -> bytes:
    """Return the PNG bytes of ``ref`` after checking path, format and checksums.

    Args:
        root: Episode directory that holds ``images/``.
        ref: A validated image record from a turn.

    Returns:
        The unmodified PNG file contents.
    """
    images = (Path(root) / IMAGE_DIR).resolve()
    path = (images / ref["path"]).resolve()
    require(path.is_relative_to(images), "image path escaped the trace")
    require(path.is_file(), f"missing image {ref['path']}")
    data = path.read_bytes()
    require(sha256_hex(data) == ref["png_sha256"], "PNG checksum mismatch")
    try:
        with Image.open(io.BytesIO(data)) as image:
            require(
                image.format == "PNG"
                and image.mode == "RGB"
                and image.size == (ref["width"], ref["height"]),
                "PNG format or size mismatch",
            )
            require(
                sha256_hex(image.tobytes()) == ref["rgb_sha256"],
                "RGB checksum mismatch",
            )
    except (OSError, SyntaxError) as exc:
        raise TraceError(f"image {ref['path']} cannot be decoded: {exc}") from exc
    return data
