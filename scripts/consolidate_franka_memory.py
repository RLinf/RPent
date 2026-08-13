"""Run an agent to consolidate Franka episode memories into a reviewed draft."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rpent.dashboard.events import NullDashboardEventSink  # noqa: E402
from rpent.planner.base import build_planner  # noqa: E402
from rpent.tools.toolkit import Toolkit  # noqa: E402
from rpent.utils.config import get_memory_dir, get_repo_root  # noqa: E402
from rpent.utils.logging import init_output_dir  # noqa: E402

DEFAULT_SOURCE_PATTERNS = (
    "dual_franka_*.json",
    "full_log.jsonl",
    "agent_journal.jsonl",
    "latest_state.json",
    "localization_diagnostic/*.json",
    "memory_candidate.md",
    "lesson_*.md",
    "plan_*.md",
    "experience_*.md",
)


@dataclass(frozen=True)
class EpisodeSources:
    """Memory-relevant files discovered for one episode."""

    episode_dir: Path
    files: tuple[Path, ...]


class MemoryConsolidationToolkit(Toolkit):
    """Small file toolkit for offline memory consolidation."""

    def __init__(self, *, output_file: Path):
        self.output_file = output_file.resolve()
        super().__init__(dashboard_events=NullDashboardEventSink())
        self.add_tool("read_text_file", _READ_TEXT_FILE_SPEC, self.read_text_file)
        self.add_tool("list_dir", _LIST_DIR_SPEC, self.list_dir)
        self.add_tool("write_memory_draft", _WRITE_MEMORY_DRAFT_SPEC, self.write_memory_draft)
        self.add_tool("finish", _FINISH_SPEC, self.finish)

    def read_text_file(self, path: str, max_chars: int = 40000) -> dict[str, Any]:
        p = _resolve_path(path)
        if not p.exists():
            return {"error": f"file not found: {p}"}
        if p.is_dir():
            return {"error": f"is a directory: {p}"}
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = (
                text[:max_chars]
                + f"\n\n[TRUNCATED: file has {len(text)} chars]"
            )
        return {"path": str(p), "size": p.stat().st_size, "content": text}

    def list_dir(self, path: str) -> dict[str, Any]:
        p = _resolve_path(path)
        if not p.exists():
            return {"error": f"directory not found: {p}"}
        if not p.is_dir():
            return {"error": f"not a directory: {p}"}
        return {
            "path": str(p),
            "files": sorted(child.name for child in p.iterdir()),
        }

    def write_memory_draft(self, content: str) -> dict[str, Any]:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(content.strip() + "\n", encoding="utf-8")
        return {
            "path": str(self.output_file),
            "bytes_written": self.output_file.stat().st_size,
        }

    @staticmethod
    def finish(status: str, summary: str) -> dict[str, Any]:
        return {"_finish": True, "status": status, "summary": summary}


def _resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = get_repo_root() / p
    return p.resolve()


def _repo_relative(path: Path) -> str:
    root = get_repo_root()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())


def _expand_episode_args(args: argparse.Namespace) -> list[Path]:
    episodes: list[Path] = []
    for raw in args.episode or []:
        episodes.append(_resolve_path(raw))
    for pattern in args.episode_glob or []:
        for match in sorted(glob.glob(str(_resolve_path(pattern)))):
            episodes.append(Path(match).resolve())
    unique: list[Path] = []
    seen: set[Path] = set()
    for episode in episodes:
        if episode in seen:
            continue
        seen.add(episode)
        unique.append(episode)
    return unique


def collect_episode_sources(
    episodes: list[Path],
    *,
    patterns: tuple[str, ...] = DEFAULT_SOURCE_PATTERNS,
) -> list[EpisodeSources]:
    sources: list[EpisodeSources] = []
    for episode in episodes:
        if not episode.exists():
            raise FileNotFoundError(f"episode not found: {episode}")
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(episode.glob(pattern)))
        files = [path for path in files if path.is_file()]
        sources.append(EpisodeSources(episode_dir=episode, files=tuple(files)))
    return sources


def collect_memory_files(env_name: str, *, max_entries: int) -> list[Path]:
    memory_dir = get_memory_dir(env_name)
    if not memory_dir.exists():
        return []
    files = []
    index = memory_dir / "MEMORY.md"
    if index.exists():
        files.append(index)
    entry_patterns = ("feedback_*.md", "project_*.md", "session_*.md", "reference_*.md")
    entries: list[Path] = []
    for pattern in entry_patterns:
        entries.extend(sorted(memory_dir.glob(pattern)))
    files.extend(entries[:max_entries])
    return files


def write_source_manifest(
    *,
    output_dir: Path,
    env_name: str,
    memory_files: list[Path],
    episode_sources: list[EpisodeSources],
) -> Path:
    data = {
        "env_name": env_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "memory_files": [_repo_relative(path) for path in memory_files],
        "episodes": [
            {
                "episode_dir": _repo_relative(source.episode_dir),
                "files": [_repo_relative(path) for path in source.files],
            }
            for source in episode_sources
        ],
    }
    path = output_dir / "memory_consolidation_sources.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_system_prompt() -> str:
    return """You are a PhysicalAgent memory consolidation agent.

Your job is to update long-term Franka memory by reasoning over reviewed
memory and episode-local artifacts. You do not control the robot. You do not
write directly into resources/franka/memory. Produce one reviewed memory draft
with write_memory_draft, then call finish.

Read current MEMORY.md and relevant existing memory entries first. Then read
the provided structured episode artifacts. Treat dual_franka_*.json audit files,
full_log.jsonl, and localization_diagnostic/*.json as first-class evidence;
memory_candidate, lesson, plan, and experience files are useful summaries but
must not replace the structured logs. Prefer facts that repeat across runs or
are strongly supported by a single successful/failed episode. Remove raw log
noise, stale absolute coordinates, and command transcripts unless a numeric
threshold or command pattern is durable.

The draft must be evidence-backed and directly usable by a future robot agent.
Write it as a prompt-ready battle card rather than a loose retrospective. Use
these sections when applicable:

- Non-negotiable rules
- Episode evidence
- Perception and localization procedure
- VLA usage boundary and stop criteria
- Reset and joint-health recovery
- Object/category recipes
- Forbidden actions
- Open questions

For project-level memory, include a compact workflow and recovery criteria.
For feedback memory, focus on one concrete gotcha or strategy correction. If
the evidence is insufficient, write a short draft explaining why it should not
be promoted yet. Always cite episode directory names or artifact filenames for
the strongest claims.
"""


def build_user_prompt(
    *,
    manifest_path: Path,
    output_file: Path,
    kind: str,
    title: str,
    summary: str,
    source_episode: str,
) -> str:
    return f"""Consolidate Franka memory into one reviewed draft.

Target kind: {kind}
Target title: {title or "(agent may choose a title in the draft body)"}
Target summary: {summary or "(agent should infer a one-sentence summary)"}
Source episode label: {source_episode or "(multiple episodes)"}

Source manifest:
{_repo_relative(manifest_path)}

Output draft path:
{_repo_relative(output_file)}

Required workflow:
1. read_text_file the source manifest.
2. read current memory files listed in the manifest, especially MEMORY.md.
3. read the episode files listed in the manifest. Start from dual_franka_*.json
   audit files, then sample full_log.jsonl and localization_diagnostic/*.json
   for numeric evidence. Use lesson_*.md, plan_*.md, memory_candidate.md, and
   experience_*.md as secondary summaries.
4. Compare the sources and remove episode-specific noise. Do not preserve stale
   absolute coordinates as commands for future runs.
5. Convert repeated findings into prompt-ready rules, recipes, stop conditions,
   and forbidden actions.
6. write_memory_draft with the final reviewed Markdown body. Do not include
   YAML front matter; the promotion script adds it.
7. finish with status success if a draft was written, otherwise stuck.
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use an agent to consolidate Franka episode memories.",
    )
    parser.add_argument("--env-name", default="franka")
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--episode-glob", action="append", default=[])
    parser.add_argument("--kind", default="feedback")
    parser.add_argument("--title", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--source-episode", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-file", default="")
    parser.add_argument("--planner", choices=["codex", "api", "claude_code"], default="codex")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout-s", type=int, default=None)
    parser.add_argument("--max-memory-entries", type=int, default=12)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompt and manifest without invoking the agent.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    episodes = _expand_episode_args(args)
    if not episodes:
        raise ValueError("provide at least one --episode or --episode-glob")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        _resolve_path(args.output_dir)
        if args.output_dir
        else get_repo_root() / "logs" / "franka_memory_consolidation" / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    init_output_dir(output_dir)
    output_file = (
        _resolve_path(args.output_file)
        if args.output_file
        else output_dir / "reviewed_memory_draft.md"
    )

    memory_files = collect_memory_files(args.env_name, max_entries=args.max_memory_entries)
    episode_sources = collect_episode_sources(episodes)
    manifest_path = write_source_manifest(
        output_dir=output_dir,
        env_name=args.env_name,
        memory_files=memory_files,
        episode_sources=episode_sources,
    )
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        manifest_path=manifest_path,
        output_file=output_file,
        kind=args.kind,
        title=args.title,
        summary=args.summary,
        source_episode=args.source_episode,
    )
    (output_dir / "memory_consolidation_system.md").write_text(
        system_prompt,
        encoding="utf-8",
    )
    (output_dir / "memory_consolidation_user.md").write_text(
        user_prompt,
        encoding="utf-8",
    )

    print(f"manifest: {manifest_path}")
    print(f"draft: {output_file}")
    if args.dry_run:
        return 0

    toolkit = MemoryConsolidationToolkit(output_file=output_file)
    dashboard_events = NullDashboardEventSink()
    planner = build_planner(
        args.planner,
        output_dir=output_dir,
        recipe_tag="memory_consolidation",
        env_name=args.env_name,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        planner_timeout_s=args.timeout_s,
        dashboard_events=dashboard_events,
        no_images=True,
    )
    result = planner.solve(
        system_prompt=system_prompt,
        user_message=user_prompt,
        toolkit=toolkit,
        max_turns=args.max_turns,
    )
    if result.error:
        raise RuntimeError(result.error)
    if not output_file.exists():
        raise RuntimeError(f"agent did not write memory draft: {output_file}")
    print(output_file)
    return 0


_READ_TEXT_FILE_SPEC = {
    "name": "read_text_file",
    "description": "Read a UTF-8 source file for memory consolidation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_chars": {"type": "integer", "default": 40000},
        },
        "required": ["path"],
    },
}

_LIST_DIR_SPEC = {
    "name": "list_dir",
    "description": "List a directory when discovering nearby memory files.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}

_WRITE_MEMORY_DRAFT_SPEC = {
    "name": "write_memory_draft",
    "description": (
        "Write the final reviewed memory Markdown body to the configured "
        "draft path. Do not include YAML front matter."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
    },
}

_FINISH_SPEC = {
    "name": "finish",
    "description": "Finish the offline memory consolidation run.",
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": ["status", "summary"],
    },
}


if __name__ == "__main__":
    raise SystemExit(main())
