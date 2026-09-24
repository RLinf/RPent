# Shared initial context assembly

## Goal

Give RPent one reusable context assembly function for rendered instructions,
the current query, selected memory, skills, and initial text/image observations.
CLI, Dashboard, and external MCP episodes must use it without changing existing
planner behavior. This implements the context-first stage requested on
2026-09-24; native runtime and subagent execution remain separate work.

## Contracts

- `ContextDocument(title, text, source=None)` represents an already resolved
  memory or skill document. The source is provenance, not a path to read during
  assembly. Documents contain text, not permissions or executable handlers.
- `load_skill(path)` reads one UTF-8 Markdown file and records its resolved path.
  It preserves the current title convention: the parent directory for SKILL.md,
  otherwise the filename stem. Read failures propagate before an episode starts.
- `assemble_context(prompt=..., query=..., memory=(), skills=(),
  initial_context=())` performs no I/O and returns a `ContextBundle` retaining
  the separate inputs in caller order. Input collections are snapshotted.
- `ContextBundle.system_prompt` renders the prompt followed by the existing
  `## Skill: <title>` sections. Whitespace and skill content remain unchanged.
- `ContextBundle.user_message` renders the query followed by `## Memory:
  <title>` sections. Initial observations follow that text as separate content
  parts. Without initial observations, the result remains a string.
- BinaryContent is passed through without converting image bytes to text.
  Existing planner and interactive multimodal limitations remain in force.
- `EmbodiedAgent.run` accepts optional resolved memory documents. It continues
  loading explicitly supplied skill files afresh for each episode.
- Robot prompt factories and memory tools retain their current responsibilities.
  No automatic memory ingestion, skill discovery, retrieval, or context pruning
  is introduced. Existing memory access rules are not bypassed by file loading.
- `Planner.solve` stays compatible. The new bundle is projected to its existing
  `system_prompt` and `user_message` arguments at the three entry points.
- CLI operator-edited input and exploration handoffs are assembled only after
  their current query has been selected; the original task must not overwrite it.

## Layout

Use `rpent/context.py` for the small public API. Reuse PromptBundle rendering,
existing SDK adapters, memory management, and artifact lifecycles. This module
does not import robot packages or planner implementations.

## Validation

Offline tests cover role separation, exact existing rendering, fresh skill
loads, Unicode paths, missing files, collection isolation, image ordering,
explicit memory through EmbodiedAgent, and CLI/Dashboard continuation queries.
Run the affected tests, repository unit suite, pre-commit, and both Sphinx
language builds. No model service or robot runtime is needed for these checks.
