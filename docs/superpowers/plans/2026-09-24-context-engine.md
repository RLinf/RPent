# Context Engine Implementation Plan

> **For agentic workers:** Implement in this session using the existing tests
> and fixtures. Steps use checkbox syntax to track the completed work.

**Goal:** Unify initial context assembly without changing existing planner inputs.

**Architecture:** A pure assembler retains prompt, query, memory, skills, and
observations separately. A skill file loader resolves filesystem inputs before
assembly. Entry points project the bundle onto the current Planner protocol.

**Tech Stack:** Python 3.10–3.12, dataclasses, PydanticAI BinaryContent, pytest.

**Spec:** ../specs/2026-09-24-context-engine-design.md

## Global Constraints

- Preserve the existing Planner.solve signature and SDK message-role behavior.
- Preserve prompt whitespace, skill headings, ordering, and initial image blocks.
- Keep memory retrieval, access checks, writes, and merging in their current owners.
- No new dependency, runtime, CLI option, or implicit skill discovery.
- Keep English and Chinese documentation aligned.

## Review Focus

- Caller-provided lists must not leak changes into a previously assembled bundle.
- A missing or unreadable skill must fail before MCP startup.
- Explicit memory must not enter system instructions.
- CLI operator queries and continuation queries must retain their current meaning.
- Multimodal content and cache-prefix ordering must survive the adapter unchanged.

## Task 1: Public assembler and skill source loader

Files: `rpent/context.py`, `tests/unit_tests/rpent/test_context_contracts.py`.

Interfaces: `ContextDocument(title, text, source=None)`, `load_skill(path)`,
`assemble_context(*, prompt, query, memory=(), skills=(), initial_context=())`.
The result exposes the separate inputs and `system_prompt` / `user_message`.

- [x] Write tests for plain rendering, memory/skill separation, fresh file loads,
  failures, collection isolation, and initial image preservation.
- [x] Run the new tests and confirm the missing context API is the failure.
- [x] Implement the two immutable data containers, loader, and assembler.
- [x] Run `pytest tests/unit_tests/rpent/test_context_contracts.py -q`.

Expected public use:

```python
context = assemble_context(
    prompt="Use the registered tools.",
    query="Place the block.",
    memory=[ContextDocument("grasp", "Approach from above.", "memory/grasp.md")],
    skills=[load_skill("benchmark/SKILL.md")],
)
assert "Approach from above." not in context.system_prompt
assert "Approach from above." in context.user_message
```

## Task 2: Integrate the current entry points

Files: `rpent/embodied_agent.py`, `rpent/cli/main.py`, `rpent/cli/dashboard.py`,
and the existing EmbodiedAgent, CLI, and Dashboard contract test files.

- [x] Extend the real local MCP test with explicit memory and image observations;
  add coverage for failing skill loads before transport startup.
- [x] Extend continuation coverage to inspect the actual query and instructions.
- [x] Replace inline skill concatenation and assemble each entry point's final
  query immediately before calling the planner.
- [x] Add the optional `memory` argument to `EmbodiedAgent.run`.
- [x] Run the context, planner, CLI, Dashboard, and external benchmark tests.

The planner call keeps its contract:

```python
result = planner.solve(
    system_prompt=context.system_prompt,
    user_message=context.user_message,
    toolkit=toolkit,
    max_turns=max_turns,
)
```

## Task 3: Document and validate

Files: paired `usage/configure_planner.rst` and `usage/embodied_agent.rst` pages.

- [x] Document public assembly, explicit memory, provenance, and current limits.
- [x] Run `pre-commit run --all-files` and the complete unit suite.
- [x] Run both documented `sphinx-build -W --keep-going` commands.
- [x] Review the entire diff and record checks and any unavailable prerequisites.

## Execution record

- Started from current origin/main (`ec8cd12`) on `codex/context-engine`.
- The user authorized implementation after reviewing the architecture in chat.
- Validation uses an isolated Python 3.11 environment under WSL Ubuntu because
  the repository uses Linux interfaces such as fcntl.
- Context tests: 13 expected failures before implementation, then 13 passes.
- The new EmbodiedAgent memory case failed on the missing keyword before
  integration; the focused context/MCP/CLI/Dashboard set then passed 54 tests.
- `pre-commit run --all-files` passed after formatting and the docstring correction.
- Both Sphinx builds passed; the English title underline warning was corrected
  and its build rerun successfully.
- Independent read-only review reported no actionable findings.
- The first full test run on the Windows checkout had 638 passes, 6 failures,
  and 3 skips. The failures were two CRLF shell-script cases and four cases
  requiring CPU Torch. The CI Flywheel dependencies were then installed with
  `uv pip install -e '.[flywheel]' --torch-backend cpu`. The two shell cases
  passed from an LF checkout exported from HEAD with every changed file overlaid
  byte-for-byte.
- Final full-suite command in `/tmp/rpent-context-check-s0vd2rf4`:
  `/tmp/rpent-context-env/bin/python -m pytest tests/unit_tests -q -rs --junitxml=unit-results.xml`.
  Result: 646 passed, 1 skipped in 49.77 seconds. The skipped RoboTwin seed-language
  contract requires the optional `rlinf.envs.robotwin.robotwin_env` module.
- JUnit evidence: `/tmp/rpent-context-check-s0vd2rf4/unit-results.xml` in WSL Ubuntu.
  The copy contains the final tested Python changes; subsequent edits only record
  these results in this plan. No live model, simulator, or hardware benchmark was run.
