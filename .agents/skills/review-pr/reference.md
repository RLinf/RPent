# PR review checklist

Checklist for step 5 of [SKILL.md](SKILL.md), ordered by finding category. Design rules live in [code/patterns.md](../code/patterns.md); area checks live in `code`, `robots`, and `docs`. Cross-reference `main` via `git show "$BASE:<path>"` and `git grep "<pattern>" "$BASE"`, citing `path:line` on `$PR_HEAD` (plus `main:path:line` for cross-checks).

## correctness

- [ ] **Logic**: off-by-one, inverted conditions, wrong defaults, swapped args, mutated shared state, missing locks/drains.
- [ ] **Edge cases**: empty/`None` inputs, missing optional config, first/last step, timeout/step limit, interruption/cancellation (`ToolCancelled`), real-robot vs. simulator reset semantics.
- [ ] **Concurrency and RPC**: cancellation of queued/active tool calls (`Toolkit._scheduler`), RPC timeouts/retries, thread safety, event-loop blocking, Dashboard vs. CLI event delivery.
- [ ] **Resources and lifecycle**: orphaned subprocess daemons on startup failure, stopping borrowed external endpoints, leaked file handles/sockets/temp dirs, non-idempotent `stop()`/`close()`.
- [ ] **Success semantics**: planner `finish` never conflated with native environment `terminated` or verifier reward.
- [ ] **Error handling**: silent `except`, swallowed original exceptions, missing boundary validation, or over-validating internal state ([code/patterns.md](../code/patterns.md#no-unnecessary-protection-code)).
- [ ] **Refactor regressions**: compare every changed function against `main` across planners (`api`, `claude_code`, `codex`, `gemini_cli`, `flash`), CLI, Dashboard, and robots.
- [ ] **Caller and consumer sweep**: when a signature, `RobotSpec` hook, RPC route, tool schema, CLI flag, or prompt variable changes, grep all callers, subclasses, planners, tests, and docs at `$PR_HEAD`.
- [ ] **Claims vs. diff**: verify PR description claims (no behavior change, test coverage, metrics, hardware runs) against the diff and test results.

## design

A mismatch with RPent conventions, bloated workaround, or dead boilerplate is a design defect, not a style nit.

- [ ] **Sibling**: matches the newest idiomatic sibling on `main` (`git log "$BASE" -- <dir>`), not legacy boilerplate.
- [ ] **Patterns**: follows [code/patterns.md § Design](../code/patterns.md#design) (no robot/planner-name branches in shared code, contained robot code under `robots/<robot>/`, single-owner abstractions, fix at owning layer, direct common path, preserved behavior).
- [ ] **Extension points**: plugs into `RobotSpec`, `Toolkit`, `Planner`, `BaseEnvClient`/`BaseVLAClient`/`BaseWAMClient`, `runtime.py` helpers, and `components/tools/rgbd/`.
- [ ] **Lazy imports and packaging**: optional simulator/VLA/hardware imports stay inside use sites; `import rpent`, `enumerate_robots()`, and `rpent --help` work with base dependencies only.
- [ ] **Test-driven guards**: no product code weakened for incomplete test mocks/stubs (`getattr`/`hasattr` or fallback branches); fix the test instead.
- [ ] **Prompts and tools**: rendered variables and tool names match `Toolkit.list_tools()` across planners; large payloads stay out of tool-result text; instructions stay concise.
- [ ] **Dead code and scope creep**: every new function, field, flag, and prompt variable has a consumer; no unrelated reformatting or extra edits.

## reuse

- [ ] **Duplication**: reuses existing RPent and RLinf helpers (`rpent/components/`, `rpent/robots/runtime.py`, `rpent/utils/rpc/`, `rpent/utils/daemon.py`, `components/tools/rgbd/`, `tests/e2e_tests/common.py`, and shared docs guides/partials).
- [ ] **Citation**: cite the existing helper as `main:<path>:<line>` when flagging reuse; name the closest comparison when nothing applies.

## tests, docs, and docs-style

Area scripts and checklists run in step 6 ([code § Check](../code/SKILL.md#check), [robots § Check](../robots/SKILL.md#check), [docs § Check](../docs/SKILL.md#check)). Review these PR-wide checks alongside them:

- [ ] **Coverage and completeness**: user-facing changes include offline CPU tests that fail without the change; new robot integrations ship `component.toml` manifests, offline unit tests, declarative GPU e2e suites, paired EN/ZH docs, and README highlights.
- [ ] **Docs readability (fresh agent)**: run a fresh-agent pass (no `rpent/`, `robots/`, or `components/` context) following [docs § Check](../docs/SKILL.md#check) and `docs/rules/` for plain human prose, native Chinese, concise main paths, no internal code citations or developer internals leaking into user guides, no non-runnable placeholder examples (`example.com`), and no negative-contrast restatement.
- [ ] **Sibling alignment**: compare new simulator/robot guides against the closest current sibling on `main` and check `conf.py` redirects for moved pages.

## style

Flag only real problems; Ruff lint/format, `__init__.py`, and license headers come from `code/check.py`, and code conventions from [code/patterns.md § Code](../code/patterns.md#code).

- [ ] **Commits and PR metadata**: Conventional Commits on commits; valid PR title, filled Description/Testing/Manual verification/Checklist (`lint_pr.py lint --pr <N>`).
