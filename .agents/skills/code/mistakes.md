# Mistakes agents make in RPent code

Coding agents repeat a small set of mistakes in RPent, and reviewers send them back each time. Check this list before reporting a change as done. Each entry names the mistake, the fix, and the rule in [patterns.md](patterns.md) or [AGENTS.md](../../../AGENTS.md).

## Design

| Mistake | Fix | Rule |
|---|---|---|
| A branch in `rpent/cli/`, `rpent/dashboard/`, `rpent/session/`, or `rpent/tools/` that checks `robot == "..."` or `planner == "..."` | Put the behavior behind a `RobotSpec` hook/flag, `Toolkit` method, or `Planner` method | patterns.md § Generalize; don't special-case |
| Copying an existing Env/VLA client, server, or toolkit with minor edits for a new robot | Reuse or parameterize `BaseEnvClient`/`BaseEnvFacade`/`BaseVLAClient`/`BaseVLAFacade` or `Toolkit` | § Generalize; don't special-case |
| Spawning a daemon before registering it in `daemons`, or tearing down an externally provided endpoint | Register owned daemons via `try_spawn_server`/`try_wait_server` (`rpent/robots/runtime.py`) and stop only owned daemons | § Small abstractions with one owner |
| Treating planner `finish` or a clean CLI exit as benchmark task success | Report native environment `terminated`/verifier outcome separately from bounded `finish` completion | AGENTS.md § Runtime ownership and success semantics |
| Putting robot-specific prompts, coordinate transforms, or simulator quirks in shared `rpent/` modules | Keep robot policy in `robots/<robot>/` and expose a clean contract | § Contain component code |
| Fixing a shared RPC, session, or toolkit bug only in one robot adapter | Fix the owning layer in `rpent/` so all robots and planners benefit | § Fix at the owning layer |
| A refactor that changes behavior on an untouched planner (`api`, `codex`, `claude_code`, `gemini_cli`, `flash`) or Dashboard path | Trace every entry point and planner against `main`; name, document, and test intended differences | § Preserve behavior |

## Defensive and dead code

| Mistake | Fix | Rule |
|---|---|---|
| `hasattr`, `getattr(x, "y", None)`, `isinstance` ladders, or `None` checks for guaranteed internal contracts | Trust internal contracts; validate external inputs once at the owning boundary | § No unnecessary protection code |
| Replacing `obj.attr` with `getattr`/`hasattr` or adding fallback guards in product code so incomplete test mocks pass | Keep direct member access; construct the real object or satisfy its contract in the test | § No unnecessary protection code |
| `try: ... except Exception:` that logs and continues in a broken state, or bare `except: pass` | Let the error surface, or raise a contextual error with `raise ... from exc` | § No unnecessary protection code |
| Retries, `time.sleep` waits, or compatibility shims for unproven failures | Remove them, or add a test demonstrating the failure | § No unnecessary protection code |
| Unused helpers, CLI flags, prompt variables, or commented-out code | Delete them | patterns.md § Code |
| Returning base64 images or huge arrays in tool-result text | Save artifacts to disk and return concise paths/metadata | § Prompts and tool contracts |

## Conventions

| Mistake | Fix | Rule |
|---|---|---|
| Top-level import of `torch`, `mujoco`, `robosuite`, `libero`, `openpi`, `rlinf`, `sam3`, `isaacsim`, or hardware SDKs | Import optional dependencies inside the function or branch that uses them | AGENTS.md § Optional dependencies and lazy imports |
| `print(...)` in runtime code | Use `rpent.utils.logging.get_logger(...)` | AGENTS.md § Logging and errors |
| Hardcoded machine paths, GPU indices, ports, or API keys | Read from CLI args, config, or environment variables | patterns.md § Code |
| Comments that narrate code, record PR review IDs (`F1`, `F2`), or tell fix history | State the contract, invariant, or non-obvious reason in one sentence | AGENTS.md § Comments and docstrings |
| Public functions or classes without type hints or Google-style docstrings | Add them | AGENTS.md § Ruff and style |
| New Python file without `# Copyright <year> The RPent Authors.`, or Python directory without `__init__.py` | Add them; `code/check.py` reports both | AGENTS.md § Package layout and headers |

## Accuracy

| Mistake | Fix |
|---|---|
| Guessing a flag, tool name, prompt placeholder, or RPC route from memory | Read the owning code and grep all callers before using it |
| Renaming a public symbol, CLI flag, or tool schema without updating callers, prompts, tests, and EN/ZH docs | Search the whole repository and update all references in the same change |

## Tests

| Mistake | Fix | Rule |
|---|---|---|
| Dumping a new `test_*.py` directly under `tests/unit_tests/` | Place it under `tests/unit_tests/{rpent,robots/<robot>,components,installs,tooling,skills}/` | [Testing and CI](https://rpent.readthedocs.io/en/latest/rst_source/development/testing.html) |
| Unit tests that require GPUs, network access, model checkpoints, or simulator extras | Keep `tests/unit_tests/` offline on CPU with local loopback or edge fakes; put GPU checks under `tests/e2e_tests/` | AGENTS.md § Tests |
| Tests that mostly assert mocks were called or mirror internal implementation details | Test observable behavior through the public contract | patterns.md § Tests |
| Weakening an assertion or skipping a failing test so CI goes green | Fix the underlying bug, or explain to the user why the contract changed | CONTRIBUTING.md § Project red lines |

## Process

| Mistake | Fix |
|---|---|
| Starting before checking the closest current sibling on `main` and open PRs/issues | Inspect `main` and open PRs first |
| Reporting done without running checks, or claiming skipped tests passed | Run `python3 .agents/skills/route.py check` and `pytest tests/unit_tests -v`, reporting exact commands and results |
| Leaving `TODO(agent)` or placeholder code | Finish it or stop and ask; document known limits in the PR description |
| Pushing, opening/editing a PR, or posting a review without user approval | Wait for explicit user approval after previewing the content |
