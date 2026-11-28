---
name: code
description: "RPent code, tests, tools, and CI (rpent/, robots/, components/, tests/, .github/, pyproject.toml, setup.py, any .py file): read before writing or changing them, then check and verify the change. Covers RPent's design patterns (extension points, containment, daemon ownership, lazy imports, no unnecessary protection code), code conventions, offline CPU unit-test layout, license headers, and verification evidence. Use before adding or changing a planner, session, toolkit, robot adapter, evaluation hook, CLI/Dashboard path, or test, and to check or verify such a change."
paths:
  - rpent/**
  - robots/**
  - components/**
  - tests/**
  - .github/**
  - pyproject.toml
  - setup.py
  - "**/*.py"
---

# Code

RPent's design patterns live in [patterns.md](patterns.md), and the rules CI and reviewers enforce live in [AGENTS.md](../../../AGENTS.md). This skill says what to read before changing code, checks the change afterwards, and guides test selection and verification. Robot integration contracts, packaging extras, and GPU e2e wiring belong to the [robots](../robots/SKILL.md) skill, and docs and README highlights to the [docs](../docs/SKILL.md) skill.

## Write

1. Find the closest current sibling on `main` and start from it (`git log <base> -- <dir>` shows which sibling is recent, so you follow current contracts rather than legacy code).
2. Plug into the extension point the component kind already has (AGENTS.md § Extending RPent) instead of adding robot- or planner-name branches to shared code.
3. Read [patterns.md](patterns.md): its design philosophy first, then the Design, Code, and Tests patterns that apply to the change.
4. Mirror the production module layout under `tests/unit_tests/` (`rpent/`, `robots/<robot>/`, `components/`, `installs/`, `tooling/`, `skills/`), test through the public contract offline on CPU, and mock only external edges ([Testing and CI](https://rpent.readthedocs.io/en/latest/rst_source/development/testing.html)).
5. Keep optional simulator, VLA, and hardware imports inside the function or branch that uses them so `import rpent`, `enumerate_robots()`, and `rpent --help` work in a core-only environment.
6. Before you finish, go through [mistakes.md](mistakes.md), run the check below, and run the relevant unit tests.

## Check

```bash
python3 .agents/skills/code/check.py              # files changed against main
python3 .agents/skills/code/check.py FILE ...
```

Findings use the shared format of [README.md](../README.md#findings):

| Rule | Severity | Reports |
|---|---|---|
| `code/pre-commit` | major | `pre-commit run --files` (or `ruff` fallback) fails: Ruff lint (`--preview`), format, and `CPY001` headers. Ruff runs with `--fix`, so re-run after it rewrites a file. |
| `code/init` | major | a directory with Python files under `rpent/`, `robots/`, or `components/` without `__init__.py` |
| `code/license` | major or minor | a new Python file without `# Copyright <year> The RPent Authors.` (major), or with a year other than the current one (minor) |
| `code/test-layout` | minor | a new `test_*.py` file placed directly under `tests/unit_tests/` instead of a component subdirectory |
| `code/shell-syntax` | critical | a changed shell script that fails `bash -n` |

Then judge what a script cannot decide, comparing every changed function against its version on `main` ([mistakes.md](mistakes.md) lists the concrete forms):

- [ ] Shared orchestration code (`rpent/cli/`, `rpent/dashboard/`, `rpent/session/`, `rpent/tools/`) gains no branch that names a specific robot or planner, and `rpent/planner/` gains no robot-specific branch (patterns.md § Generalize; don't special-case, § Contain component code).
- [ ] Each resource has one owner: owned daemons tear down on partial startup failure via `rpent/robots/runtime.py`, and borrowed external endpoints are never stopped (§ Small abstractions with one owner).
- [ ] No protection code for states callers cannot produce, validation happens once at the owning boundary, and no product code is weakened with `getattr`/`hasattr` for incomplete mocks (§ No unnecessary protection code).
- [ ] Behavior is preserved across affected planners (`api`, `claude_code`, `codex`, `gemini_cli`, `flash`), CLI and Dashboard entry points, and robots, or the difference is named, documented, and tested (§ Preserve behavior).
- [ ] No new function, field, CLI flag, or prompt variable without a caller or consumer, and no drive-by edits outside the task scope.
- [ ] Code conventions hold: `rpent.utils.logging.get_logger()` instead of `print`, type hints and Google-style docstrings on public APIs, short contract-focused comments, and error messages stating expected vs. found (§ Code).
- [ ] Tests fail without the change, run offline on CPU, use real objects or local loopback transports, and never weaken assertions (§ Tests).

## Verify

Before opening a pull request, run the repository checks in an isolated environment ([CONTRIBUTING.md](../../../CONTRIBUTING.md), [Testing and CI](https://rpent.readthedocs.io/en/latest/rst_source/development/testing.html)):

```bash
pre-commit run --all-files
pytest tests/unit_tests -v
```

Inspect any files rewritten by hooks and re-run affected tests. Note that [unittest.yml](../../../.github/workflows/unittest.yml) runs `infra-unit_cpu` across Python 3.10–3.12; one local environment does not cover the whole matrix.

When reporting validation evidence:

- Record the revision or working diff, exact commands, environment, results, artifact paths, and skipped checks with reasons.
- Distinguish a failed check from unavailable prerequisites and from an untested path.
- Keep bounded integration checks (action execution, recorded `finish`, saved artifacts, owned-daemon cleanup) distinct from native task success (`terminated`). Never infer task success from planner prose or a clean exit code.
