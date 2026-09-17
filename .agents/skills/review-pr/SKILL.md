---
name: review-pr
description: Explain and review an RPent pull request, re-review its revisions, or review an explicitly selected local diff. Check correctness, current main conventions, prose, and test value, then provide precise draft comments.
---

# Review an RPent change

Use [AGENTS.md](../../../AGENTS.md) and
[CONTRIBUTING.md](../../../CONTRIBUTING.md) as repository guidance. Prioritize
correctness and real maintenance costs. Adapt the review to the change;
the guidance below is not an exhaustive checklist or a quota of findings.

## 1. Establish scope and versions

- For a PR, use available GitHub tools or Git to read its description, live
  base repository/branch/SHA, exact head SHA, diff, checks, and review threads.
  Fetch those refs and the latest `RLinf/RPent` main. Verify remote URLs;
  `origin` may be a contributor's fork.
- Use the verified base/head merge base to identify the PR's changes. Read
  full files and consumers at the head, then compare affected interfaces with
  current upstream main. Keep PR scope separate from later main refactors.
- Do not use an unrelated checkout or uncommitted work as upstream evidence.
  Use `git show <ref>:<path>`, `git diff`, and `git log`, or equivalent GitHub
  reads. If a ref is inaccessible, state the missing evidence.
- For local review, establish the requested staged, unstaged, or branch diff
  and its baseline. Identify that snapshot explicitly in the report.

## 2. Understand the change

Identify the problem, previous behavior, proposed behavior, main implementation
choices, and affected entry points. Trace configuration and registration through
the actual callers; verify the PR description against the code.

For RPent, inspect relevant paths across planner/session, prompt/tool schema,
toolkit, Env/VLA RPC, and CLI/Dashboard runtime. Identify the affected callers
and the prompts, tool results, and observations the model receives.

## 3. Review correctness and project fit

Choose the depth of review from the affected contracts and plausible failure
modes. Follow relevant callers beyond the diff where needed. Keep findings
attributable to the change; identify pre-existing problems separately when
they materially affect the review.

- **Correctness and lifecycle:** verify intended behavior, defaults, state
  transitions, and error handling through the affected entry points. For
  concurrent or resource-owning code, inspect cancellation, partial startup,
  cleanup, and ownership of services and artifacts. Check that constraints
  hold at the operation that executes them, including alternate callers.
- **Design and current-main integration:** compare with the nearest current
  implementation and shared helpers, registration, configuration, and optional
  imports. Assess abstractions, compatibility paths, and validation against
  actual consumers and supported data. Propose simplifications with concrete
  evidence while preserving required behavior. Explain any mismatch with a
  main refactor; being behind main or mergeable alone is not evidence of one.
- **Documentation and model inputs:** when public behavior or prose changes,
  use [docs-check](../docs-check/SKILL.md) to verify accuracy, useful explanation,
  and natural, consistent language. For prompt or tool changes, inspect the
  rendered instructions, schemas, results, and affected planner modes; judge
  omissions, conflicts, and repetition by their effect on the model's task.
- **Tests and evidence:** check that coverage exercises the changed contract
  and would detect the intended regression through the relevant entry path.
  Assess whether added cases provide useful coverage and whether existing
  tests can be extended. Follow [AGENTS.md](../../../AGENTS.md) for test design
  and style; avoid speculative nits or issues already resolved by passing gates.

Use [verify-change](../verify-change/SKILL.md) for relevant validation. Existing
CI results and author reports are evidence with a scope; they do not establish
unexercised runtime behavior. Distinguish an integration chain that executes
actions, records `finish`, and cleans up from benchmark task success.

## 4. Report in this order

Use the user's language. Explain the PR before listing findings:

1. **PR explanation:** the problem, main changes and approach, and affected
   behavior or modules. Keep this concise and grounded in the implementation.
2. **Findings:** order by severity. For each, provide a short title, exact
   file/line, triggering condition, impact, evidence, concrete repair, and a
   comment ready to paste into the review. Use P0 for critical immediate
   issues, P1 for high-impact defects, P2 for normal defects, and P3 for minor
   issues; distinguish blockers from optional suggestions.
3. **Verification:** record the reviewed head/base/main refs or local snapshot,
   checks actually run, results, and unverified scope. Say when no actionable
   findings were found without implying every execution path was tested.

Verify each inline location against the current diff. Use the smallest relevant
range and the correct old/new side; include current-main references when they
support a finding. Use a general comment for cross-cutting issues or locations
outside commentable diff ranges instead of inventing a line number.

Draft comments should describe the desired final behavior and necessary
constraints so an author can implement the fix without adding review narration
to code or documentation. Return drafts in the conversation unless publishing
the review is part of the user's authorized request.

## 5. Re-review revisions

Refresh the exact head and any changed base, read existing threads, and verify
each previous finding against the new code and consumers. Mark it resolved,
unresolved, or superseded with evidence. Inspect the fix's effect on callers
and validation, including any new regressions. Update line numbers; an author's
“fixed” reply alone is not verification.
