"""Local-memory variant of the single-attempt LIBERO evaluation prompt."""

from __future__ import annotations

from robots.libero.prompts import system as base
from rpent.context.prompt_utils import Numbered, PromptNode

(
    _,
    STEP_READ_GUIDES,
    _,
    STEP_INSPECT_INITIAL,
    STEP_PERCEPTION_PASS,
    STEP_EXECUTE,
    STEP_PRIMITIVES,
    STEP_RECOVERY,
    STEP_FINISH,
) = base.WORKFLOW_STEPS

MEMORY_PROFILE = """Use the LOCAL exploration corpus for this evaluation. Its three layers have
different jobs and all applicable layers are mandatory:

1. GLOBAL: `{{memory_dir}}/global/` — reusable robot/perception/primitive lessons.
2. SUITE: `{{memory_dir}}/suite/suite_libero10_<regime>_t{{task}}.md` — the
   task/regime strategy, validated ranges, and failure table.
3. TASK: `{{memory_dir}}/task/{{reference_tag}}.json` plus
   `{{memory_dir}}/task/recipe_{{reference_tag}}.jsonl` — the matched successful
   audit and command order from seed 0.

Read the task pair and the exact suite leaf when present, then select only the
relevant global leaves through `MEMORY.md`. Recipes are technique references,
not coordinates: re-localize every entity in the current image. Never read or
promote `_inbox/`, `_merged/`, or `wip/` during evaluation."""

STEP_READ_LOCAL_MEMORY = """READ THE THREE LOCAL MEMORY LAYERS FIRST:
- task audit: `{{memory_dir}}/task/{{reference_tag}}.json`
- task recipe: `{{memory_dir}}/task/recipe_{{reference_tag}}.jsonl`
- suite leaf: find the matching task/regime leaf under `{{memory_dir}}/suite/`
- global index: `{{memory_dir}}/MEMORY.md`, then only relevant leaves under
  `{{memory_dir}}/global/`

If a layer is absent, state that explicitly and continue with the available
validated layers. Record the exact files used in final `strategy_notes`. Treat
absolute coordinates as stale and re-derive them from this scene."""

WORKFLOW_STEPS = (
    STEP_READ_LOCAL_MEMORY,
    STEP_READ_GUIDES,
    STEP_INSPECT_INITIAL,
    STEP_PERCEPTION_PASS,
    STEP_EXECUTE,
    STEP_PRIMITIVES,
    STEP_RECOVERY,
    STEP_FINISH,
)


def system_prompt() -> PromptNode:
    """Assemble the local suite/task/global evaluation prompt."""
    return {
        "ROLE AND EVALUATION": base.ROLE_AND_EVALUATION,
        "MEMORY PROFILE — LOCAL SUITE + TASK + GLOBAL": MEMORY_PROFILE,
        "PROVEN LEVERS": base.PROVEN_LEVERS,
        "RUNTIME": base.RUNTIME,
        "YOUR GOAL": base.GOAL,
        "RULES (NON-NEGOTIABLE)": base.RULES,
        "LOCALIZATION": base.LOCALIZATION,
        "FIRST-STEP ALGORITHM": base.PERCEPTION_ALGORITHM,
        "WORKFLOW": Numbered(WORKFLOW_STEPS),
        "KEY HYPERPARAMETERS": base.KEY_HYPERPARAMETERS,
        "OUTPUT DISCIPLINE": base.OUTPUT_DISCIPLINE,
    }


__all__ = ["system_prompt", "WORKFLOW_STEPS"]
