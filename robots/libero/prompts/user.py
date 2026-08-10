"""User prompt section bodies for a concrete LIBERO evaluation cell."""

from __future__ import annotations

CELL = """- suite:      {{suite}}
- task:       {{task}}
- seed:       {{seed}}
- output_dir: {{output_dir}}
- audit:      {{output_dir}}/{{recipe_tag}}.json
- recipe:     {{output_dir}}/recipe_{{recipe_tag}}.jsonl"""


MODE = """Inspect `agentview_high.png` returned by `view_env_state`, then use
`back_project` or `segment` to localize objects before motion."""


BEGIN = """Read MEMORY.md and the guides, then call
`view_env_state({"step": 0})` and inspect `agentview_high.png`. Localize the
target, then plan and execute."""
