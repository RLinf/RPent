"""User prompt section bodies for a concrete LIBERO evaluation cell."""

from __future__ import annotations

CELL = """- suite:      {{suite}}
- task:       {{task}}
- seed:       {{seed}}
- output_dir: {{output_dir}}
- audit:      {{output_dir}}/{{recipe_tag}}.json
- recipe:     {{output_dir}}/recipe_{{recipe_tag}}.jsonl"""


MODE = """Inspect the embedded high-resolution images returned by
view_env_state, then use back_project or segment to localize objects before
motion."""


BEGIN = """Read MEMORY.md and the guides, then call
`view_env_state({"step": 0})` and inspect its embedded images. Localize every
task-relevant entity before planning and execution."""
