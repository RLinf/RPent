"""Global prompt definitions shared by environments."""
from __future__ import annotations

from physical_agent.context.prompt_base import BulletList, Numbered

API_OUTPUT = Numbered([
    "1-2 sentence reasoning before each tool call (observation -> decision).",
    "Don't re-read files you already read. Don't view_driver_state if you just got the state from a primitive tool call.",
    "Be parsimonious with tokens. Numerical coords in 3 decimals is enough.",
    "When `finish` is called the agent halts. Save artifacts BEFORE finish.",
])

CLI_OUTPUT = BulletList([
    "Brief reasoning before each Bash/Read call (1-2 sentences).",
    "Don't re-read files already in this session.",
    "Numerical coords in 3 decimals are enough.",
    "Stop immediately after writing the audit. Do not chat further.",
])

API_USER = {
    "Task": "Cell: suite={{suite}}  task={{task}}  seed={{seed}}.",
}

CLI_USER = {
    "Task": """
    - suite:   {{suite}}
    - task:    {{task}}
    - seed:    {{seed}}
    - output_dir: {{output_dir}}
    - output:  {{output_dir}}/
      - audit filename:  {{recipe_tag}}.json
    """,
}
