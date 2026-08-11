"""User prompt sections for one dual-Franka task run."""

TASK = """- task_id: {{task_id}}
- task_name: {{task_name}}
- instruction: {{instruction}}
- success_criteria: {{success_criteria}}
- output_dir: {{output_dir}}"""

CONSTRAINTS = """{{constraints}}"""

BEGIN = """Call view_env_state with step 0, inspect the left-wrist, base, and
right-wrist views together with both arms' TCP state, then execute the task
conservatively one arm at a time."""
