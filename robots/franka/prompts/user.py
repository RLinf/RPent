"""User prompt sections for one Franka task run."""

TASK = """- task_id: {{task_id}}
- task_name: {{task_name}}
- instruction: {{instruction}}
- success_criteria: {{success_criteria}}
- output_dir: {{output_dir}}"""

CONSTRAINTS = """{{constraints}}"""

BEGIN = """Call view_env_state with step 0, inspect both camera views and the TCP
state, then execute the task conservatively."""
