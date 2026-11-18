## RoboCasa evaluation protocol

Read state_00 and images; check task_language and success_criteria. Check task_progress after every command. Stop when state.success == True and call finish(status='success', ...); if genuinely stuck after honest exploration, call finish(status='stuck', ...). The reset tool is disabled.

Use back_project_batch or query_world_map to locate targets. If the target is far (|xy| > 0.8m), navigate_to first. Use rldx_arm / rldx_skill for VLA grasping. Then script free-space carry with move_to (gripper omitted) and release. After every command, view_env_state checks task_progress.

Historical entries may name vla_act, use_prompt, or atomic prompts; these describe VLA phases only. Use current rldx_skill / rldx_arm with complete live task_language.
