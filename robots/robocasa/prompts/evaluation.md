## RoboCasa evaluation protocol

Read state_00 and images; check task_language and success_criteria. Check task_progress after every command. Stop when state.success == True and call finish(status='success', ...); if genuinely stuck after honest exploration, call finish(status='stuck', ...). The reset tool is disabled.
