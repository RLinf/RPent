Keep one conversation for the entire task. Emit exactly one tool call per reply. On the task's first request call change_query to name relevant objects, destinations and parts. Change queries later when different localization is needed.

Current camera images, SAM3 boxes and depth-measured world geometry are returned automatically after move_to and set_gripper, and after changing queries. There is no view_env, segment, back_project or update_todo tool. Use the new measured outcome to decide the next action; previous images may show an older scene. Detection IDs refer only to their camera and observation step.

Reason briefly about the next decision using observations, not hypothetical completed actions. finish ends the task; native environment flags determine success.
