## Safety rules

1. Inspect view_env_state before motion and after every mutating tool call.
2. Express move_delta and rotate_delta in the fixed robot base frame.
3. Use small purposeful corrections and compare both camera views.

For grasp tasks, open the gripper before calling vla_grasp near contact.
Finish only when the success evidence is visible and consistent with state.
