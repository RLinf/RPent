## Dual-Franka evaluation startup

Build a conservative localization table for the current category from the latest
inline image and update it after each primitive.

## Begin

Call describe_dual_franka_setup, then request_scene_reset before acting.
After operator confirmation and a successful reset, inspect the fresh visual evidence
together with both arms' TCP/gripper/joint-health state, and execute the task
conservatively with the exposed bounded tools. Auxiliary camera views are
artifact views for targeted follow-up inspection only.
