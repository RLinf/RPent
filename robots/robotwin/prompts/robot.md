## RoboTwin observation and control

RoboTwin uses the dual-arm aloha-agilex embodiment. Each arm contributes six
joints and one gripper value to qpos14. Gripper 0 is closed and 1 is open.
Cartesian primitives target the EEF in world-frame metres with quaternions
ordered [qw,qx,qy,qz].

Before the first motion, call
`read_text_file(path="robots/robotwin/guides/GUIDE_RPENT.md")` and read the
complete guide. Then inspect view_env_state(step=0) and the head image. Use the head view for
object identity, language relations and global progress. Refine geometry with
the wrist view for the arm and target already selected. Pair RGB and world maps
from the same step, view and resolution. World maps index [row,col] and return
[x,y,z] in metres. They can contain NaN and measure visible surfaces.

Use the action model for contact-rich motion and coordinated arm movements.
Use scripted primitives for measured staging, free-space transport, retreat,
release or a small geometric correction. Verify a grasp from target motion,
elevation or an emptied source before transport. Follow the guide for guarded
approaches, planner residuals and wrist sweeps. Re-observe after each primitive.

## Episode status

Track remaining native steps as step_lim - take_action_cnt. Only fresh
TASK_ENV.eval_success=true confirms success. Stop motion at native success or
the episode limit. Check fresh status before finish and call finish exactly
once when closing the run.

In exploration, reset starts a fresh episode and may resample the layout.
The exploration audit records task_name, task_config, seed, eval_success, total
attempts, final_state and successful_strategy. List every recipe-eligible command
and its actual parameters after the last reset in order, excluding reset itself.
A winning recipe requires a recorded success step after that reset.
