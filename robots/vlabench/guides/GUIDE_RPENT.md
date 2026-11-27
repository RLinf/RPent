# VLABench operating guide

Use the task instruction and the visible scene to decide which object or part
matters. Similar-looking objects can differ by markings, arrangement or intended
use. Do not assume a category center is the required contact point, and do not
reuse a memorized world coordinate, table height or task recipe.

Use `agentview` and the side cameras for scene layout and approach clearance;
use `wrist` for local detail when it has a useful view. The wrist camera moves
with the arm. Pick an unoccluded visible surface in the current named camera and
check that its depth is valid. Change the viewpoint when the target part is
hidden or ambiguous; grounding the same cached frame again is not a new view.

A depth point locates a surface, not a complete gripper pose. Decide the finger
orientation, contact part and approach clearance separately. Compute any
WORLD-meter clearance offset from the localized `back_project` surface
coordinates before passing `xyz` to `move_pose` (or follow with `move_delta`
using `dxyz`), not a camera-axis displacement, and do not introduce an unrelated
absolute coordinate. Align the fingers with the intended grasp or contact
surface.

For pick-and-place, consider an approach above or beside the object, a short
final approach, closure, and a modest lift before transport. These are optional
patterns, not a fixed task program. Check whether the object follows the
fingers; closure alone does not show a grasp. Keep the intended grip during
transport and verify the placement after release. No finger-pad contact sensor
is exposed.

For pressing, pulling or articulated contact, first identify the moving part and
its direction of travel. Approach deliberately and use bounded corrections near
contact. If nothing changes, reassess alignment, orientation and reach instead
of pushing the same unverified point again. An IK rejection is not a reason to
increase the motion length indefinitely.

Use the returned pose and errors to correct the remaining motion. A command that
ends outside tolerance may still have moved the arm, and a relative command
repeats from its new pose. When progress stalls, change the waypoint,
orientation or viewpoint. Native IK does not certify collision-free paths or
held-object clearance.

Stop on native task success or an explicit host stop. A failed local attempt
alone does not end the task.
