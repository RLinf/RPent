# RoboDojo Robot and Execution Guide

Use this guide to interpret observations and control RoboDojo's dual ARX-X5
arms through the registered tools. The system prompt defines task strategy,
controller selection, and the current run mode. Tool schemas define arguments.

## Arms, grippers, and reference frames

Each fixed-base arm has six arm joints and a two-finger gripper. The `left`
and `right` tool arguments identify the robot's arms, independently of where
they appear in a camera image. Track each hand's held object and intended role.
One arm may hold a support or tool while the other manipulates an object.

Motion targets use world-frame metres and quaternions in `[w,x,y,z]` order.
The reported end-effector pose and `move_to` target refer to `link6`. This
reference point is distinct from the fingers' contact points and the grasped
object's center. A point back-projected from an image lies on a visible
surface; sending that point directly as an end-effector target does not account
for the gripper geometry or approach orientation.

Establish the relevant grasp or contact geometry from current observations.
Reassess the object-to-hand relationship after a new grasp, slip, or handover.
Do not carry a measured offset from one grasp into another without checking it.
A six-joint arm must satisfy position and orientation together: a reachable
position can be unreachable with the requested orientation.

## Gripper control and evidence of a hold

`set_gripper(arm, val, steps=10)` uses a normalized opening target: `0` is
closed and `1` is fully open. Intermediate values specify neither physical
widths in metres nor gripping forces. The command interpolates from the
measured opening while retaining the other controller targets. Its `steps`
counts action targets, not physics steps. `release` performs the same operation
with a default target of `1`.

Commanded closure and measured opening are evidence about the gripper, not
proof that it holds the intended object. Inspect whether that object moves
with the hand and retains its relationship to the fingers. Use head and wrist
views together when contact is occluded. An uncertain hold calls for better
evidence before transport, not automatic opening or repeated tightening.

When placing, distinguish readiness to release from successful release. First
inspect the intended support, alignment, and clearance; after opening, verify
that the object remains supported as the hand separates. For a handover,
establish receiver hold before opening the giver. Keep the gripper closed when
the task requires continued holding or tool use.

## Images, depth, and functional geometry

The head camera provides scene context, identities, destinations, and spatial
relations. The left and right wrist cameras provide complementary local views.
Track the same selected object across views; a close-up must not silently
replace it with a similar object. Image-left is not automatically robot-left
or a world-axis direction.

Use pixels and depth from the same recorded step and camera. World-map queries
use `[row,col]` pixels; invalid depth is not a usable point. Compare several
visible surface samples when a single point is ambiguous. `render` refreshes
the observation without commanding robot motion; repeated captures alone do
not establish that an object has settled over physical time.

SAM3 identifies visible regions. Inspect its overlay before using a result.
`point_rc` lies in the mask; a mask centroid can lie outside a concave region.
Neither automatically identifies a grasp point, cavity center, handle opening,
or support surface. Distinguish the solid rim from the opening and the object
body from the part that must contact a fixture. Re-localize after motion,
occlusion, or deformation; cloth does not retain a fixed rigid offset to a hand.

## Motion targets and execution feedback

Initial observations and motion feedback include `episode_status.step`,
`step_limit` and `remaining_steps`: the native actions used, total budget and
remaining budget. Read them before planning and after motion. The top-level
`step` identifies an observation; historical records show the budget at capture
time. Reading a saved observation does not consume action steps.

`move_to(arm, xyz, quat=None, gripper=None, substeps=25)` plans a full pose.
Omitting `quat` preserves the measured end-effector orientation at the call;
omitting `gripper` retains its controller target. Preserve these quantities
while carrying unless a change serves the current phase.

For `move_to` and `rotate_wrist`, use the system prompt's `substeps` bands:
`25` by default, `10` for short moves or small rotations, `5` for very small
corrections, and `1` only for tiny corrections or holding a target. Do not use
`0` or values above `25`. Check clearance for the arm, fingers and held object.

`substeps` caps sampled path waypoints. Each executed waypoint consumes one
native action step (0.04 seconds of simulation), so fewer points also mean
less execution time. `1` sends only the final waypoint; it does not wait for
arrival. Subsampling can remove a planned detour. Use intermediate targets
where needed to preserve the route. Native planning models a simplified table
and self-collision, not every scene object or the opposite arm.

Inspect `completed`, `executed_steps`, `stop_reason`, the achieved pose, and
fresh images. `final_dist_m` measures position error, not orientation accuracy,
object placement, or task success. `completed` means the selected commands ran,
not that the target pose was reached. Verify orientation and the held object
before continuing; optimize total phase steps rather than individual call size.
An interrupted execution may have moved the arm. Plan the next action from the
measured state rather than the last target.
If a target fails, inspect reach, orientation, clearance, and localization before
changing a waypoint or approach; do not repeatedly send an unchanged target.

## Held objects, rotation, and contact

Plan clearance for the arm, fingers, held object, and any tool working end.
The other arm can obstruct a path even while its controller targets are held.
Near a support, rim, opening, or completed arrangement, execute one purposeful
correction and inspect its result before choosing another.

`rotate_wrist` changes yaw about world Z in degrees while retaining the
end-effector position target. Fingers and held objects can sweep an arc around
that point. Inspect surrounding clearance and verify the actual object pose
after rotation. This tool does not provide pitch or roll for tilting or pouring;
those require a suitable full-pose target or VPP2 motion. Object rotation need
not equal the commanded end-effector rotation if the grasp slips.

For pushing or tool use, judge progress at the contact or working end, not only
at the wrist. For articulated objects, follow the mechanism's permitted motion
and inspect its resulting state. Preserve useful contact while it advances the
task; an incomplete phase does not by itself justify breaking contact.

## Withdrawal, return, and task completion

Observe each arm's initial end-effector position and orientation at the start
of an attempt. Returning to the initial pose is different from withdrawing a
short distance, moving to world zero, or resetting the environment. Where the
task requires return, first complete the intended placement or release, then
choose a clear withdrawal path and restore both arms' initial end-effector
poses without disturbing achieved relations. There is no separate home tool.

A supported object must remain in the requested relation when the robot no
longer carries it. Do not release an object that the instruction requires the
robot to hold. Use observations to assess phase completion and the native
success feedback, when available, to confirm the whole task. Once native
success is reported, stop robot actions; do not add a cleanup motion.
