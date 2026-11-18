## Control guidance

Use one chunk near contact, near success, instability, or for a
small correction; two for ordinary stable progress; three only for a
recipe-supported continuity-sensitive phase already moving correctly. When VLA
has correct contact and visible progress, avoid interrupting it with speculative
primitives. Repeated VLA calls are allowed; after an unproductive chunk, use
fresh evidence to choose whether to continue, shorten the next chunk, or improve
binding, visibility, or physical staging first.

Prefer VLA for grasp/re-grasp, receiving-arm grasp, bimanual coordination,
insertion, hanging, tool use, and contact-rich motion. Use primitives after
verified state for measured free-space transport, staging, retreat, release, or
one small geometric correction. Never transport because a gripper merely looks
closed: also require visible target motion, elevation, or an emptied source.
Never call a primitive just to test whether it helps. For planner residuals,
guarded low approaches, physical state shaping, and wrist-sweep safety, follow
robots/robotwin/guides/GUIDE_RPENT.md and re-observe after every primitive.

## Perception

Use the head view as semantic authority for identity,
distractors, destinations, language relations, and global progress. Use the
matching current wrist view to refine geometry for that same chosen candidate;
do not let it silently switch to a look-alike. Pair RGB and world maps from the
same step, view, and resolution. World maps are [row,col] -> [x,y,z] metres and
may contain NaN; visible surface points are not automatically object centers.
Relocalize after occlusion, contact, or substantial arm/object motion.
