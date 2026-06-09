---
name: pi0-pre-pos-can-hurt
description: "Pre-pos move_to before pi0_pick can interfere with Pi0's grasp — try Pi0 from default home pose for hard-to-grasp objects (plates, large flat items)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cf2f87e2-37a3-4480-8b9a-4e5666fbf61f
---

**Rule**: For some PRO task variants where Pi0 fails to pick a normally-graspable object, **try `pi0_pick` from the default home pose with NO pre-pos `move_to`**. Pi0 has its own internal approach trajectory; pre-positioning the gripper near the object can break it.

**Why**: libero_goal t4 task ("Put the plate on the top of the drawer") was failing with both sub-instr "pick up the plate" and full BDDL prompt when preceded by a pre-pos move_to (gripper centered above plate). After 3 strategies and image inspection, removing the pre-pos and letting Pi0 plan from the default eef home (-0.21, 0, 1.17) succeeded on the very next attempt — Pi0 grasped the plate in 20 chunks with `obj_lifted=0.109` m (real lift).

The hypothesis: Pi0's policy was trained on episodes that START from a specific pose distribution. A pre-pos move_to puts the eef close to the object but in a config Pi0 doesn't know how to act from — its policy outputs garbage or no-op. Starting from the trained home pose lets Pi0 use its learned approach trajectory.

**How to apply**:
- For objects Pi0 SHOULD know how to grasp (training-distribution objects: bowl, plate, cream_cheese, wine_bottle, mug, moka_pot in libero_goal/object scenes), try `pi0_pick` from default home pose FIRST.
- Only add a pre-pos move_to if Pi0 from home doesn't grasp (e.g. cluttered scene, occluded target).
- For swap/task perturbations where pre-pos previously helped: still try home-pose first; pre-pos may now hurt because Pi0's internal trajectory expects to start from home.

**Related**: [[pi0-delivery-service]] [[pi0-pick-full-prompt]] [[scripted-pick-limits]]
