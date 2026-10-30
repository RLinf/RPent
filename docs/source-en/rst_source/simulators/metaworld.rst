MetaWorld
=========

MetaWorld exposes a Sawyer arm through RGB-D observations and bounded Cartesian
control. It supports the ten task types in MetaWorld 3.0.0's MT10 catalog,
each run as a separate fixed MT1 episode. It uses a visual planner with scripted
motion primitives; no VLA or segmentation checkpoint is required.

Supported tasks
---------------

* ``reach-v3``, ``push-v3``, ``pick-place-v3``
* ``door-open-v3``, ``drawer-open-v3``, ``drawer-close-v3``
* ``button-press-topdown-v3``, ``peg-insert-side-v3``
* ``window-open-v3``, ``window-close-v3``

Installation and one episode
----------------------------------------

Use a dedicated Python 3.11 environment on Linux with an EGL-capable GPU:

.. code-block:: bash

   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[metaworld]"

Configure a planner as described in :doc:`../guides/configure_planner`, then run:

.. code-block:: bash

   rpent --robot metaworld --task reach-v3 --seed 0 \
     --planner api --model "$PLANNER_MODEL" \
     --memory-profile local --memory-dir ./memory/metaworld \
     --output-dir ./runs/metaworld-reach

Set ``PLANNER_MODEL`` to the provider-qualified model you configured.
Only local memory is currently supported; an empty directory is sufficient.
The CLI starts and cleans up its own environment service. ``--sim-python``
selects a separate simulator interpreter. ``--env-endpoint`` borrows an
existing service whose task, seed, camera and step limit must match.
Connecting resets the fixed episode, so use an exclusively owned service.
The borrowed service remains its owner's responsibility and is not stopped by
the runner. To record its episodes, start it with ``--video-dir``:

.. code-block:: bash

   python -m robots.metaworld.env_server --task drawer-open-v3 --seed 0 \
     --port 18765 --video-dir ./runs/standalone/videos

Observations and controls
-------------------------

``view_env_state`` returns the latest 480 x 480 RGB image, end-effector
position, gripper opening, instruction and simulation step count.
``back_project(row, col)`` uses the corresponding metric depth and camera
calibration to return a visible surface in world meters. Camera axes are
x right, y up and z backward. Pixel-center intrinsics and single-sample rendering
keep RGB and depth aligned; multisample antialiasing is disabled. Invalid pixels,
nonpositive/nonfinite depth and far-plane background are rejected.
Neither observation tool advances physics or updates native frame history.
Projection uses the saved image's depth and calibration even if a subsequent
motion's state capture fails; inspect its recorded step number before acting.

``move_to(target_xyz, gripper, max_steps=50)`` applies closed-loop XYZ
corrections using MetaWorld's native 0.01 meter action scale. Each call is
limited to 100 steps; it returns ``reached``, distance and actual step count.
The wrist orientation is fixed. Gripper +1 closes and -1 opens during movement;
use ``set_gripper(gripper, steps=10)`` to actuate in place (at most 20 steps).
A zero-step move does not change the gripper.
Motions are not collision-aware and stop at the episode boundary. A reached
waypoint is not a task-success claim.

The default camera is ``corner2``; ``--camera`` also accepts ``corner``,
``corner3``, ``topview``, ``behindGripper`` and ``gripperPOV``.
The default episode limit is 500 steps. An episode ends at its limit or the
native task-success predicate. Native task tolerances can be larger than
the motion primitive's position tolerance.

Results and scope
-----------------

``result.json`` records native success separately from the planner's finish
assessment, together with seed, camera, episode and planner budgets.
``environment.json`` records the simulator's actual dependency versions, native
step limit, action scale, rendering configuration and video path. These come from
the simulator interpreter, including when ``--sim-python`` is used.

``states.json``, RGB and depth artifacts record each tool boundary. Owned servers
stream ``videos/episode_0000.mp4`` with the initial frame and one RGB frame after
every accepted simulation step, at the native control rate. Read-only calls and
rejected actions add no frames. Reset closes the previous clip and selects a new
numbered filename without overwriting earlier clips. Video frames are streamed
to the encoder, not accumulated in planner memory. No reset or native expert
policy is exposed to the planner.

The adapter sends only camera observations and robot proprioception; MetaWorld's
flat observation also contains object and goal coordinates, which are omitted.
This is a single fixed MT1 episode (``num_tasks=1``) seeded at construction;
it is not the MT10/MT50 evaluation protocol or a claim of benchmark success rates.
Exploration and Flash replay are not implemented.

The service wraps native MetaWorld directly. RLinf's existing MetaWorld wrapper
owns vectorized training workers and does not expose calibrated depth/render
operations required here. The adapter reuses RPent's Env RPC, main-thread
dispatch, runtime ownership, toolkit and artifact contracts; native MetaWorld
still owns physics, task definitions and success.

Verification
------------

.. code-block:: bash

   pytest tests/unit_tests/robots/metaworld -q
   CUDA_VISIBLE_DEVICES=0 bash tests/e2e_tests/run_gpu_suite.sh \
     metaworld /path/to/new-results /path/to/new-venvs

The clean GPU suite checks RGB-D for all ten tasks, all supported cameras,
pixel-center projection against a rendered tilted plane, read-only observation
invariants, invalid/late action rejection, a bounded real tool call, recorded
finish, video frame counts across reset, and owned-process cleanup.
It does not call a remote planner or require task success. Maintainers can request
the same suite with ``/ci-metaworld``.
