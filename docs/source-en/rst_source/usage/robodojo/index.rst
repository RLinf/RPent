RoboDojo
========

.. toctree::
   :maxdepth: 1
   :hidden:

   installation

RoboDojo connects Isaac Sim / IsaacLab, dual ARX-X5 arms and an XPolicyLab Pi_05
policy to RPent's shared planner, tool and memory infrastructure.
The integration is experimental: offline contracts do not establish simulator
compatibility or task success. For installation and a runnable CLI example,
see :doc:`installation`.

For shared backend interfaces, see :doc:`../../development/add_robot`.

Key modules
-----------

* ``robots/robodojo/env_server.py`` — Isaac Sim RPC server (main-thread
  rendering; head + dual-wrist RGB-D with intrinsics/extrinsics; joint/ee
  actions; per-camera video recording).
* ``robots/robodojo/env_client.py`` — rpent-side client inheriting
  ``BaseEnvClient``.
* Shared ``rpent/robots/components/xpolicylab_vla_server.py`` and
  ``rpent/robots/components/xpolicylab_vla_client.py`` — Pi_05 policy
  service (XPolicyLab WebSocket) adapted to the shared ``BaseVLAFacade`` /
  ``BaseVLAClient`` protocol.
* ``robots/robodojo/toolkit.py`` / ``tools.py`` — primitives:
  ``view_env_state``, ``back_project``, ``segment``, ``move_to``,
  ``set_gripper``, ``pi0_pick``, ``stabilize``, ``place_in_bin``,
  ``get_reward_details``, etc.
* ``robots/robodojo/robot_spec.py`` — ``RobotSpec`` factory (CLI, run config,
  runtime orchestration).
* ``robots/robodojo/tasks.py`` — task inventory from the configured source checkout.

Implementation details
----------------------

The environment server initializes Isaac Sim before simulator imports and
serializes simulator requests on its main thread for camera rendering. Each
process owns one simulator application. Reset returns an observation dictionary;
step returns ``(obs, reward, done, info)``. Chunk stepping is unsupported, so
primitives issue individual steps through the environment action path, retaining
its bounds and counters.

The policy runs in a separate Python environment through
``rpent.robots.components.xpolicylab_vla_server``.
Its ``--policy-root`` points to ``XPolicyLab/policy/Pi_05`` in the configured
checkout. The adapter passes observations/actions through unchanged and
serializes ``update_obs``/``get_action`` with ``reset``; it does not isolate
sessions. RoboDojo requires three-camera inputs and 14-DoF joint actions;
selecting another policy backend does not convert these formats.

The runtime passes the run output directory as the environment's ``--save-dir``
and the policy launcher's ``--output-dir``. The inner policy log is
``vla_server.log`` in that directory. When started directly without these flags,
both services default to the current working directory. Use separate output
directories for concurrent runs.

Tools and information access
----------------------------

The planner supplies tools directly; call their listed names. Dustbin placement
and bottle-specific scoring guidance are included only in the
``put_bottles_into_dustbin`` task context, not the generic system prompt.
Recorded-state reading and calibrated depth projection are local to
``robots/robodojo/tools.py``. Projection uses Isaac's negative optical Z axis;
segmentation uses the shared SAM3 client. ``view_env_state``, ``back_project``,
``segment``, ``get_reward_details`` and ``get_safety_status`` are read-only:
they do not advance the environment or trigger post-action state capture.

``robots.robodojo.tools.TOOL_GROUPS`` marks direct outputs as ``general``
(depth/segmentation and motion), ``privileged`` (``get_reward_details`` and
``get_safety_status``), or ``mixed`` (``view_env_state``, ``set_gripper``,
``place_in_bin``). Safety alarms expose ground-truth object world coordinates;
reward details expose per-object success predicates. Mixed outputs include
success, status, or historical results that can contain privileged information.

The Python toolkit factory accepts ``allowed_tool_groups``; for example,
``frozenset({"general"})`` registers only general robot tools. The default
``None`` preserves the existing tool set. This hook filters schemas and
handlers, not automatic post-action state, raw observation fields, logs,
memory, or common file tools. It is not an evaluation isolation mode.

Development and frozen replay
-----------------------------

Normal planners retain the development tool set and privileged feedback.
``--planner flash`` selects eval-fair: RPent's native Flash planner invokes
``RobotSpec.run_flash`` without an LLM. ``--explore`` remains unsupported for
RoboDojo; development here means the normal planner loop, not that CLI mode.

Development writes ``flash_trace.json`` in the run output directory. To record
a transferable waypoint, call ``segment`` on ``cam_head``, then ``back_project``
at its returned ``centroid_rc`` (row, col), then the action. The pixel is the
floored mean of foreground mask coordinates, not the box center. Recording
and replay share this derivation; recorded mask area and coordinate sums
allow export to reject a changed centroid. Do not batch different objects'
segmentations ahead of depth calls. Repeat that perception pair before
each action. Export the trace before evaluation:

.. code-block:: bash

   python -m robots.robodojo.flash.generate \
     --trace /path/to/dev-run/flash_trace.json \
     --task put_bottles_into_dustbin \
     --destination /path/to/memory/robodojo/flash

The version-2 JSON contains a task, symbolic SAM3 queries, the explicit
``sam3_mask_centroid_floor_v1`` derivation method, optional wrist-camera
refinement, and ordered actions with arguments and three-dimensional offsets.
It contains no reference object coordinates, score or predicate results.
Export accepts ``move_to``, ``set_gripper`` and ``pi0_pick``; unsupported actions
(including ``place_in_bin`` and ``stabilize``), failed calls and unanchored moves
are rejected. Record those operations as supported basic actions instead.
Existing plans are never overwritten. Review and freeze the plan before eval;
no candidate selection or plan writing occurs during replay.
Version-1 box-center plans and traces without mask moments must be recorded
again; they are not silently converted. The shared XPolicyLab facade owns only
the policy processes it spawns and stops them on close, startup failure,
SIGTERM and normal interpreter exit; borrowed policy services remain running.

Use the usual runtime flags together with:

.. code-block:: bash

   rpent --robot robodojo --task put_bottles_into_dustbin --layout 1 \
     --planner flash --memory-profile local --memory-dir /path/to/memory/robodojo \
     --source-root /path/to/RoboDojo \
     --sim-python /path/to/sim-env/bin/python \
     --pi05-python /path/to/pi05-env/bin/python

The plan is ``flash/<task>_plan.json`` under the selected robot memory. HF mode
uses the native ``robodojo/flash/**`` sync filter; no RoboDojo plans are bundled
or guaranteed to be published there. Missing or invalid plans fail closed.

Replay first grounds all anchors in the opening head frame. Each waypoint is
the live anchor plus its recorded offset (maximum offset length 0.5 m).
Export with ``--refine-camera cam_left_wrist`` or ``cam_right_wrist`` to require
a live wrist reading before moves; it must agree within 5 cm, otherwise replay
stops. This does not move the wrist to create visibility: record an approach
that makes the selected view useful, or leave refinement disabled.
``pi0_pick`` is unchanged; acceptance additionally requires a closed gripper
and wrist-localized object within 12 cm of either EEF. A failed hold retries
the preceding approach with fresh head grounding, at most three pick attempts.
Errors, lost localization, unreachable waypoints and step-budget exhaustion
stop replay without reset. These checks are not a collision-safety guarantee.

Eval-fair excludes privileged tools and common file/memory tools. The service
advertises the mode in its metadata, rejects reset and diagnostic RPCs, returns
zero reward and no task-success feedback, and exposes only RGB-D, camera
calibration, instruction and arm proprioception. Ground-truth bottle alarms
are disabled. Existing dev endpoints are rejected for eval-fair; boot creates
the episode and the eval client does not reset it again. Automatic state logs
therefore contain public observations and action diagnostics only. The reduced
eval prompts contain no scoring instructions and are unused by Flash.

Flash ``done`` and its planner completion status mean the frozen sequence
completed, not that the official task predicate passed. Official scoring must
remain outside the replay context. GPU, real policy and simulator validation
are required before claiming benchmark compatibility or success.

Task language
~~~~~~~~~~~~~

Task-language RPC and public observations use RoboDojo's description manager,
not raw ``gen_instruction`` templates. If the manager is unavailable or returns
unresolved language, labels are filled from ``get_label_descriptions`` for
environment 0, choosing the first description deterministically. Missing labels,
empty language and remaining template markers raise an explicit error; task names
are not substituted. The official instruction stays public in eval-fair.
Omitting ``pi0_pick.prompt`` uses this resolved official language. Explicit
overrides remain supported for contact segments but must identify the intended
object; unresolved markers are rejected before inference. Official multi-object
tasks do not specify a grasp order, and descriptive overrides do not guarantee
that a checkpoint can select arbitrary instances. Verify the actual held target.

Capability scope and limitations
------------------------------------------------------------

RoboDojo provides dual-arm motion and gripper primitives, three-camera RGB-D,
SAM3 perception, XPolicyLab Pi_05 and frozen Flash replay. Task names come from
the configured checkout; examples include ``put_bottles_into_dustbin``,
``fill_pen_holder`` and ``stack_bowls_random``, not a validated success suite.
``place_in_bin`` is registered only for ``put_bottles_into_dustbin``.
Handover is not implemented. Low-Z tabletop and lateral scripted IK motions
have reachability limits: inspect ``reached`` and ``dist_to_target`` rather
than assuming the commanded pose was achieved. See :doc:`../flash` for the
shared evaluation-only planner; replay executes actions, it is not a
read-only robot operation.

Bounded smoke runs and shutdown diagnostics
-----------------------------------------------

For ``fill_pen_holder``, use an explicit smoke budget of
``--planner-timeout-s 1500 --max-turns 40`` with an outer
``timeout --signal=INT --kill-after=20s 1700s``. These are validation overrides,
not regular defaults. The outer limit leaves time for startup and cleanup and
keeps the run below 30 minutes. Stop after a timeout or failed gate; inspect the
last completed tools and provider latency before scheduling another attempt.

The CLI records planner errors in ``transcript_<cell>.json`` (``error``) and
final errors, including finalization failures, in ``run_diagnostics.json``.
Read these in both dev and Flash runs; a zero process status is not proof of
official task success.

Verify all three videos by full decoding and check every owned server's exit
status. Between ``[robodojo-env] shutdown begin`` and process exit, require no
``[Error]``, traceback, or ``Fatal Python error``. Headless GLFW warnings are
expected noise, not a reason to ignore shutdown errors. The environment
releases writers, camera annotators/render products, and syntheticdata graph
handles before stopping Replicator and closing the stage/app on the main thread.
