RoboDojo
========

.. toctree::
   :maxdepth: 1
   :hidden:

   installation

RoboDojo is a pluggable simulation backend for RPent (``rpent --robot robodojo``)
that brings Isaac Sim / IsaacLab (dual ARX-X5 arms, Pi_05 policy) into the
RPent LLM-in-the-loop runner, alongside the existing LIBERO / RoboCasa /
RoboTwin backends. The planner (LLM), toolkit protocol, SAM3 perception, and
memory layers are reused unchanged; only the "body" (simulator/robot) is
swapped.

Key modules
-----------

* ``robots/robodojo/env_server.py`` — Isaac Sim RPC server (main-thread
  rendering; head + dual-wrist RGB-D with intrinsics/extrinsics; joint/ee
  actions; per-camera video recording).
* ``robots/robodojo/env_client.py`` — rpent-side client inheriting
  ``BaseEnvClient``.
* Shared ``rpent/robots/components/xpolicylab_vla_server.py`` and
  ``robots/robodojo/vla_client.py`` — Pi_05 policy
  service (XPolicyLab WebSocket) adapted to the shared ``BaseVLAFacade`` /
  ``BaseVLAClient`` protocol.
* ``robots/robodojo/toolkit.py`` / ``tools.py`` — primitives:
  ``view_env_state``, ``back_project``, ``segment``, ``move_to``,
  ``set_gripper``, ``pi0_pick``, ``stabilize``, ``place_in_bin``,
  ``get_reward_details``, etc.
* ``robots/robodojo/robot_spec.py`` — ``RobotSpec`` factory (CLI, run config,
  runtime orchestration).
* ``robots/robodojo/tasks.py`` — task inventory from the configured source checkout.

Quick start
-----------

.. code-block:: bash

   cd <rpent checkout>
   export PATH="$PWD/.venv/bin:$PATH" \
     SAM3_CHECKPOINT_PATH=$PWD/checkpoints/sam3/sam3.pt \
     HF_HUB_DISABLE_XET=1 CELL_TIMEOUT_S=3600
   rpent --robot robodojo --task put_bottles_into_dustbin --layout 1 \
     --source-root /path/to/RoboDojo \
     --sim-python /path/to/sim-env/bin/python \
     --pi05-python /path/to/pi05-env/bin/python \
     --cuda-device 0 --planner codex --model deepseek-v4-flash --max-turns 30

Output (reward-details audit, three-camera mp4s, transcript) is written to
``logs/<timestamp>_robodojo_<task>_l<layout>/``.

See :doc:`installation` for source, asset, and runtime configuration.

Tools and information access
----------------------------

The planner supplies tools directly; call their listed names. Dustbin placement
and bottle-specific scoring guidance are included only in the
``put_bottles_into_dustbin`` task context, not the generic system prompt.
Recorded-state reading and calibrated depth projection use shared perception
helpers; control and dual-arm monitoring remain backend-specific.

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
at the integer box center, then the action. Repeat that perception pair before
each action. Export the trace before evaluation:

.. code-block:: bash

   python -m robots.robodojo.flash.generate \
     --trace /path/to/dev-run/flash_trace.json \
     --task put_bottles_into_dustbin \
     --destination /path/to/memory/robodojo/flash

The version-1 JSON contains a task, symbolic SAM3 queries, optional wrist-camera
refinement, and ordered actions with arguments and three-dimensional offsets.
It contains no reference object coordinates, score or predicate results.
Export accepts ``move_to``, ``set_gripper`` and ``pi0_pick``; unsupported actions
(including ``place_in_bin`` and ``stabilize``), failed calls and unanchored moves
are rejected. Record those operations as supported basic actions instead.
Existing plans are never overwritten. Review and freeze the plan before eval;
no candidate selection or plan writing occurs during replay.

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
