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
