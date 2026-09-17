Adding a Robot Backend
======================

.. toctree::
   :maxdepth: 1
   :hidden:

   installation

Use RoboDojo as a worked example of adding a backend under ``robots/<name>/``.
It connects Isaac Sim / IsaacLab, dual ARX-X5 arms and an XPolicyLab Pi_05
policy to RPent's shared planner, perception, tool and memory infrastructure.
The integration is experimental: offline contracts do not establish simulator
compatibility or task success. For installation and a runnable CLI example,
see :doc:`installation`.

Start with :doc:`../../development/add_robot` and
:doc:`../../development/add_primitive` for the shared interfaces. Compare
:doc:`../libero`, :doc:`../robocasa` and :doc:`../robotwin` for simulation,
or :doc:`../dual_franka` for hardware provisioning and safety requirements.

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

1. Register the backend and own its runtime
------------------------------------------------------------

Export ``get_robot_spec`` and ``get_toolkit`` from ``robots/<name>/__init__.py``.
Discovery in ``rpent/robots/base.py`` imports the package lazily; no central
registry entry is needed. Users select it with ``rpent --robot <name>``.

Implement ``RobotSpec`` in ``robot_spec.py``: provide the name and prompts,
register CLI arguments with ``add_cli_args``, build ``RunConfig`` in
``parse_config``, and launch requested components in ``init_runtime``.
Accept ``runtime_kwargs``, ``dashboard_events`` and ``config`` in ``get_toolkit``;
create its ``MemoryManager`` from the configured memory directory. RoboDojo's
factory passes ``runtime_kwargs`` to its internal toolkit as ``primitives_kwargs``.
Use shared
``try_spawn_server``, ``try_wait_server`` and ``ProcessDaemon`` helpers; return
owned processes for cleanup, but do not stop borrowed endpoints. Implement
``run_flash(toolkit, cell_tag, note)`` only when frozen replay is supported.

Keep simulator and model imports at their use sites. Accept source roots,
Python executables and endpoints explicitly through CLI/configuration; do not
read a developer's workspace file or hard-code a local path. RoboDojo's
``--source-root``, ``--sim-python`` and ``--pi05-python`` illustrate this split.

2. Define the environment contract
----------------------------------

Build ``env_client.py`` on ``BaseEnvClient`` and ``env_server.py`` on
``BaseEnvFacade``. Reuse RPC routing, metadata validation and process lifecycle.
Specify observation keys, action units, reset ownership and return shapes in
tests against both callers. ``BaseEnvClient`` caches step observations but does
not normalize tuple arity: its general documentation describes a five-item
step result, whereas RoboDojo's consumers use ``(obs, reward, done, info)``.
RoboDojo reset returns an observation dictionary; ``chunk_step`` raises
``NotImplementedError``, and its primitives issue individual steps. Do not
copy these backend-specific shapes into a different consumer unchanged.

For renderers requiring the main thread, follow RoboDojo's
``MainThreadServeMixin`` before ``BaseEnvFacade`` in the inheritance order.
Initialize Isaac before simulator imports and dispatch simulator work to the
main thread rather than running it in RPC worker threads. Test reset-on-connect
and mode metadata explicitly; eval must not silently attach to a dev service.

3. Assemble tools, prompts and tasks
------------------------------------------------------------

Keep registration and state/artifact handling in ``toolkit.py`` and primitives
in ``tools.py``. Use shared ``perception_tools.py`` and the SAM3 client for
recorded-state access, segmentation and calibrated depth projection. Check
camera conventions: RoboDojo uses negative optical Z, which is not universal.
Keep motion, gripper control and dual-arm monitoring in the owning backend.

Mark non-mutating tools with ``@readonly`` so tool execution does not append
an automatic post-action state capture. Reading cached observations or running
segmentation is not a robot action; RoboDojo marks ``view_env_state``,
``back_project`` and ``segment`` accordingly. This marker does not mean that
the returned information is safe for evaluation; classify outputs separately.

Provide ``prompts`` and ``prompt_bundle`` for the shared prompt builder.
Tools are injected by the planner and called by name: do not instruct agents
to discover MCP URLs or send JSON-RPC requests. Keep task-specific placement
or scoring guidance in task context, not the generic system prompt.
Implement task discovery in ``tasks.py``; RoboDojo reads
``<source_root>/task/RoboDojo/config/*.yml`` excluding ``_task.yml``.
A discovered task is not a validated benchmark result.

4. Reuse the policy service
------------------------------------------------------------

Use ``BaseVLAClient`` and the shared launcher
``python -m rpent.robots.components.pi05_vla_server`` with
``--policy-backend rlinf`` or ``--policy-backend xpolicylab``.
The former loads the RLinf policy in process; the latter adapts an external
XPolicyLab WebSocket service through ``BaseVLAFacade``. Configure the launcher
in ``robot_spec.py`` rather than adding a robot-local server. Backend selection
does not convert checkpoints or observations: RoboDojo requires its 14-DoF
joint actions and three-camera inputs. Preserve ``pi0_pick`` dual-arm monitoring
when adapting the thin client.

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

Backend implementation checklist
--------------------------------

1. Create the package exports and ``RobotSpec``; verify discovery and CLI help
   without simulator/model dependencies. Use
   ``tests/unit_tests/robots/test_robodojo_runtime_contracts.py`` as an example.
2. Test reset, step shapes, unsupported chunk stepping, metadata and owned
   process cleanup with fake clients/facades in the runtime contracts above.
3. Register tool schemas, mark read-only calls and classify information access.
   Cover defaults and filtered groups in
   ``tests/unit_tests/robots/test_tool_schema_contracts.py``; shared perception
   cases live in
   ``tests/unit_tests/rpent/robots/components/test_perception_tools_contracts.py``.
4. Add task context and prompt bundles; test that unrelated tasks receive no
   task-specific instructions. Test policy backend selection using
   ``tests/unit_tests/rpent/robots/components/test_pi05_vla_server_contracts.py``.
5. If implementing Flash, test frozen-plan validation, anchor offsets, bounded
   retries and privilege isolation with fake state/toolkits, following
   ``tests/unit_tests/robots/robodojo/test_flash_contracts.py``. Filtering tool
   names alone is insufficient: audit observations, automatic logs and memory.
6. Add paired pages under ``docs/source-en/rst_source/usage/`` and
   ``docs/source-zh/rst_source/usage/``, navigation and feature-matrix entries in
   both READMEs and overview pages. Run ``pre-commit run --all-files``,
   ``pytest tests/unit_tests -q`` and strict builds via
   ``make -C docs html LANG=en SPHINXOPTS='-W --keep-going -E'`` and ``LANG=zh``.
   Follow ``CONTRIBUTING.md`` and ``tests/README.md`` for dependencies and
   runtime validation; report GPU, real-policy and simulator checks separately.
