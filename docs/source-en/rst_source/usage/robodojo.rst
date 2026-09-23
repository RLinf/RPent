RoboDojo
========

RoboDojo connects Isaac Sim / IsaacLab, dual ARX-X5 arms and an RLinf Pi0.5
policy to RPent's planners, tools and memory. The integration is experimental;
simulator compatibility and task success require GPU validation.
Follow the upstream RoboDojo and XPolicyLab instructions for simulator, CUDA
and policy dependencies, and download the required assets and checkpoints.
For shared backend interfaces, see :doc:`../development/add_robot`.

Python environment
------------------

Install the runtime and agent dependencies from the RPent root in Python 3.11.
The ``robodojo-sim`` extra installs ``rlinf-robodojo-runtime`` and the
RLinf environment adapter; the runtime owns the Isaac Sim / IsaacLab pins.
``robodojo`` also installs SAM3 and the openpi policy runtime.
Use one robot extra per environment. Run uv from this directory so it reads
the root project's packaging overrides and cuRobo build dependencies:

.. code-block:: bash

   uv pip install -e ".[robodojo]" --extra-index-url https://pypi.nvidia.com

These overrides allow dependency resolution; they do not establish simulation
task success. The RLinf policy path has had an end-to-end policy-chain smoke
check with real weights and one prediction, but no simulation task success
rate has been established. Its action chunk remains 50, matching the training
configuration's ``Pi0Config.action_horizon``.

The simulation bridge requires runtime version 0.3.0 or later.
The Git references are temporary until versioned releases are published.
Fresh dependency installation and GPU rollout must be validated separately;
offline bridge tests do not establish simulator compatibility.
IsaacLab's upstream non-editable packaging can omit its extension configuration.
If that defect affects the installed revision, use a separately prepared
editable IsaacLab installation; the runtime wheel does not repair IsaacLab.

Sources and assets
------------------

The runtime wheel includes the validated RoboDojo code tree, so a RoboDojo
checkout is not required. By default, the environment uses
``robodojo_runtime.source_root()``; ``--source-root`` can override it
with a local checkout. Scene data remains separate from Python dependencies.
Set ``ROBODOJO_ASSETS_ROOT`` to the directory containing ``Assets/``:

* **RoboDojo robot/object/material/layout assets:** download only ``Assets/**``
  from `RoboDojo's dataset <https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo>`_.
  For example, with the Hugging Face CLI available:

  .. code-block:: bash

     hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
       --include 'Assets/**' --local-dir /data/robodojo
     export ROBODOJO_ASSETS_ROOT=/data/robodojo

  The packaged runtime reads ``ROBODOJO_ASSETS_ROOT/Assets``; no symlink
  into the installed code is needed. Check that ``Robots``, ``Object``,
  ``Material`` and ``Eval_Layout`` contain real files, not LFS pointers.
* **NVIDIA USD/material assets referenced by IsaacLab:** these are separate
  from ``isaaclab_assets``. Obtain the matching asset pack from
  `NVIDIA's asset download instructions <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_faq.html#isaac-sim-setup-assets-content-pack>`_.
  RoboDojo's ``utils/ensure_usd_path.py`` specifically rewrites URLs under
  ``Assets/Isaac/5.0``; for those references, download the **5.0** asset pack,
  preserve ``/data/nvidia/Assets/Isaac/5.0`` and export:

  .. code-block:: bash

     export ROBODOJO_USD_ASSET_PREFIX=/data/nvidia

  This existing upstream variable applies only to that URL prefix, not every
  IsaacLab asset or a 5.1 asset tree. Other IsaacLab references use Kit's
  ``/persistent/isaac/asset_root/cloud`` setting; configure that separately
  for a matching local pack when offline. Do not relabel a 5.1 pack as 5.0.

The two similarly named dependency entries are **not** two scene-data bundles:
``isaacsim[all,extscache]`` supplies simulator binaries and extension caches
(the Linux x86-64 / CPython 3.11 5.1.0 Kit and Kit-SDK cache wheels alone are
3,021,340,845 and 1,345,115,764 bytes). Keep this runtime installation; a data
environment variable cannot replace it. ``isaaclab_assets`` at ``afca7b09``
contains 119,143 bytes of source/configuration and no USD files, so retain it
too. Git tree sizes are about 53.6 MB for IsaacLab and 129.9 MB for cuRobo
(including robot meshes); these are uncompressed tree totals, not measured
clone sizes. cuRobo's packaged meshes remain part of its runtime.
Thus uv still downloads a large simulator runtime, but not the separate scene
datasets or policy checkpoints. Download checkpoints separately and use
``PI05_CHECKPOINT_PATH`` and ``SAM3_CHECKPOINT_PATH`` as described below.

RPent configuration
-------------------

The default ``--policy-backend rlinf`` requires a policy interpreter that
provides the ``pi05_robodojo_arx_x5`` config and its openpi dependencies;
RLinf main does not include that config yet. Set ``PI05_CHECKPOINT_PATH`` to a
compatible RLinf checkpoint. The ``robodojo`` extra supplies this policy runtime.
Use ``--policy-backend xpolicylab`` for an independently prepared XPolicyLab runtime.

Configure SAM3's checkpoint using ``SAM3_CHECKPOINT_PATH``, and export the
placement settling budget. The default leaves objects unstable in official
mode; the variable is read by the RoboDojo checkout, not by RPent, and the CLI
passes it on to the child services it starts:

.. code-block:: bash

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000

Start with the packaged code:

.. code-block:: bash

   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0

XPolicyLab is not included in the runtime wheel. ``--xpolicylab-root`` defaults
to ``SOURCE_ROOT/XPolicyLab``; set it for
a separate checkout. In a single environment, services default to the current
interpreter; use ``--sim-python`` or ``--pi05-python`` to override it when needed.
The CLI constructs
child import paths without reading a workspace's ``config/runtime.env`` or
changing the parent environment. Child processes inherit shell environment
variables. Omitting
``--cuda-device`` preserves ``CUDA_VISIBLE_DEVICES``, including an unset value;
passing it explicitly selects the GPU for locally started services.
For XPolicyLab, an omitted GPU flag starts its Python policy entry point with
``--pi05-python`` directly; an explicit flag uses its shell launcher.

Use ``--env-endpoint``, ``--vla-endpoint``, and ``--sam3-endpoint`` to attach
to already running services. A borrowed service requires no local source or
Python path for that component. The CLI starts the shared
``rpent.robots.components.pi05_vla_server --embodiment robodojo`` by default.
With ``--policy-backend xpolicylab``, it starts ``xpolicylab_vla_server`` with
``--policy-root`` pointing to ``XPolicyLab/policy/Pi_05`` instead.
Select the matching backend when borrowing a VLA endpoint. Changing backend
does not convert checkpoints; the RLinf client encodes native observations
into openpi's wire format.

Every owned service logs and writes into the run's output directory: the CLI
passes it as ``--save-dir`` to the environment server and as ``--output-dir``
to the optional XPolicyLab entry point. Its inner policy log is ``vla_server.log``.
Direct service launches default to the current directory; use separate output
directories for concurrent runs.

Verify the installation
-----------------------

Run one bounded development episode and confirm the services come up before the
planner takes over:

.. code-block:: bash

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000
   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0 \
     --planner codex --model <planner-model> --max-turns 1 \
     --sim-python /path/to/sim-env/bin/python \
     --pi05-python /path/to/pi05-env/bin/python \
     --output-dir /path/to/run-output

Expected behaviour:

* ``/path/to/run-output`` contains ``robodojo_env_server.log``,
  ``sam3_server.log``, ``robodojo_vla_server.log`` and, once the policy server is
  spawned with XPolicyLab, ``vla_server.log``.
* The environment server reports ready, and the first observation carries
  ``cam_head``, ``cam_left_wrist`` and ``cam_right_wrist`` with intrinsics and
  extrinsics, plus joint and gripper state.
* The run writes one MP4 per camera under ``/path/to/run-output/videos``.
* Shutdown leaves no owned child process behind and the GPUs return to idle.

A service that exits during startup is the usual failure mode; read its log in
the run output directory first. Isaac Sim start-up takes tens of seconds and the
first run also compiles shaders.

Key modules
-----------

* ``robots/robodojo/rlinf_env.py`` — agent recording, camera metadata and episode
  diagnostics over RLinf's ``RoboDojoEnv``.
* ``robodojo_runtime/bridge.py`` — simulator creation, reset, observations and
  control execution; imported lazily by RLinf.
* ``robots/robodojo/env_server.py`` — Isaac Sim RPC server (main-thread
  rendering; head + dual-wrist RGB-D with intrinsics/extrinsics; joint/ee
  actions; per-camera video recording).
* ``robots/robodojo/env_client.py`` — rpent-side client inheriting
  ``BaseEnvClient``.
* Shared ``rpent/robots/components/pi05_vla_server.py`` and
  ``rpent/robots/components/pi05_vla_client.py`` — default RLinf openpi policy
  with the ``robodojo`` embodiment.
* Optional ``rpent/robots/components/xpolicylab_vla_server.py`` and
  ``rpent/robots/components/xpolicylab_vla_client.py`` — Pi_05 policy
  service (XPolicyLab WebSocket) adapted to the shared ``BaseVLAFacade`` /
  ``BaseVLAClient`` protocol.
* ``robots/robodojo/toolkit.py`` / ``tools.py`` — primitives:
  ``view_env_state``, ``back_project``, ``segment``, ``move_to``,
  ``set_gripper``, ``pi0_pick``, ``stabilize``, ``place_in_bin``.
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

The RLinf client maps head, left-wrist and right-wrist RGB to ``main_images``,
``wrist_images`` and ``extra_view_images``. ``states`` contains left arm (6),
right arm (6), left gripper (1), right gripper (1), with observed gripper
values unchanged (1=open, 0=closed); ``task_descriptions`` carries the instruction.

The XPolicyLab adapter passes observations/actions through unchanged and
serializes ``update_obs``/``get_action`` with ``reset``; it does not isolate
sessions. RoboDojo requires three-camera inputs and 14-DoF joint actions;
neither backend converts checkpoints or joint actions to end-effector actions.

Tools and information access
----------------------------

The planner supplies tools directly; call their listed names. Dustbin placement
and bottle recovery guidance are included only in the
``put_bottles_into_dustbin`` task context, not the generic system prompt.
Recorded-state reading and calibrated depth projection are local to
``robots/robodojo/tools.py``. Projection uses Isaac's negative optical Z axis;
segmentation uses the shared SAM3 client. ``view_env_state``, ``back_project`` and
``segment`` are read-only:
they do not advance the environment or trigger post-action state capture.

``robots.robodojo.tools.TOOL_GROUPS`` marks direct outputs as ``general``
(depth/segmentation and motion) or ``mixed`` (``view_env_state``,
``set_gripper``, ``place_in_bin``). Neither group exposes reward details or
ground-truth safety alarms. Planners judge progress from observations only.
The underlying ``env.get_reward_details`` and ``env.get_safety_status`` RPCs
remain available on dev servers for external evaluation and diagnostics, not
as planner tools. Reward and official success belong to the evaluation path
after the planner's action channel is closed. ``finalize_run`` records the
runner-provided result; it does not call these RPCs itself.

The Python toolkit factory accepts ``allowed_tool_groups``; for example,
``frozenset({"general"})`` registers only general robot tools. The default
``None`` preserves the existing tool set. This hook filters schemas and
handlers, not automatic post-action state, raw observation fields, logs,
memory, or common file tools. It is not an evaluation isolation mode.

Development and frozen replay
-----------------------------

Normal planners retain the development tool set without scoring or safety diagnostics.
``--planner flash`` selects eval-fair: RPent's native Flash planner invokes
``RobotSpec.run_flash`` without an LLM. ``--explore`` remains unsupported for
RoboDojo; development here means the normal planner loop, not that CLI mode.

Development records actions and observations in the shared ``states.json``
manifest. Read-only perception results are attached to the next action for
Flash export. Existing ``flash_trace.json`` exports remain readable. To record
a transferable waypoint, call ``segment`` on ``cam_head``, then ``back_project``
at its returned ``centroid_rc`` (row, col), then the action. The pixel is the
floored mean of foreground mask coordinates, not the box center. Recording
and replay share this derivation; recorded mask area and coordinate sums
allow export to reject a changed centroid. Do not batch different objects'
segmentations ahead of depth calls. Repeat that perception pair before
each action. Export the trace before evaluation:

.. code-block:: bash

   python -m robots.robodojo.flash.generate \
     --trace /path/to/dev-run/states.json \
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

Eval-fair additionally excludes common file/memory tools and task-specific helpers.
The service advertises the mode in its metadata, rejects reset and diagnostic RPCs, returns
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

Task-language RPC and public observations use RoboDojo's initialized description
manager. Empty language or unresolved template markers raise an error. The
official instruction stays public in eval-fair.
Omitting ``pi0_pick.prompt`` uses this resolved official language. Explicit
overrides remain supported for contact segments but must identify the intended
object; unresolved markers are rejected before inference. Official multi-object
tasks do not specify a grasp order, and descriptive overrides do not guarantee
that a checkpoint can select arbitrary instances. Verify the actual held target.

Capability scope and limitations
--------------------------------

RoboDojo provides dual-arm motion and gripper primitives, three-camera RGB-D,
SAM3 perception, RLinf Pi0.5 (or optional XPolicyLab) and frozen Flash replay. Task names come from
the configured checkout; examples include ``put_bottles_into_dustbin``,
``fill_pen_holder`` and ``stack_bowls_random``, not a validated success suite.
``place_in_bin`` is registered only for ``put_bottles_into_dustbin``.
Handover is not implemented. Low-Z tabletop and lateral scripted IK motions
have reachability limits: inspect ``reached`` and ``dist_to_target`` rather
than assuming the commanded pose was achieved. See :doc:`flash` for the
shared evaluation-only planner; replay executes actions, it is not a
read-only robot operation.

Bounded smoke runs and shutdown diagnostics
--------------------------------------------

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
