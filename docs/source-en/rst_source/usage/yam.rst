YAM
===

YAM connects a real dual-arm robot through the public RPent robot registry,
Toolkit, RPC services, exploration lifecycle and MemoryManager. Dual-arm motion
names follow RoboTwin; exploration follows LIBERO. Task 103 is an operator
primitive console; task 104 separates policy prediction from execution.

Dependencies and installation
-------------------------------

Use Linux and Python 3.11 on the control machine. The Agent needs only RPent;
RealSense, MuJoCo and the station's RLinf/i2rt installation belong on the control
machine. The GPU machine needs the same RLinf YAM policy implementation and its
compatible OpenPI/Torch environment.

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e '.[yam]'
   export RPENT_REPO_ROOT="$PWD"
   export RPENT_RLINF_ROOT=/path/to/station-RLinf

This adapter was developed against an RLinf YAM fork at ``3554fd2c`` **plus
station changes**. Unmodified official RLinf is not a sufficient dependency.
Before deployment retain the fork commit, working patch, untracked runtime
sources and environment lock with the station configuration. Required APIs are
``YamControlRuntime`` (command, hold, move_to, feedback), ``I2RTYamBackendFactory``,
``YamKinematicsAdapter``, and the ``openpi_rlinf`` model loader with
``pi05_yam_joint`` data/policy transforms. RPent does not install or publish
those local changes. Calibration, i2rt model meshes, checkpoint and norm stats
must also be supplied. No trained YAM checkpoint ships with this extension.

Site configuration
--------------------

Copy ``robots/yam/config.example.json`` outside versioned source and fill the
camera serials, calibration file, table model, confirmed reset/home poses and
station control settings. The example cannot start until ``park_on_close`` is
enabled with measured home joints. Do not copy another station's joint poses.
Each pose needs ``left_qpos`` and ``right_qpos`` (seven values each),
``duration_s``, ``max_joint_delta``, ``tolerance`` and ``timeout_s``.
Preserve validated servo and collision settings when upgrading this adapter.

Calibration supplies a shared ``left_base`` frame and the right-base transform.
``top`` is fixed; ``left`` and ``right`` are wrist cameras. RGBD and wrist FK
must refer to the same capture. All three streams use 640x480 at 30 Hz on the
validated station. Camera pipelines warm up before arm connection.

Task instructions are registered in ``robots/yam/tasks.py``. For the tabletop
example, set both ``task_name`` and the exact matching ``task_language`` from
``TASK_INSTRUCTIONS`` in separate site files:

* ``tabletop_cleanup_a``: Pepsi in the left bag, Coca-Cola in the right bag.
* ``tabletop_cleanup_b``: Coca-Cola in the left bag, Pepsi in the right bag.

Both use top-view left/right, sort all three bottles, uncover the spoons, put
white spoon in left white bowl and pink spoon in right pink bowl, and return
bowls to their marked positions. Initial spoon positions may vary. Task names
stay stable across runs; evidence IDs are unique. Dashboard refuses an A/B
language mismatch; switch ENV configuration explicitly after safe shutdown.

Independent services
----------------------

Start ENV from the control machine's prepared environment:

.. code-block:: bash

   python -m robots.yam.env_server --config /path/to/task_a.json \
     --transport socket --host 127.0.0.1 --port 8110

The process initially serves without connecting motors. The operator's first
``status`` command initializes cameras and arms; this is an explicit hardware
startup, not a passive connection test. Arrange the station before issuing it.
Use ``env.is_started`` for a passive programmatic startup check.

.. code-block:: bash

   python -m robots.yam.operator_control --config /path/to/task_a.json \
     --endpoint socket://127.0.0.1:8110 --event status
   python -m robots.yam.operator_control --config /path/to/task_a.json \
     --endpoint socket://127.0.0.1:8110 --event reset_pose
   # Copy the CURRENT episode_id printed by status; scene ready is an operator fact.
   python -m robots.yam.operator_control --config /path/to/task_a.json \
     --endpoint socket://127.0.0.1:8110 --episode-id CURRENT_ID \
     --event ready --note 'Scene restored; ready for this attempt'

On the GPU machine, activate its prepared RLinf policy environment, install
RPent there, and set its own ``RPENT_RLINF_ROOT`` before starting:

.. code-block:: bash

   python -m robots.yam.vla_server --model-path /path/to/yam-checkpoint \
     --norm-stats-path /path/to/norm_stats.json \
     --transport socket --host 127.0.0.1 --port 8220

Omit ``--norm-stats-path`` only if the checkpoint supplies the expected stats.
Forward the GPU service to the Agent/control machine over SSH, or bind it to a
trusted private interface and supply that endpoint. Pickle RPC must not be
exposed to untrusted clients. Policy output is 30 absolute qpos14 targets; only
the execution caller truncates it. Grippers use 0=closed, 1=open. State layout
is left six joints, left gripper, right six joints, right gripper. RGB order is
``top/left/right``. Metadata and action validation reject incompatible services.

Dashboard and terminal exploration
------------------------------------

The following is an **explicit station launch profile**, not library defaults.
Model availability depends on the installed planner/account. Adjust endpoints
and paths on the launching machine. ``low`` is used rather than an unsupported
``none`` reasoning setting for this model.

.. code-block:: bash

   rpent --robot yam --dashboard --explore --planner codex \
     --model gpt-6-astra --reasoning-effort low \
     --env-endpoint socket://127.0.0.1:8110 \
     --vla-endpoint socket://127.0.0.1:8220 \
     --memory-profile local --memory-dir /path/to/memory/yam \
     --max-episode-steps 1800 --explore-attempts-per-session 50 \
     --explore-sessions 1 --max-turns 300 --planner-timeout-s 14400 \
     --dashboard-host 127.0.0.1 --dashboard-port 8090

``max-episode-steps`` must match ENV. On the browser computer:

.. code-block:: bash

   ssh -N -L 8090:127.0.0.1:8090 USER@CONTROL_HOST

Open ``http://127.0.0.1:8090`` and submit ``/rpent-task tabletop_cleanup_a 0``.
For B, prepare the B service/config and submit ``/rpent-task tabletop_cleanup_b 0``.
Browsing does not start hardware. ENV is checked for every task; VLA is shared.
Dashboard owns neither service. Manual primitives and Agent calls are serialized;
after manual action, its result accompanies the next Agent message. ``/continue``
resumes only a ready nonterminal episode without a stop. Automatic continuation
has the same checks and a bounded no-action loop. It cannot bypass an explicit
finish, operator interruption, or an outstanding manual action.

For the terminal, omit Dashboard flags and add ``--task-name tabletop_cleanup_a``
to the same command. Use ``--without-vla`` instead of ``--vla-endpoint`` for
geometric tools only. To reproduce the station's policy preference, send the
Agent: ``Use chunks=2, use_length=30 when the scene supports it; inspect after
each call.`` This is two predictions / 60 requested steps, not two action steps.
A longer unreviewed prediction horizon may include an early release.

Verdicts, memory and shutdown
-------------------------------

Only current-episode ``eval_success=True`` establishes task success. Use the
operator terminal above with ``--event success``, ``failure`` or ``abort`` and
current episode ID plus evidence note. ``--command /done`` aliases ``ready``;
``/success``, ``/failure`` and ``/abort`` alias the corresponding events.
Dashboard directs these verdict commands to the operator terminal, not the LLM.
``reset`` consumes readiness and starts an episode; it does not move to home or
restore the physical scene. Waiting returns nonterminal ``pending`` after at
most 20 seconds; only an established new episode increments the attempt count.

Explore uses official ``_internal/inbox``, ``task_only``, ``suite`` and ``global``
memory. Valid failure notes can merge after an unsuccessful run; success
recipes contain only actions from the verified successful episode. A/B bag rules
remain task-specific. See ``result.json``, transcripts, session step records,
recipe/audit and the updated ``MEMORY.md`` for distinct evidence. A planner exit,
an interface test or a successful primitive does not prove full task success.

Stopping/closing Dashboard requests hold only. After handling any held object
and checking the home path, shut down ENV from the operator terminal:

.. code-block:: bash

   python -m robots.yam.operator_control --config /path/to/task_a.json \
     --endpoint socket://127.0.0.1:8110 --event shutdown

ENV moves to configured home, verifies convergence, then disables output.
Failed home keeps the runtime available for operator recovery. Abrupt process
termination or power loss cannot guarantee this sequence.

Diagnostic tasks and interface migration
------------------------------------------

Stop the Dashboard Agent before opening either console. ENV must already be
started. For diagnostics, use operator ``--event start`` to consume readiness
before explicit moves; diagnostics never create readiness themselves.

.. code-block:: bash

   rpent --robot yam --task-name tabletop_cleanup_a --task-id 103 \
     --env-endpoint socket://127.0.0.1:8110 --max-episode-steps 1800
   rpent --robot yam --task-name tabletop_cleanup_a --task-id 104 \
     --env-endpoint socket://127.0.0.1:8110 --max-episode-steps 1800 \
     --vla-endpoint socket://127.0.0.1:8220 --output-dir /path/to/diagnostics

Enter JSON lines: ``{"tool":"status"}``; task 103 allows ``move_to``,
``rotate_wrist``, ``set_gripper``, ``release`` with ``arguments``. Task 104 uses
``{"tool":"infer"}`` (saves prediction without moving), then
``{"tool":"execute","arguments":{"use_length":30}}`` at most once.
Changed episode, action count, joint pose or a prediction older than 30 seconds
is rejected. An uncertain RPC outcome cannot be replayed. ``quit`` holds;
it does not release, home or create success memory.

Replace all old ``exploration_status`` calls with ``status()``. The return has
native episode fields plus ``reason`` and ``can_continue`` at the top level.
No permanent alias is provided. ``move_to(arm, xyz, quat, gripper, substeps)``
retains ``xyz_bounds`` for an observed task-valid region. A finite candidate
search is used, not all points in a continuous radius. ``substeps`` never removes
required path samples. ``rotate_wrist`` accepts ``gripper``; use
``set_gripper(arm, val, steps)`` and ``release(arm, val=1, steps)`` instead of
open/close aliases. All xyz are metres in left_base; quaternions are wxyz.

During upgrade preserve site JSON, calibration, weights, datasets and memory;
replace code only in a maintenance window. Update launch flags explicitly and
reinstall the package. Keep the old Git bundle/patches for rollback. Source and
wheel installs include robot modules; set ``RPENT_REPO_ROOT`` to the intended
checkout to give logs, guides and memory a stable workspace location. No new
VLA training, HF upload or live exploration is performed by installation.
