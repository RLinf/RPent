YAM
===

.. Product image: https://i2rt.com/products/yam-6-dof-arm

.. figure:: https://i2rt.com/cdn/shop/files/st0_768396f1-edfb-4839-96c1-f7b5dffa214a.png?v=1788854436&width=1200
   :alt: Full view of the YAM robot arm, including its base, joints, and gripper
   :figclass: rpent-robot-figure
   :align: center

   YAM arm for real-world experiments.

YAM connects a real dual-arm robot through the public RPent robot registry,
Toolkit, RPC services, exploration lifecycle and MemoryManager. Dual-arm motion
names follow RoboTwin; exploration follows LIBERO. Task 103 is an operator
primitive console; task 104 separates policy prediction from execution.

Dependencies and installation
-------------------------------

Use Linux and Python 3.11. The Agent also needs ``RPent[yam]``: projection
uses OmegaConf and RealSense SDK calculations without connecting cameras.
The control and GPU machines use the following fixed RLinf YAM revision,
which supplies the cancellable runtime, kinematics and ``pi05_yam_joint``
transforms for the official ``openpi.get_model`` loader.

.. code-block:: bash

   git clone https://github.com/scilwb/RLinf.git /path/to/RLinf
   git -C /path/to/RLinf checkout --detach 4c65548e7ade32b13ae101211f588a34a9f98305
   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   export RPENT_REPO_ROOT="$PWD"
   export RPENT_RLINF_ROOT=/path/to/RLinf

On the control machine, install the pinned i2rt SDK and its compatible
``ruckig`` build backend together with RPent and RLinf:

.. code-block:: bash

   uv pip install --torch-backend cpu \
     --build-constraints "$RPENT_RLINF_ROOT/requirements/embodied/envs/yam-build-constraints.txt" \
     -e '.[yam]' -e "$RPENT_RLINF_ROOT" \
     -r "$RPENT_RLINF_ROOT/requirements/embodied/envs/yam.txt"
   uv pip check

On the GPU machine, use a separate environment and install the fixed OpenPI
revision with CUDA 12.8 Torch:

.. code-block:: bash

   uv pip install --torch-backend cu128 \
     -e '.[yam]' -e "$RPENT_RLINF_ROOT[embodied]" \
     'rpent-openpi @ git+https://github.com/RLinf/openpi.git@a560f4dd8205b8423ecd4c8a0fabb5f54140b8a0' \
     -r "$RPENT_RLINF_ROOT/requirements/embodied/models/openpi.txt" \
     'torch==2.7.1' 'torchvision==0.22.1' 'torchcodec==0.5' \
     'tokenizers==0.22.2' 'numpy==1.26.4' 'opencv-python==4.11.0.86'
   uv pip check

The ``yam`` extra keeps ``huggingface-hub<1`` because i2rt 1.1.2 requires
``click<8.2``. Agent-only installations can use ``uv pip install -e '.[yam]'``.
Calibration, i2rt model meshes, a trained checkpoint and norm stats must be
provided separately; RPent does not include these assets.

Robot and task configuration
----------------------------

Copy ``robots/yam/config/example.yaml`` outside versioned source and fill the
camera serials, calibration file, table model, confirmed reset/home poses and
robot control settings. The example cannot start until ``park_on_close`` is
enabled with measured home joints. Do not copy another station's joint poses.
Each pose needs ``left_qpos`` and ``right_qpos`` (seven values each),
``duration_s``, ``max_joint_delta``, ``tolerance`` and ``timeout_s``.
Preserve validated servo and collision settings when upgrading this adapter.

Calibration supplies a shared ``left_base`` frame and the right-base transform.
``top`` is fixed; ``left`` and ``right`` are wrist cameras. RGBD and wrist FK
must refer to the same capture. The example uses 640x480 at 30 Hz for all three
streams. Camera pipelines warm up before arm connection.

Task instructions are registered in ``robots/yam/tasks.py``. Pass ``task_name``,
``seed`` and ``max_episode_steps`` as CLI arguments. ENV and Agent both default
to the registered ``TASK_INSTRUCTIONS`` entry. ``--task-language`` can override
it for a terminal run; Dashboard requires the registered instruction for the
selected task. Stop the existing ENV safely before starting a different task.

In each shell used below, set ``TASK_NAME`` to the selected registered task:

.. code-block:: bash

   export TASK_NAME='your-registered-task'

Independent services
----------------------

Start ENV from the control machine's prepared environment:

.. code-block:: bash

   python -m robots.yam.env_server --robot-config /path/to/robot.yaml \
     --task-name "$TASK_NAME" \
     --seed 0 --max-episode-steps 1000 \
     --transport socket --host 127.0.0.1 --port 8110

The process initially serves without connecting motors. The operator's first
``status`` command initializes cameras and arms; this is an explicit hardware
startup, not a passive connection test. Arrange the station before issuing it.
Use ``env.is_started`` for a passive programmatic startup check.

.. code-block:: bash

   python -m robots.yam.operator_control --robot-config /path/to/robot.yaml \
     --endpoint socket://127.0.0.1:8110 --event status
   python -m robots.yam.operator_control --robot-config /path/to/robot.yaml \
     --endpoint socket://127.0.0.1:8110 --event reset_pose
   # Copy the CURRENT episode_id printed by status; scene ready is an operator fact.
   python -m robots.yam.operator_control --robot-config /path/to/robot.yaml \
     --endpoint socket://127.0.0.1:8110 --episode-id CURRENT_ID \
     --event ready --note 'Scene restored; ready for this attempt'

On the GPU machine, activate the environment installed above. The shared model
server uses the official unified RLinf OpenPI loader:

.. code-block:: bash

   export RPENT_RLINF_ROOT=/path/to/RLinf
   python -m rpent.robots.components.pi05_vla_server \
     --embodiment yam \
     --model-path /path/to/yam-checkpoint \
     --norm-stats-path /path/to/norm_stats.json \
     --transport socket --host 127.0.0.1 --port 8220

Omit ``--norm-stats-path`` only if the checkpoint supplies the expected stats.
Forward the GPU service to the Agent/control machine over SSH, or bind it to a
trusted private interface and supply that endpoint. Pickle RPC must not be
exposed to untrusted clients. Policy output is 30 absolute qpos14 targets; each
call executes exactly the first five. Grippers use 0=closed, 1=open. State layout
is left six joints, left gripper, right six joints, right gripper. RGB order is
``top/left/right``. Metadata and action validation reject incompatible services.

Dashboard and terminal exploration
------------------------------------

Choose an available planner model and adjust the endpoints and paths. Replace
``MODEL_ID`` below with a model supported by your configured Codex backend; use
a reasoning setting supported by that model.

.. code-block:: bash

   rpent --robot yam --dashboard --explore --planner codex \
     --model MODEL_ID --reasoning-effort low \
     --robot-config /path/to/robot.yaml \
     --env-endpoint socket://127.0.0.1:8110 \
     --vla-endpoint socket://127.0.0.1:8220 \
     --memory-profile local --memory-dir /path/to/memory/yam \
     --max-episode-steps 1000 \
     --dashboard-host 127.0.0.1 --dashboard-port 8090

``max-episode-steps`` must match ENV. On the browser computer:

.. code-block:: bash

   ssh -N -L 8090:127.0.0.1:8090 USER@CONTROL_HOST

Open ``http://127.0.0.1:8090`` and submit ``/rpent-task <task-name> 0``, replacing
``<task-name>`` with the same registered task used by ENV.
Browsing does not start hardware. ENV is checked for every task; VLA is shared.
Dashboard owns neither service. Manual primitives and Agent calls are serialized;
after manual action, its result accompanies the next Agent message. ``/continue``
lets the Agent read a new operator ready or verdict receipt, or resume a ready
nonterminal episode without a stop. A ready or verdict receipt is never motion
permission by itself. While the Agent is waiting for the operator, program
generated continuation stays paused until an explicit ``/continue`` or a new
episode. It cannot bypass an explicit finish or an outstanding manual action.

For the terminal, omit Dashboard flags and add ``--task-name "$TASK_NAME"``
to the same command. Use ``--without-vla`` instead of ``--vla-endpoint`` for
geometric tools only.

Verdicts, memory and shutdown
-------------------------------

Only current-episode ``eval_success=True`` establishes task success. Use the
operator terminal above with ``--event success``, ``failure`` or ``abort`` and
current episode ID plus evidence note. ``--command /done`` aliases ``ready``;
``/success``, ``/failure`` and ``/abort`` alias the corresponding events.
Dashboard directs these verdict commands to the operator terminal, not the LLM.
``reset`` consumes readiness and starts an episode; it does not move to home or
restore the physical scene. Missing readiness or verdict returns nonterminal
``pending`` immediately; only an established new episode increments the attempt count.

Explore uses the shared ``_internal/inbox``, ``task-specific``, ``task-family`` and ``global``
memory. Normally completed unsuccessful runs may contribute failure notes.
Runtime errors and Dashboard task replacement leave the inbox unpublished.
Success recipes retain issued actions from the verified successful episode,
including partial execution; execution details remain in the step records. See
``result.json``, transcripts, session step records, recipe/audit and the updated
``MEMORY.md`` for distinct evidence. A planner exit,
an interface test or a successful primitive does not prove full task success.

Stopping/closing Dashboard requests hold only. After handling any held object
and checking the home path, shut down ENV from the operator terminal:

.. code-block:: bash

   python -m robots.yam.operator_control --robot-config /path/to/robot.yaml \
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

   python -m robots.yam.manual --task-name "$TASK_NAME" --task-id 103 \
     --env-endpoint socket://127.0.0.1:8110 --max-episode-steps 1000
   python -m robots.yam.manual --task-name "$TASK_NAME" --task-id 104 \
     --env-endpoint socket://127.0.0.1:8110 --max-episode-steps 1000 \
     --vla-endpoint socket://127.0.0.1:8220 --output-dir /path/to/diagnostics

Enter JSON lines: ``{"tool":"status"}``; task 103 allows ``move_to``,
``rotate_wrist``, ``set_gripper``, ``release`` with ``arguments``. Task 104 uses
``{"tool":"infer"}`` (saves prediction without moving), then
``{"tool":"execute","arguments":{"use_length":5}}`` at most once.
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

The ``--robot-config`` entry point accepts YAML. Convert older JSON configuration
before upgrading. Source and wheel installs include robot modules; set
``RPENT_REPO_ROOT`` to the intended workspace for logs, guides and memory.
