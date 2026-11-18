Adaptive Robot Workflows
========================

Run a VLM through end-effector and perception tools with ``--robot rl``.
LIBERO, RoboCasa and RoboTwin share the episode planner; LIBERO also supports
the experimental zap_turn, zap_episode and zap_subtask modes. Native simulator success determines the
outcome. No VLA weights or task-recipe memory are required.

.. list-table:: Supported combinations
   :header-rows: 1

   * - Simulator
     - Episode
     - ZAP modes
   * - ``libero`` (including LIBERO-PRO)
     - Supported
     - Experimental
   * - ``robocasa``
     - Supported
     - Unsupported
   * - ``robotwin``
     - Supported
     - Unsupported

Installation and Resources
--------------------------

The :download:`installation guide <../../../../tools/rl/INSTALL.md>` starts
from a fresh checkout and covers system libraries, isolated Python environments,
models, SAM3, simulator assets and troubleshooting. It uses your checkout and
data directory, with no cluster-specific paths. Copy
``tools/rl/config.example.sh`` to ``.rl-env.sh`` and customize its settings.

The common environment runs the planner, SAM3 and LIBERO. RoboCasa and RoboTwin
use separate simulator environments; the model server has its own environment.
Linux NVIDIA EGL is required; RoboTwin also requires Vulkan and a CUDA toolkit
for cuRobo. Install assets and verify imports before running episodes.

Run a Task
----------

The :download:`first-task guide <../../../../tools/rl/README.md>` provides
complete service startup and four commands: LIBERO episode, LIBERO ZAP,
RoboCasa episode and RoboTwin episode. Start with one episode.

After configuring and starting the model and shared SAM3 services, from the
checkout root:

.. code-block:: bash

   source .venv/bin/activate
   source .rl-env.sh
   source tools/rl/env.sh
   bash tools/rl/run.sh --simulator libero --workflow episode \
     --suite libero_object --task 0 --seed 1 --cuda-device 1

``--cuda-device`` selects the simulator GPU independently of model/SAM3 GPUs.
RoboCasa uses ``--task-name`` and ``--split``; RoboTwin uses ``--task-name``,
``--task-config`` and a published valid seed. The first-task guide contains
working examples and links to task manifests.

Workflow and Tools
------------------

All modes use the API planner. Select the LIBERO frame with ``--world-cord``.

.. list-table:: LIBERO context policies
   :header-rows: 1

   * - Workflow
     - Perception and plan
     - Conversation
   * - ``episode``
     - ``view_env``, ``segment``, ``back_project``
     - Entire episode
   * - ``pear``
     - RGB and measured robot state; one ``move_to`` or ``set_gripper``
     - Entire episode; native automatic completion
   * - ``zap_turn``
     - Automatic perception; text todo
     - Independent request snapshots
   * - ``zap_episode``
     - Automatic perception; no todo
     - Entire episode
   * - ``zap_subtask``
     - Automatic perception; structured todo
     - Cleared by every valid ``update_todo``

The existing episode and ZAP modes expose ``move_to``, ``set_gripper`` and ``finish``. Automatic
perception uses current SAM3 boxes and depth-measured world geometry.
Each reply has one tool call; subtask initialization instead requires the
query and todo tools together. Context resets preserve the global budget.
Native task text comes directly from the simulator, without task prompt files.
RoboCasa adds ``move_base`` and RoboTwin anatomical arm selection and paired
commands. Both use ``episode``.

The :download:`workflow guide <../../../../tools/rl/WORKFLOWS.md>` gives exact
bootstrap/reset rules, complete tool lists, SAM3 pool setup and fixed-panel
comparison commands. ``--image_size`` selects 1024, 512 or 256 for VLM images;
native RGB-D/SAM3 remain at 1024 pixels and displayed pixel/box coordinates are mapped.

Planner Configuration
---------------------

Edit the commented ``run_args`` array in ``tools/rl/run.sh`` to set defaults
for single episodes and batches. Each parameter occupies one line. Explicit
command-line arguments override ``RPENT_*`` environment values, which override
the defaults in that file. Inspect the effective default argv without starting
a model or simulator:

.. code-block:: bash

   bash tools/rl/run.sh --print-defaults

Both ``run.sh`` and ``python -m tools.rl.run_batch run`` accept the following
options. Batch collection resolves defaults from the same script, passes the
resolved values to every child, and pins them in ``manifest.json``; changing
them requires a new output directory.

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``--max-tokens``
     - 32768
     - Per-response completion budget, including reasoning and tool protocol.
   * - ``--context-window``
     - 262144
     - Served context capacity used for counting; does not resize the server.
   * - ``--compaction-fraction``
     - 0.7
     - Trigger when current input exceeds this fraction of the context.
   * - ``--compaction-max-messages``
     - 80
     - Independent message-count trigger; 0 disables this trigger.
   * - ``--compaction-keep-messages``
     - 10
     - Tail messages to retain, plus opening/pinned context and intact tool pairs.
   * - ``--max-turns`` / ``--max-episode-steps``
     - 80 / 10000
     - Model-request / physical-step budgets per trajectory.
   * - ``--planner-timeout-s``
     - 7200
     - Planner wall-clock limit; batch ``--episode-timeout`` defaults to 14400.

A normal reply and its tool result are two messages; 10 tail messages usually
cover about 5 interactions. Token and message-count triggers use OR. For a
token-only trigger, pass ``--compaction-max-messages 0``. A retained window may
still exceed the token threshold; the audit records that condition.

Sampling defaults match Qwen3.8-27B thinking mode:
``--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repetition-penalty 1.0``.
All six values are explicit in model requests; ``--top-k -1`` disables top-k.
``--thinking`` is the default; ``--no-thinking`` disables thinking without
changing sampling. The 27B launcher uses ``--reasoning-effort xhigh``; use
``low`` or ``medium`` for other supported tiers. ``none`` omits the effort
field and does not disable thinking.

``--reasoning-history``, ``--image-history``, ``--action-mode``,
``--max-actions-per-call`` and ``--image_size`` are also listed in the same
array. The default histories are -1/-1, action mode is single, batch length is 3,
and VLM image size is 1024. Compaction still applies after history retention.

For example, append ``--max-tokens 8192 --temperature 0.8
--compaction-max-messages 0`` to a single or batch command. Model input and the
requested output must still fit the actual server context. The
:download:`reference <../../../../tools/rl/REFERENCE.md>` documents services
and saved artifacts.

View Results
------------

``result.json`` records native success, errors, turns and tokens.
``states.json`` retains measured states and native termination flags.
Camera/depth artifacts, ``episode.mp4``, ``rollout.json``, native transcripts
and ``api_trace/`` retain the execution evidence. A clean exit alone proves
neither success nor a complete training record. The batch launcher audits
complete requests/responses and image hashes before publishing records.

Collect Tasks and Seeds
-----------------------

The reference describes bounded collection, valid/evaluated indexes,
infrastructure retries and interruption/resume behavior. Use a fresh batch
directory, start with one episode per GPU, and size concurrency from measured
scene memory. ``tools/rl/jobs`` contains example task/seed manifests.

Batch collection, the remote backend and the local service launcher default to
two concurrent cold starts per GPU. ``--max-initializing-per-gpu`` limits
initialization, independently of the number of resident episodes. Keep this
limit bounded when running multiple collectors on one host: each collector
has its own limit. LIBERO reset RPCs allow up to 900 seconds for the spawned
simulator and EGL initialization; action and planner budgets are unchanged.

Remote simulator host
---------------------

Adaptive episode can place motion, rendering and SAM3 on a separate backend
using ``python -m robots.rl.remote_manager`` and
``RPENT_REMOTE_TOOLKIT_ENDPOINT``. Both hosts need matching source. The
reference covers installation on each host, private routing, GPU capacities,
source/task/budget checks, file hashes and lease cleanup. Remote ZAP is
unsupported.

Remote episode supports single perception, single or batch actions, image
size and reasoning/image history limits. The backend echoes these settings
and exposes the same strict tool declarations as the planner. A source,
configuration or schema mismatch rejects the session before any tool executes.

On the environment host, start SAM3 and a pool as described above, then start
an independent backend with explicit capacities:

.. code-block:: bash

   export RPENT_ROOT=/path/to/RPent
   export BACKEND_OUTPUT=/path/to/backend-artifacts
   export SAM_POOL=http://127.0.0.1:8124
   cd "$RPENT_ROOT"
   python -m robots.rl.remote_manager \
     --output-dir "$BACKEND_OUTPUT" \
     --gpu-slots 0:32,1:32,2:32,3:32,4:32,5:32,6:32,7:32 \
     --max-initializing-per-gpu 2 --sam3-endpoint "$SAM_POOL" \
     --host 127.0.0.1 --port 28490

The slot counts are limits; qualify that layout with real resident environments
before collecting at that concurrency. Preserve the backend directory, including
failed sessions. Use a server-owned SSH reverse forward or another private route
to expose this loopback listener on each planner host. Bind forwarded ports to
loopback and keep private keys on their originating server.

On a planner host with the same RPent revision, pass the forwarded endpoint
with the ordinary workflow flags:

.. code-block:: bash

   export RPENT_REMOTE_TOOLKIT_ENDPOINT=http://127.0.0.1:28490
   python -m robots.rl.rl_episode \
     --request /path/to/request.json --output-dir /path/to/episode \
     --workflow episode --action-mode batch --max-actions-per-call 3 \
     --perception-mode single --libero-type pro \
     --tokenizer-path /path/to/Qwen3.5-4B --cuda-device 0

``--cuda-device`` selects the remote environment GPU here. The planner keeps
verified copies of the original RGB, depth and geometry under its own output
directory; trainers must reference that directory, not environment-host paths.
If an action response is lost or leaves the outcome uncertain, the planner
stops the episode without replaying the action in that session.

Train the Planner with RLinf
----------------------------

Use the paired `RLinf episode PPO/GRPO guide <https://github.com/zhexuanxu/RLinf/blob/rpent-ppo/docs/source-en/rst_source/examples/agentic/rpent_episode_sync.rst>`__ for environment installation, full Qwen3.5 SFT model files, native task manifests, service routing, training, checkpoint export and the 60-episode regression panel.

For the H200 profile, model serving, training, RPent planner subprocesses,
LIBERO-PRO and SAM3 share ``/opt/venv/reason``. Install the offline package on
each node, then start its local services:

.. code-block:: bash

   bash "$STACK_ROOT/envs/offline_package/install.sh"
   source "$STACK_ROOT/envs/offline_package/activate.sh"
   python -m robots.rl.services --sam3-checkpoint "$SAM3_CHECKPOINT_PATH" \
     --envs-per-gpu 3 --output-dir "$STACK_ROOT/logs/services-node0"

Set ``tools.rpent.python`` to ``/opt/venv/reason/bin/python``. Run one SAM3 and three environment slots per GPU. Across four nodes, this
supports the smoke test’s 80 concurrent candidates.
The episode workflow uses SAM3 and native scene assets. Keep the SFT workflow,
prompt, action batch size and explicit perception tools unchanged.

For collection on separate environment GPUs, ``--sam3-per-gpu 2`` starts two
SAM3 workers per GPU. Set ``--envs-per-gpu`` to the capacity measured on that
host; the default is three. SAM3 and the environment backend use the same
interpreter. The service manager stops all child processes when it exits.

Episodes save task outcomes and training data. Enable ``--record-api-trace``
when launching RPent directly, or ``tools.rpent.record_api_trace=true`` in
RLinf, to retain full HTTP and compaction traces. This records transport details
without changing model inputs or history compaction.

The trainer supplies ``task`` and ``planner`` objects in a request JSON. Planner options include ``max_turns``, ``max_tokens``, timeouts, ``history_compaction``, ``thinking`` and ``sampling_params``. Explicit sampling values override standalone defaults. RLinf retains sampled token IDs, behavior logprobs and ordered training images. Existing SFT encoding in ``ms-swift/examples/rpent`` must preserve those requests and image order; see ``tools/rl/REFERENCE.md`` for the integration contract.

Image feedback, action batches and history
------------------------------------------------------------------------------------

LIBERO supports ``--workflow episode_image``. It provides both initial camera
images and the latest pair after actions, exposes no ``view_env``, and binds
``segment`` / ``back_project`` to the latest observation without a ``step`` input.
Ordinary ``episode`` keeps explicit historical-frame queries.

LIBERO uses absolute grip-site ``xyz`` and ``orientation_rpy=[roll,pitch,yaw]``,
radians, fixed-axis XYZ: ``R=Rz(yaw)@Ry(pitch)@Rx(roll)``. At least one target
is required; omitted parts retain their measured starting value.
``--action-mode batch --max-actions-per-call 3`` uses sequential ``execute_actions``.
The entire list is validated first; success, truncation, partial or error stops it.
``--reasoning-history`` and ``--image-history`` accept -1 (no extra pruning),
0 (none), or N (latest N reply/observation groups); both cameras form one group.
Defaults are -1; compaction still applies. These switches support local LIBERO
episode workflows. Complete contracts and portable commands: ``tools/rl/WORKFLOWS.md``.


Multimodal token accounting
----------------------------------------

API compaction counts the current request after reasoning/image retention.
For Qwen, a local copy of the served checkpoint supplies the chat template,
tokenizer and image processor. A loopback SGLang endpoint can supply its local
checkpoint path automatically. For an explicit path, set:

.. code-block:: bash

   export RPENT_TOKENIZER_PATH=/path/to/served-checkpoint

For a remote SGLang endpoint with a locally mounted checkpoint, also set
``RPENT_TOKEN_COUNT_BACKEND=sglang``. The default ``auto`` detects loopback
SGLang; ``chat_template`` uses the checkpoint template without server-specific
tool normalization. Unexpected differences from provider usage emit a warning.

The processor counts resized patches and spatial merging without running the
vision encoder. Text, tool schemas, reasoning and template markers are counted
as rendered by the chat adapter. Image metadata is cached by content SHA and
processor configuration; every occurrence still occupies context, including
cache hits. Edited histories are recounted rather than anchored to old usage.

The launcher uses a 70% token threshold and an 80-to-10 message window, with
tool-call/result pairs and opening prompts preserved. A retained window can
still exceed the threshold when its remaining content is large; the audit
reports this condition rather than reducing output or context budgets.

With ``RPENT_RECORD_MODEL_REQUESTS=1``, ``api_trace/token_counts`` records
before/after counts, image sizes/SHA/visual tokens, trigger decisions, counting
time and the difference from provider input usage. When no local processor is
available, a warning and the audit identify the text/geometry estimate; it
does not treat image bytes as text. Checkpoint/template or server preprocessing
overrides require matching configuration and a fresh integration check:

.. code-block:: bash

   RPENT_RUN_TOKEN_COUNT_INTEGRATION=1 \
   RPENT_TOKEN_COUNT_ENDPOINTS=http://127.0.0.1:30000/v1 \
   pytest tests/e2e_tests/libero/test_token_counting.py -v

This opt-in check needs a running SGLang Qwen server and local processor files.
It checks three image sizes, tool-returned images, thinking/history settings
and both compression triggers against real input usage. Each request permits
one output token and executes no simulator actions; these are accounting
checks, not benchmark episodes.

Episode Observation Lists
-------------------------

Local ``episode`` supports ``--perception-mode batch`` to replace the three
standalone perception tools with ``observe(calls=[...])``. The default is
``single``. Episode-image, ZAP workflows and remote toolkits reject this option.

Each subcall has a local ``id`` and ``op``. Views have no source; measurements
use ``source`` with an issued image/mask ID or a same-call reference such as
``@v.agentview`` or ``@cup``. IDs bind frame and camera. The complete list is
validated first; independent nodes run concurrently, according to dependencies.
No extra model calls occur inside a tool. Action batching remains independent.

``--max-observe-calls 8`` limits list length; ``--observe-concurrency 8`` limits
concurrency. Both launchers support them and pin them for resume. See the
:download:`observation-list guide <../../../../tools/rl/OBSERVE.md>` for startup,
source rules, examples, result fields and audit.

Robot-Base Coordinates
----------------------

For any LIBERO RL workflow, use ``--world-cord general`` to express absolute positions in metres from the arm's mounting base: +x forward, +y to the robot's left, +z up. The default, ``base``, retains simulator-world coordinates and the original ``grip_site`` control point.

In ``general``, the TCP is the measured midpoint of the two finger pads. The controller compensates its offset from ``grip_site``, including changes in orientation and finger opening. Orientations remain fixed-axis XYZ roll, pitch and yaw in radians. The initial observation gives the actual table height above the base. Each reset establishes a fixed frame for that episode; missing native calibration is an error.

LIBERO ground tasks use their compiled collision floor, rather than a table. Their metadata reports ``support_surface="floor"`` and ``floor_height_above_base_m``; ``table_height_above_base_m`` is null. Both table and floor heights come from measured collision geometry, without task offsets.

The selected frame applies to robot poses, ``world_xyz``, ``top_z_m``, ``size_xyz_m``, ``axis_yaw`` and ``z_range_m`` across segment, back-project, batched observation and ZAP perception. Geometry is recomputed from transformed points; a size is an extent along the selected axes. Saved ``*_world.npz`` maps stay in simulator world; ``coordinates.json`` records the transform and TCP. Images, pixels and masks retain their usual camera orientation and observation-step association.

RGB-Only Control
----------------

Use ``--workflow pear`` for a LIBERO observation–action conversation inspired by `PEAR <https://arxiv.org/abs/2611.18562>`_. It retains RPent's absolute ``move_to`` and normalized ``set_gripper`` actions. Initial observations and action replies contain agentview RGB, wrist RGB and measured robot state. Use the same planner checkpoint and LIBERO assets; SAM3 and Pi0.5 weights are unnecessary. This workflow starts no SAM3 service and computes no depth maps; it exposes only the two action tools.

Launch it with the ordinary planner and task options:

.. code-block:: bash

   bash tools/rl/run.sh --simulator libero --suite libero_goal --task 0 --seed 0 \
     --workflow pear --world-cord general --action-mode single --reasoning

Use ``--no-reasoning`` to disable model thinking and request a direct tool call without prose. These flags alias ``--thinking`` and ``--no-thinking``; thinking is enabled by default. Every reply must contain exactly one valid action. Missing or multiple calls, unknown tools, invalid arguments, and reasoning text when disabled immediately end the episode as a normal failure, without another model request or infrastructure retry.

Native success or exhausted environment, request, time or context budgets ends the episode automatically. There is no ``finish`` tool. A blocked or partial motion returns its measured outcome so the next turn can continue. Remote toolkits support ``episode`` and ``pear`` with matching source and frame settings; start a matching backend without ``--sam3-endpoint`` for pear-only use. See the workflow guide for the backend command and model endpoint settings.
