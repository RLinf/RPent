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

All modes use the API planner and the same robot/frame/gripper instructions.

.. list-table:: LIBERO context policies
   :header-rows: 1

   * - Workflow
     - Perception and plan
     - Conversation
   * - ``episode``
     - ``view_env``, ``segment``, ``back_project``
     - Entire episode
   * - ``zap_turn``
     - Automatic perception; text todo
     - Independent request snapshots
   * - ``zap_episode``
     - Automatic perception; no todo
     - Entire episode
   * - ``zap_subtask``
     - Automatic perception; structured todo
     - Cleared by every valid ``update_todo``

Motion tools are ``move_to``, ``set_gripper`` and ``finish``. Automatic
perception uses current SAM3 boxes and depth-measured world geometry.
Each reply has one tool call; subtask initialization instead requires the
query and todo tools together. Context resets preserve the global budget.
Native task text comes directly from the simulator, without task prompt files.
RoboCasa adds ``move_base`` and RoboTwin anatomical arm selection and paired
commands. Both use ``episode``.

The :download:`workflow guide <../../../../tools/rl/WORKFLOWS.md>` gives exact
bootstrap/reset rules, complete tool lists, SAM3 pool setup and fixed-panel
comparison commands. ``--image_size`` selects 1024, 512 or 256 for VLM images;
native RGB-D/SAM3 stay1024 and displayed pixel/box coordinates are mapped.

Planner Configuration
---------------------

The default budgets are 100 model requests, 10000 simulator steps,
32768 output tokens per request, a 262144-token context, accumulated-history compaction
at 70%, and a 7200-second planner timeout. The batch launcher also enforces a
14400-second episode timeout. All LIBERO modes share ``simple``, ``base``,
``base_gripper`` and ``base_gripper_camera`` prompt variants.

The launcher explicitly defaults presence penalty to0. The example sets
repetition penalty1.0. Overrides are captured in actual model requests. Thinking sampling uses
temperature 1.0, top-p 0.95 and top-k 20. Qwen3.8 defaults to ``xhigh`` effort.
The :download:`reference <../../../../tools/rl/REFERENCE.md>` lists all service,
planner and batch overrides and describes the actual saved requests.

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

Remote simulator host
---------------------

Adaptive episode can place motion, rendering and SAM3 on a separate backend
using ``python -m robots.rl.remote_manager`` and
``RPENT_REMOTE_TOOLKIT_ENDPOINT``. Both hosts need matching source. The
reference covers installation on each host, private routing, GPU capacities,
source/task/budget checks, file hashes and lease cleanup. Remote ZAP is
unsupported.

RLinf Training
--------------

The reference preserves offline installation and RLinf/SFT integration details.
RLinf owns policy training/checkpoints; RPent owns simulator/tool execution.
Its episode entry is ``robots.rl.rl_episode``. Dataset encoding belongs to
``ms-swift/examples/rpent`` and must preserve saved requests and image order.
