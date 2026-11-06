RoboTwin
============

.. figure:: https://robotwin-platform.github.io/assets/images/teaser.png
   :alt: RoboTwin 2.0 environment overview
   :width: 90%
   :align: center

   RoboTwin 2.0 overview. Source: `RoboTwin project <https://robotwin-platform.github.io/>`_. The results in this figure are reported by RoboTwin.

Run bimanual tabletop tasks with RPent in `RoboTwin <https://robotwin-platform.github.io/>`_, then reproduce the C2R experiment in randomized scenes. RPent connects to the simulator through RLinf and uses LingBot-VLA to generate actions.

.. _robotwin-overview:

Overview
------------

Check the model, task, and runtime requirements before following the installation and run steps.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      LingBot-VLA (default), UniWAM, Fast-WAM.

   .. grid-item-card:: Planners

      ``api``, ``claude_code``, ``codex``

   .. grid-item-card:: Tasks

      50 C2R tasks

   .. grid-item-card:: Hardware

      Linux, NVIDIA GPU; Python 3.11; CUDA and GL/EGL/Vulkan.

Tasks
~~~~~~~~~~~~

C2R uses successful clean-scene trajectories as prior experience and evaluates in randomized scenes. The two task configurations serve these roles.

.. list-table::
   :header-rows: 1

   * - Configuration
     - Scene
     - Use
   * - ``demo_clean``
     - Clean scenes.
     - Collect task experience.
   * - ``demo_randomized``
     - Random backgrounds, clutter, lighting, and table height.
     - C2R evaluation: 50 tasks, each with five verified seeds.

.. _robotwin-observation-action:

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The table distinguishes planner tools, model inputs, and the environment’s success criterion.

.. list-table::
   :header-rows: 1

   * - Item
     - Description
   * - Observation
     - Head and left/right wrist RGB views, depth/world coordinates for localization, and robot state. Both models use RGB views, robot state, and task text; UniWAM also uses recent joint-state feedback.
   * - Action
     - The selected backend exposes ``lingbot_act`` or ``wam_act`` alongside motion primitives. The environment accepts 16-value end-effector actions (``ee``) or 14-value joint/gripper actions (``qpos``).
   * - Reward / success
     - Evaluate success using native ``TASK_ENV.eval_success``. Planner completion and episode timeouts are separate outcomes.
   * - Task prompt
     - Listed evaluation task/seed pairs use their verified ``task_language`` after reset; custom seeds use language from the native environment.

Installation
------------

RoboTwin requires Python 3.11. The host must already provide a compatible
CUDA toolkit/NVCC, a compiler toolchain, and the system GL/EGL/Vulkan
libraries that SAPIEN depends on. Create an environment and install the
RoboTwin dependency set:

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent
   uv venv --python 3.11 .venv-robotwin
   source .venv-robotwin/bin/activate
   uv pip install -e ".[robotwin]"

You do not need to run the RLinf installer or clone RoboTwin separately.

For networks closer to Chinese mirrors:

.. code-block:: bash

   uv pip install -e ".[robotwin]" \
      --default-index https://mirrors.aliyun.com/pypi/simple \
      --index https://pypi.tuna.tsinghua.edu.cn/simple

.. note::

   ``.[robotwin]`` uses SAPIEN 3.0.0b1. Other versions can change simulator
   observations and reduce model performance.

.. note::

   The RoboTwin and LingBot runtimes in ``.[robotwin]`` are installed from
   released PyPI packages. cuRobo is still built from a GitHub official tag,
   so the installation needs access to GitHub even when a PyPI mirror is
   configured.

Download Assets
---------------

Download the supported RoboTwin asset snapshot and set its location:

.. code-block:: bash

   robotwin-download-assets --output ~/.robotwin/assets
   export ROBOTWIN_ASSETS_PATH=~/.robotwin/assets
   # use the following command for users in mainland China
   # HF_ENDPOINT=https://hf-mirror.com robotwin-download-assets --output ~/.robotwin/assets

The downloader validates existing files and skips the download when the target
directory already contains a complete RoboTwin asset set.

Download the Model
------------------

Download the LingBot checkpoint and set its location:

.. code-block:: bash

   # add HF_ENDPOINT=https://hf-mirror.com for mainland China users
   hf download RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500 \
      --revision e727b46cd220b66981ea4d2fd9ba84adc189e2cc \
      --local-dir /path/to/LingBot-VLA-RoboTwin-EEF-ckpt1500
   export LINGBOT_MODEL_PATH=/path/to/LingBot-VLA-RoboTwin-EEF-ckpt1500

The checkpoint includes the default RoboTwin robot configuration.

UniWAM Backend
~~~~~~~~~~~~~~

Select UniWAM with ``--am-backend uniwam``. The planner calls
``wam_act(max_chunks=1)`` to execute 16 joint/gripper actions per chunk.
UniWAM uses recent action history to predict the next actions and supports
RoboTwin evaluation, exploration and memory configuration.

In a separate Python 3.11 model environment, install the policy dependencies
following the official
`UniWAM RoboTwin deployment guide <https://github.com/UniWAM/UniWAM/tree/8dcd3401bccda8c002fc558e356b975b1789eecd/inference/robotwin/uniwam>`_,
then install RPent with ``pip install -e /path/to/RPent``.
A verified dependency combination is Torch 2.6.0+cu124, torchvision 0.21.0+cu124,
FlashAttention 2.7.4.post1, Transformers 4.57.3, Diffusers 0.35.1,
DeepSpeed 0.16.9 and NumPy 1.26.4.

Download the RoboTwin-clean history-flow checkpoint from the
`UniWAM model collection <https://huggingface.co/collections/chenpyyy/uniwam>`_,
plus `Wan2.2-TI2V-5B <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B>`_ and
`Qwen3-VL-2B-Instruct <https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct>`_.
Arrange the resources as follows; existing directories can be symlinked:

.. code-block:: text

   uniwam/
   ├── policy/                  # complete official inference/robotwin/uniwam directory
   ├── checkpoint/
   │   ├── config.json
   │   └── pytorch_model/
   ├── wan/                     # Wan2.2-TI2V-5B
   └── vlm/                     # Qwen3-VL-2B-Instruct

The checkpoint's ``config.json`` uses ``flow_source.mode: history`` and
``history_length: 16``. RPent runs four inference steps.

After :doc:`configuring a planner <../guides/configure_planner>`, run from the
RoboTwin environment:

.. code-block:: bash

   export UNIWAM_MODEL_PATH=/path/to/uniwam
   export UNIWAM_PYTHON=/path/to/model-env/bin/python
   rpent --robot robotwin --am-backend uniwam \
      --am-checkpoint "$UNIWAM_MODEL_PATH" --am-python "$UNIWAM_PYTHON" \
      --task-name beat_block_hammer --task-config demo_randomized --seed 100000

``--am-checkpoint`` selects the resource directory above; ``--am-python`` selects
the model environment's interpreter. RPent starts and cleans up the model process.

To reuse the model across tasks, start a standalone worker in the model environment:

.. code-block:: bash

   python -m rpent.robots.components.uniwam.server \
      --checkpoint /path/to/uniwam --port 8118

For task runs, replace ``--am-checkpoint`` and ``--am-python`` with
``--am-endpoint http://127.0.0.1:8118``. The operator manages this worker's lifetime.

Integration tests cover mixed-tool execution and the complete CLI execution chain.
Set the two path variables above, plus ``ROBOTWIN_ASSETS_PATH``,
``CUDA_VISIBLE_DEVICES`` and ``RPENT_E2E_OUTPUT_DIR``, then run:

.. code-block:: bash

   RPENT_RUN_UNIWAM_INTEGRATION=1 \
      pytest tests/e2e_tests/robotwin/test_uniwam.py -v --timeout=1200

The tests use real models, simulation and a scripted planner, and save their
results in ``RPENT_E2E_OUTPUT_DIR``.

Fast-WAM Configuration
~~~~~~~~~~~~~~~~~~~~~~

Prepare a RoboTwin Fast-WAM worker using :doc:`../development/add_action_model`.
It uses head/left-wrist/right-wrist RGB, native joint targets, and qpos14 action
chunks. Select it with ``--am-backend fast-wam``; LingBot weights are not
required. From the simulator environment:

.. code-block:: bash

   rpent --robot robotwin --task-name beat_block_hammer \
     --task-config demo_randomized --seed 100000 \
     --am-backend fast-wam --am-endpoint http://127.0.0.1:8117 \
     --memory-profile local --planner api --model <planner-model>

``wam_act(prompt=None, max_chunks=1)`` replaces ``lingbot_act``; each chunk
executes the worker's configured ``execute_steps`` through ``action_type="qpos"``.
Fast-WAM uses a required local memory manager and file tools, with a backend-specific
directory that may be empty; ``--memory-dir`` selects a compatible local corpus.
HF memory and exploration are unsupported. Native ``eval_success`` determines
task success. The LingBot and HF memory workflows below apply to the VLA path.

Run a Task
----------

Run one episode from the activated environment:

.. code-block:: bash

   # add HF_ENDPOINT=https://hf-mirror.com for mainland China users,
   # as it will download robotwin task related memory data
   rpent --robot robotwin \
      --task-name beat_block_hammer \
      --seed 100000 \
      --planner codex \
      --model gpt-5.5

Change ``--task-name`` to run another task. For standard randomized
evaluation seeds, see the note below. See ``rpent --robot robotwin --help`` for
the complete option list.

.. note::

   ``--seed`` is the exact RoboTwin scene seed. For the standard
   ``demo_randomized`` evaluation, use one of the five validated seeds for the
   selected task in the `RoboTwin evaluation suite
   <https://github.com/RLinf/RPent/blob/main/robots/robotwin/eval/demo_randomized.json>`_.

   The suite was filtered with RoboTwin expert execution: seeds that could not
   be initialized stably or did not pass the expert rollout were skipped.
   Other seeds can still be passed explicitly for custom runs.

.. _view-the-result:

View Results
------------

The terminal shows server startup, planner output, and tool calls. By default,
the run is saved under
``logs/<timestamp>_robotwin_<task-name>_s<seed>/``. Start with these files when
checking a run:

- ``run.log`` contains the RPent process log.
- ``robotwin_env_server.log`` and ``lingbot_vla_server.log`` contain simulator
  and model startup errors.
- ``transcript_*.json`` contains the planner conversation and final response.

RoboTwin's native ``TASK_ENV.eval_success`` value in the latest tool result is
the task-success source. Calling ``finish`` ends the Planner loop; it does not
define a second success condition.

Add ``--dashboard`` to watch the planner and the head and wrist camera views in
a browser. The command prints the Dashboard URL after startup.

Common Options
--------------

RPent uses RoboTwin's ``demo_randomized`` task configuration by default, which
adds scene disturbances (random backgrounds, clutter, lighting, and table
height). Pass ``--task-config demo_clean`` for a simple, clean scene.

- ``--robotwin-assets-path`` overrides ``ROBOTWIN_ASSETS_PATH``.
- ``--am-checkpoint`` overrides ``LINGBOT_MODEL_PATH``.
- ``--env-cuda-device 0 --am-cuda-device 1`` assigns GPU 0 to the local
  simulator and GPU 1 to the local action model. For GPU defaults and overrides,
  see :doc:`../guides/cli`.

For planner setup, external service endpoints, and offline resources, see
:doc:`../guides/configure_planner`, :doc:`../guides/advanced_deployment`, and
:doc:`../guides/memory`.

Before evaluation runs with the HF memory profile, RPent syncs optional RoboTwin memory and task
references from the public `RLinf/RPent-memory
<https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/robotwin>`_
dataset. These references can improve planning by providing previously verified
techniques; the run still starts if they are unavailable.

.. _planner-memory-and-recipes:

Task Memory
-----------

The published planner memory lives under ``robotwin/`` in the dataset. The HF
profile syncs it to ``<RPent-clone-path>/memory/robotwin/`` for read-only use.

``MEMORY.md`` indexes reusable experience across tasks, including
perception cues, control heuristics, recovery strategies, parameter-selection
guidance, and common failure modes. The planner can follow the index to read
only the memory entries relevant to the current task or observed failure.

For each evaluation task, ``task-specific/<task>_s0.json`` is the semantic recipe
distilled from a successful trajectory. It describes the phase-level goals,
observable completion gates, control and VLA guidance, and known failure modes.
The companion ``task-specific/<task>_s0_recipe.jsonl`` records the historical tool
calls from that trajectory, providing evidence about action order, tool choice,
and action-chunk cadence.

The ``_s0`` suffix is a uniform recipe-slot name used for convenient prompt
lookup; it does not mean RoboTwin seed 0. Since some randomly generated seeds
may be unsolvable, the source seed for each recipe is selected using RoboTwin's
official expert program. The actual source seed is recorded in the recipe
metadata.

These recipes are derived from successful ``demo_clean`` trajectories for use
as strategy priors in independently generated ``demo_randomized`` scenes. They
transfer phase structure, observable gates, control choice, and VLA chunk
cadence. Source task language, arm choices, pixels, coordinates, poses,
clearances, and contacts are not commands for a new episode. The current native
task language and fresh observations remain authoritative, and all geometry
must be localized again.

An ``evidence_status`` of ``supported`` means the recipe is backed by a
successful clean trajectory. An ``experimental`` recipe remains a weak prior.
Start with ``MEMORY.md`` and read only the notes relevant to the current
task and failure mode.

Exploration Mode
----------------

Add ``--explore`` to let the planner retry a task across fresh episodes and
write local memory. As with LIBERO, one run allows up to three planner sessions
with at most five attempts per session by default:

.. code-block:: bash

   rpent --robot robotwin --task-name beat_block_hammer \
     --task-config demo_randomized --seed 100000 \
     --planner codex \
     --explore --explore-sessions 3 --explore-attempts-per-session 5 \
     --memory-dir /path/to/robotwin-memory

``reset`` uses the environment's ordinary episode reset, so the planner
re-runs perception after every reset. The runner exports only the winning
commands after the final reset. Exploration memory is written to the current
local inbox. Drafts are merged when the run completes normally without an agent
execution error. Pass ``--no-auto-merge-memory`` to disable automatic merging.

.. raw:: html

   <span id="reproducing-results"></span>

Experiment Reproduction
-----------------------

The following result was obtained for
:doc:`Harness VLA <../resources/harnessvla>` on RoboTwin C2R. On the
`reproduce/robotwin
<https://github.com/RLinf/RPent/tree/reproduce/robotwin>`_ branch, use
``gpt-5.5`` with ``xhigh`` reasoning effort to reproduce this result:

- ``demo_randomized``: 62.4% (156/250)

This run produced 156 successes, 58 task failures, and 36 episode timeouts.

The evaluation covers 50 RoboTwin tasks and runs five episodes per task, for a
total of 250 episodes. For each task, use the five official verified expert
seeds listed in ``robots/robotwin/eval/demo_randomized.json``. Because the
solvable seeds can differ across tasks, select each task's corresponding seeds
from that file rather than applying one fixed seed list to every task.
For a listed task/seed pair, RPent also binds the table's ``task_language``
after resetting the exact scene. Custom seeds that are not listed continue to
use the language generated by the native RoboTwin environment.

Reproduction command for one episode:

.. code-block:: bash

   rpent --robot robotwin \
     --task-name beat_block_hammer \
     --task-config demo_randomized \
     --seed 100000 \
     --planner codex \
     --model gpt-5.5 \
     --reasoning-effort xhigh \
     --max-turns 100 \
     --planner-timeout-s 4800 \
     --max-episode-steps 10000

This example runs ``beat_block_hammer`` at seed 100000. To reproduce the full
score, run every task with its own verified expert seeds from ``demo_randomized.json``. Before running, configure the
RoboTwin assets and LingBot-VLA checkpoint as described earlier on this page.
An episode counts as successful only if the final value of
``TASK_ENV.eval_success`` is ``true``; the planner's ``finish`` call does not by
itself indicate success.

HF Corpus Selection
-------------------

By default, CLI and Dashboard keep using the unversioned ``robotwin/`` corpus
from HF ``main``. To select a published subdirectory explicitly, add
``--memory-profile hf --memory-version <corpus-directory>``; this reads
``robotwin/<corpus-directory>/``. No model-to-corpus mapping is assumed.
Use ``--memory-repo owner/dataset@revision`` to select another dataset or revision.
For offline evaluation, download the subtree first and use
``--memory-profile local --memory-dir <downloaded-corpus>`` without
``--memory-version``. The selected directory must retain the RoboTwin memory layout.
