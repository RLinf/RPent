RoboTwin
========

`RoboTwin <https://robotwin-platform.github.io/>`_ is a simulation benchmark
for dual-arm robot manipulation, with a range of tabletop tasks and randomized
scenes. RPent runs RoboTwin through RLinf and uses LingBot-VLA to generate robot
actions.

.. note::

   The current code has not yet completed full effect-parity validation for
   RoboTwin; complete validation results will be released later.

Installation
------------

RoboTwin requires Python 3.11. The host must already provide a compatible
CUDA toolkit/NVCC, a compiler toolchain, and the system GL/EGL/Vulkan
libraries that SAPIEN depends on. Create an environment and install the
RoboTwin dependency set:

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
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

   ``.[robotwin]`` currently installs several dependencies from fixed Git
   commits, so the installation needs access to GitHub even when a PyPI mirror
   is configured. These will be replaced by released package versions after
   publication.

Download assets
---------------

Download the supported RoboTwin asset snapshot and set its location:

.. code-block:: bash

   robotwin-download-assets --output ~/.robotwin/assets
   export ROBOTWIN_ASSETS_PATH=~/.robotwin/assets
   # use the following command for users in mainland China
   # HF_ENDPOINT=https://hf-mirror.com robotwin-download-assets --output ~/.robotwin/assets

The downloader validates existing files and skips the download when the target
directory already contains a complete RoboTwin asset set.

Download the model
------------------

Download the LingBot checkpoint and set its location:

.. code-block:: bash

   # add HF_ENDPOINT=https://hf-mirror.com for mainland China users
   hf download RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500 \
      --revision e727b46cd220b66981ea4d2fd9ba84adc189e2cc \
      --local-dir /path/to/LingBot-VLA-RoboTwin-EEF-ckpt1500
   export LINGBOT_MODEL_PATH=/path/to/LingBot-VLA-RoboTwin-EEF-ckpt1500

The checkpoint includes the default RoboTwin robot configuration.

Run a task
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

View the result
---------------

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

Common options
--------------

RPent uses RoboTwin's ``demo_randomized`` task configuration by default, which
adds scene disturbances (random backgrounds, clutter, lighting, and table
height). Pass ``--task-config demo_clean`` for a simple, clean scene.

- ``--robotwin-assets-path`` overrides ``ROBOTWIN_ASSETS_PATH``.
- ``--vla-model-path`` overrides ``LINGBOT_MODEL_PATH``.
- ``--cuda-device`` runs the simulator and VLA on the same GPU.
- ``--env-cuda-device`` and ``--vla-cuda-device`` place the simulator and VLA
  on different GPUs. Do not combine these options with ``--cuda-device``.

For planner setup, external service endpoints, and offline resources, see
:doc:`configure_planner`, :doc:`advanced_deployment`, and
:doc:`../development/memory`.

Before each run, RPent automatically syncs optional RoboTwin memory and task
references from the public `RLinf/RPent-memory
<https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/robotwin>`_
dataset. These references can improve planning by providing previously verified
techniques; the run still starts if they are unavailable.

Planner memory and recipes
--------------------------

The read-only planner resources live under ``robotwin/`` in the dataset and are
synced to ``<RPent-clone-path>/resources/robotwin/``.

``memory/MEMORY.md`` indexes reusable experience across tasks, including
perception cues, control heuristics, recovery strategies, parameter-selection
guidance, and common failure modes. The planner can follow the index to read
only the memory entries relevant to the current task or observed failure.

For each evaluation task, ``recipe/<task>_s0.json`` is the semantic recipe
distilled from a successful trajectory. It describes the phase-level goals,
observable completion gates, control and VLA guidance, and known failure modes.
The companion ``recipe/recipe_<task>_s0.jsonl`` records the historical tool
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
Start with ``memory/MEMORY.md`` and read only the notes relevant to the current
task and failure mode.

Reproducing results
-------------------

The following result reproduces Harness VLA on RoboTwin C2R using ``gpt-5.5``
with ``xhigh`` reasoning effort:

- ``demo_randomized``: 58.0% (145/250)

The evaluation covers 50 RoboTwin tasks and runs five episodes per task, for a
total of 250 episodes. For each task, use the five official verified expert
seeds listed in ``robots/robotwin/eval/demo_randomized.json``. Because the
solvable seeds can differ across tasks, select each task's corresponding seeds
from that file rather than applying one fixed seed list to every task.

Reproduction command for one episode:

.. code-block:: bash

   rpent --robot robotwin \
     --task-name "task" \
     --task-config demo_randomized \
     --seed "seed" \
     --planner codex \
     --model gpt-5.5 \
     --reasoning-effort xhigh \
     --max-turns 100 \
     --planner-timeout-s 3600 \
     --max-episode-steps 10000

Replace ``task`` with a task name from ``demo_randomized.json`` and ``seed``
with one of that task's verified expert seeds. Before running, configure the
RoboTwin assets and LingBot-VLA checkpoint as described earlier on this page.
Success is determined by the latest
``TASK_ENV.eval_success`` value at the end of the episode, not merely by whether
the planner calls ``finish``.
