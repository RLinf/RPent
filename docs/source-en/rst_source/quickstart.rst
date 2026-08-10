Quick Start
===========

Before you begin, follow :doc:`installation` to install RPent and
download the LIBERO-PRO simulator assets. The steps below use LIBERO-PRO
with the ``claude_code`` planner to demonstrate a complete run.

1. Configure keys and checkpoints
---------------------------------

Export your Anthropic key, plus the paths to the VLA and SAM3 checkpoints:

.. code-block:: bash

   # Anthropic key; no need to export the base url if you use the
   # official endpoint.
   export ANTHROPIC_BASE_URL=https://xxx
   export ANTHROPIC_API_KEY=sk-xxx

   # VLA checkpoint — download from
   # https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT
   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

   # SAM 3.0 checkpoint — download from either
   # https://huggingface.co/facebook/sam3
   # https://modelscope.cn/models/facebook/sam3
   export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt

2. Run one LIBERO task
----------------------

Run a single LIBERO PRO task (``libero_object_swap``, task ``2``, seed
``0``) using the ``claude_code`` planner:

.. code-block:: bash

   rpent --env libero --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

To switch to another planner, such as ``codex`` or ``api``, see
:doc:`Agentic Planner <usage/configure_planner>`.

3. Monitor the run in the Dashboard
-----------------------------------

Add ``--dashboard`` to start a local Dashboard service and print its URL
in the terminal. Open the URL to confirm the configuration on the
launcher screen. Once the run starts, the page streams the agent's
reasoning, live camera and Pi0 views, an action timeline, and clip
replays. Use ``--dashboard-language zh-cn`` for the Chinese UI.

.. code-block:: bash

   rpent --env libero --dashboard --dashboard-language zh-cn \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

Key CLI options
---------------

The table lists only the options needed for a first run. Run
``rpent --help`` for other general options. See the
:doc:`LIBERO guide <usage/libero>` for detailed environment configuration.

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Flag
     - Default
     - Description
   * - ``--env``
     - required
     - Environment backend, e.g. ``libero``
   * - ``--suite``
     - required
     - Task suite, e.g. ``libero_object_task``, ``libero_spatial_swap``
   * - ``--task``
     - required
     - Task id within the suite
   * - ``--seed``
     - ``0``
     - Random seed
   * - ``--planner``
     - ``api``
     - ``api`` | ``claude_code`` | ``codex``
   * - ``--model``
     - —
     - Model id; for ``api``, prefix the provider (``anthropic:…``,
       ``openai:…``, ``openai-chat:…``)
   * - ``--dashboard``
     - off
     - Start a local Dashboard service for this run

What you should see
-------------------

A successful run:

1. Shows startup messages for ``env_server``, ``vla_server``, and
   ``sam3_server`` in the terminal.
2. Prints per-turn agent output and tool calls in the terminal, followed
   by the elapsed time, token usage, and path to the run record.
3. With the Dashboard enabled, also streams agent output, camera views,
   the action timeline, and clip replays to the Dashboard.
4. By default, artifacts are saved under ``logs/<timestamp>_<suite>_t<task>_s<seed>/``. They include ``transcript_*.json`` (run record), ``states.json`` (the ``EnvState`` manifest), ``recipe_*.jsonl`` (action sequence), and ``episode.mp4`` (episode video). Each step artifact has a directory named after its logical artifact name; zero-padded step files live inside it, for example ``agentview_depth.npz/00.npz`` and ``agentview_depth.npz/01.npz``. Run-level artifacts remain at the output root.

Inspect the final state through the Dashboard or
``view_env_state(step=-1)``. Its top-level ``terminated`` value is the
benchmark outcome. ``states.json`` is internal ``EnvState`` storage and should
not be parsed by callers. You can also open ``episode.mp4`` to review the run.
If something goes wrong, inspect the four log files described at the
bottom of :doc:`installation`.
