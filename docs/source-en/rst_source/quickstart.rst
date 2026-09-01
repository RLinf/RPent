Quick Start
===========

Before you begin, follow :doc:`installation` to install RPent and
download the LIBERO-PRO simulator assets. The steps below use LIBERO-PRO
with the ``claude_code`` planner to demonstrate a complete run.

1. Configure keys and checkpoints
---------------------------------

Export your Anthropic key, then download and configure the VLA and SAM3
checkpoints:

.. code-block:: bash

   # Anthropic key; no need to export the base url if you use the
   # official endpoint.
   export ANTHROPIC_BASE_URL=https://xxx
   export ANTHROPIC_API_KEY=sk-xxx

   # VLA checkpoint — download from
   # https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT
   pip install "huggingface_hub>=0.34,<1.0"

   hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
     --exclude optimizer.pt \
     --local-dir ./checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT

   export PI05_CHECKPOINT_PATH=$PWD/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT

   # SAM 3.0 checkpoint — download from
   # https://modelscope.cn/models/facebook/sam3
   pip install -U modelscope

   modelscope download facebook/sam3 \
     --local-dir ./checkpoints/sam3

   export SAM3_CHECKPOINT_PATH=$PWD/checkpoints/sam3/sam3.pt

2. Run one LIBERO task
----------------------

Run a single LIBERO PRO task (``libero_object_swap``, task ``2``, seed
``0``) using the ``claude_code`` planner:

.. code-block:: bash

   rpent --robot libero --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

To switch to another planner, such as ``codex`` or ``api``, see
:doc:`Agentic Planner <usage/configure_planner>`.

3. Monitor the run in the Dashboard
-----------------------------------

Add ``--dashboard`` to start a local Dashboard and print its URL in the terminal:

.. code-block:: bash

   rpent --robot libero --dashboard --dashboard-language zh-cn \
     --planner claude_code --model claude-opus-4-8

Open the URL and confirm the configuration. Once the services are ready, enter
``/rpent-task libero_object_swap 2 0`` in the page to start a task. The Dashboard
streams agent reasoning, camera views, and the action timeline; submit another
task after the current one finishes. Use ``--dashboard-language zh-cn`` for the
Chinese UI.

Key CLI options
---------------

The table lists the main CLI options. Run ``rpent --help`` for other
general options. See the :doc:`LIBERO guide <usage/libero>` for detailed
robot configuration.

**Main**

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Flag
     - Default
     - Description
   * - ``--robot``
     - — (required)
     - Robot backend. Currently ``libero``.
   * - ``--suite``
     - — (required)
     - Task suite, e.g. ``libero_object_task``, ``libero_spatial_swap``
   * - ``--task``
     - — (required)
     - Task id within the suite
   * - ``--seed``
     - ``0``
     - Random seed
   * - ``--libero-type``
     - ``LIBERO_TYPE`` or ``pro``
     - LIBERO variant: ``standard`` | ``pro`` | ``plus``

**Planner**

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Flag
     - Default
     - Description
   * - ``--planner``
     - ``api``
     - ``api`` | ``claude_code`` | ``codex``
   * - ``--model``
     - —
     - Model id; for ``api``, prefix the provider (``anthropic:…``,
       ``openai:…``, ``openai-chat:…``)
   * - ``--max-turns``
     - ``100``
     - Max agent turns
   * - ``--max-tokens``
     - ``8192``
     - Max tokens per LLM reply
   * - ``--reasoning-effort``
     - ``none``
     - Reasoning effort for ``api``, ``claude_code``, and ``codex``:
       ``none`` | ``low`` | ``medium`` | ``high`` | ``xhigh``. Disabling
       reasoning reduced the average runtime from approximately 13.2 to
       7.9 minutes (about 40%) in our LIBERO Pro Long evaluations.
       Higher effort may improve task success rate. Supported levels
       ultimately depend on the selected model.
   * - ``--no-images``
     - off
     - Text-only mode: never send image bytes (for models that reject
       image input)

**Environment**

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Flag
     - Default
     - Description
   * - ``--max-episode-steps``
     - ``10000``
     - Max env steps
   * - ``--cuda-device``
     - inherited
     - GPU device exposed to the env / VLA / SAM3 servers
   * - ``--env-endpoint``
     - — (spawn)
     - ``[protocol://]host:port`` of an existing env_server
       (``protocol=http|socket``, default ``http``). If unset, one is
       spawned locally.
   * - ``--vla-endpoint``
     - — (spawn)
     - ``[protocol://]host:port`` of an existing vla_server (same rules).
       If unset, one is spawned locally.
   * - ``--sam3-endpoint``
     - — (spawn)
     - ``[protocol://]host:port`` of an existing RPent SAM3 service
       (same rules). If unset, one is spawned locally.

**Dashboard**

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Flag
     - Default
     - Description
   * - ``--dashboard``
     - off
     - Start a local Dashboard
   * - ``--dashboard-language``
     - ``en``
     - Dashboard UI language: ``en`` | ``zh-cn``

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
