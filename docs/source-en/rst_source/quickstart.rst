Quick start
===========

This page ports the ``README.md`` Quick Start into the documentation.
It assumes you have already followed :doc:`installation` (RPent cloned
and ``pip install -e ".[full]"`` completed).

1. Configure keys and checkpoints
---------------------------------

Export the API keys for the LLM provider(s) you want to use as the
reasoning brain, plus the path to the VLA checkpoint:

.. code-block:: bash

   # LLM API keys (used by the `api` planner via pydantic-ai)
   export ANTHROPIC_BASE_URL=https://xxx
   export ANTHROPIC_API_KEY=sk-xxx
   export OPENAI_BASE_URL=https://xxx
   export OPENAI_API_KEY=sk-xxx

   # VLA checkpoint — download from
   # https://huggingface.co/datasets/RLinf/rlinf-pi05-libero-130-fullshot-sft
   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   export LIBERO_TYPE=pro
   export CUDA_VISIBLE_DEVICES=0

You only need to set the keys for the providers you actually target.
For example, if you only run ``--planner claude_code``, you can skip
``OPENAI_*``.

2. Run one LIBERO task
----------------------

Run a single LIBERO PRO task (``libero_object_swap``, task ``2``, seed
``0``) using the ``api`` planner against an Anthropic model with an
8192-token cap:

.. code-block:: bash

   rpent --suite libero_object_swap --task 2 --seed 0 \
     --planner api --model anthropic:claude-opus-4-8 --max-tokens 8192

**Model id conventions.** ``--model`` accepts a provider-prefixed id
for the ``api`` planner, and a bare id for the ``claude_code`` /
``codex`` planners:

- OpenAI-compatible chat endpoints — ``--model openai-chat:glm-5.2``
- OpenAI responses endpoints — ``--model openai:gpt-5.5``
- ``claude_code`` / ``codex`` — no provider prefix, e.g.
  ``--model claude-opus-4-8``

See :doc:`usage/configure_planner` for the full brain-swapping guide.

3. Watch it run in the dashboard
--------------------------------

Add ``--dashboard`` to open a browser monitor for the run. It boots a
launcher screen where you pick the config, then streams the agent's
reasoning, live camera and Pi0 views, an action timeline, and clip
replays. Use ``--dashboard-language zh-cn`` for the Chinese UI.

.. code-block:: bash

   rpent --dashboard --dashboard-language zh-cn \
     --suite libero_goal_task --task 1 --seed 0 --planner claude_code

4. RoboCasa
-----------

RoboCasa uses a separate entrypoint and one-time setup:

.. code-block:: bash

   bash scripts/setup_robocasa.sh                                # one-time
   bash scripts/run_robocasa.sh PickPlaceCounterToCabinet 0 0    # <task> <gpu> <seed>

See `docs/SETUP_ROBOCASA.zh.md
<https://github.com/RLinf/RPent/blob/main/docs/SETUP_ROBOCASA.zh.md>`_
for the full RoboCasa365 + RLDX-1 walkthrough, and
:doc:`usage/robocasa` for what the RoboCasa toolkit
exposes to the agent.

Key CLI options
---------------

The most common flags of ``rpent`` at a glance:

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Flag
     - Default
     - Description
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
     - Reasoning brain: ``api`` | ``claude_code`` | ``codex``
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
   * - ``--max-episode-steps``
     - ``10000``
     - Max env steps
   * - ``--libero-type``
     - ``LIBERO_TYPE`` or ``pro``
     - LIBERO variant: ``standard`` | ``pro`` | ``plus``
   * - ``--cuda-device``
     - inherited
     - GPU device(s) exposed to the env / vla servers
   * - ``--dashboard``
     - off
     - Start the local dashboard for this run
   * - ``--dashboard-language``
     - ``en``
     - Dashboard UI language: ``en`` | ``zh-cn``
   * - ``--vla-endpoint``
     - —
     - Reuse an already-running vla_server instead of spawning one
   * - ``--no-driver``
     - off
     - Attach to an existing env_server / vla_server

What you should see
-------------------

A successful run:

1. Prints ``env server ready at 127.0.0.1:<port>`` once the driver
   process is up.
2. Prints per-turn agent reasoning (or streams it to the dashboard).
3. Ends when the LLM calls ``finish(success=True)``, or hits
   ``--max-turns`` / ``--max-episode-steps``.
4. Writes ``<output_dir>/transcript_*.json`` with the full turn-by-turn
   record and ``<output_dir>/episode.mp4`` with the rendered rollout.

If something goes wrong, inspect the three log files described at the
bottom of :doc:`installation`.
