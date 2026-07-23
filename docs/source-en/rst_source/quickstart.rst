Quick Start
===========

This page ports the ``README.md`` Quick Start into the documentation.
It assumes you have already followed :doc:`installation` (RPent cloned
and ``pip install -e ".[full]"`` completed).

1. Configure keys and checkpoints
---------------------------------

Export your Anthropic key, plus the path to the VLA checkpoint:

.. code-block:: bash

   # Anthropic key; no need to export the base url if you use the
   # official endpoint.
   export ANTHROPIC_BASE_URL=https://xxx
   export ANTHROPIC_API_KEY=sk-xxx

   # VLA checkpoint — download from
   # https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT
   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   export LIBERO_TYPE=pro
   export CUDA_VISIBLE_DEVICES=0

2. Run one LIBERO task
----------------------

Run a single LIBERO PRO task (``libero_object_swap``, task ``2``, seed
``0``) using the ``claude_code`` planner:

.. code-block:: bash

   rpent --env libero --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

See :doc:`usage/configure_planner` to configure other planners
(``api``, ``codex``) and model providers.

1. Watch it run in the dashboard
--------------------------------

Add ``--dashboard`` to open a browser monitor for the run. It boots a
launcher screen where you pick the config, then streams the agent's
reasoning, live camera and Pi0 views, an action timeline, and clip
replays. Use ``--dashboard-language zh-cn`` for the Chinese UI.

.. code-block:: bash

   rpent --env libero --dashboard --dashboard-language zh-cn \
     --suite libero_goal_task --task 1 --seed 0 --planner claude_code

Key CLI options
---------------

The most common flags of ``rpent`` at a glance:

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Flag
     - Default
     - Description
   * - ``--env``
     - required
     - Environment backend. Currently ``libero``.
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
   * - ``--no-images``
     - off
     - Text-only mode: never send image bytes (for models that reject image input)
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
   * - ``--env-endpoint``
     - — (spawn)
     - ``[protocol://]host:port`` of an existing env_server
       (``protocol=http|socket``, default ``http``). If unset,
       one is spawned locally.
   * - ``--vla-endpoint``
     - — (spawn)
     - ``[protocol://]host:port`` of an existing vla_server (same rules).
       If unset, one is spawned locally.

What you should see
-------------------

A successful run:

1. Prints ``RPC server listening on http://127.0.0.1:<port>`` once each
   subprocess (env_server, vla_server) is up.
2. Prints per-turn agent reasoning (or streams it to the dashboard).
3. Ends when the LLM calls ``finish(success=True)``, or hits
   ``--max-turns`` / ``--max-episode-steps``.
4. Writes ``<output_dir>/transcript_*.json`` with the full turn-by-turn
   record and ``<output_dir>/episode.mp4`` with the rendered rollout.

If something goes wrong, inspect the three log files described at the
bottom of :doc:`installation`.
