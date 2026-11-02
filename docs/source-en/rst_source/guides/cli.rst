Command-Line Reference
======================

Run ``rpent --robot <name> --help`` for the full options of an environment. RPent supplies shared options; each environment adds task selection, resource paths, and service settings.

.. code-block:: bash

   rpent --help
   rpent --robot libero --help
   rpent --robot robocasa --help
   rpent --robot robotwin --help

Common Options
--------------

These options control the planner, memory, output directory, and Dashboard.
Each new process defaults to ``--planner api``; pass ``--planner deepseek_harness``
when using dsh. Tasks within one Dashboard Session retain its launch configuration.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Meaning
   * - ``--robot``
     - ``—``
     - Select a discovered robot or simulator.
   * - ``--planner``
     - ``api``
     - ``api``, ``claude_code``, ``codex``, ``deepseek_harness``, or ``flash``.
   * - ``--model``
     - ``—``
     - ``api`` requires a provider prefix; ``deepseek_harness`` requires an exact
       model ID. ``claude_code`` and ``codex`` have backend defaults.
   * - ``--max-turns``
     - ``100``
     - Planner turn limit. ``api`` counts requests including retries and follow-ups;
       ``deepseek_harness`` counts root model steps, excluding separate retries
       and compaction requests. Follow-up input does not reset the budget.
   * - ``--max-tokens``
     - ``8192``
     - ``api``: token limit per response. ``deepseek_harness``: runtime budget;
       the subscription transport does not enforce a hard per-reply cap.
   * - ``--reasoning-effort``
     - ``api``/``deepseek_harness``: ``high``; ``claude_code``/``codex``: ``none``
     - Reasoning effort for ``api``, ``claude_code``, and ``codex``:
       ``none``, ``low``, ``medium``, ``high``, or ``xhigh``. Disabling
       reasoning reduced the average runtime from approximately 13.2 to
       7.9 minutes (about 40%) in our LIBERO Pro Long evaluations.
       Higher effort may improve task success rate. Supported levels
       ultimately depend on the selected model. ``deepseek_harness`` supports
       ``low``, ``medium``, ``high``, and ``xhigh``; ``none`` is rejected.
   * - ``--planner-timeout-s``
     - ``—``
     - Defaults to ``CODEX_TIMEOUT_S`` (Codex only), then ``CELL_TIMEOUT_S``, then 1200 seconds; terminal interactive API/Claude sessions are exempt.
       DeepSeek Harness excludes idle user waits; cleanup can exceed the deadline.
   * - ``--base-url``
     - ``—``
     - ``api`` only: override the model endpoint. Claude Code and Codex use their
       environment variables; DeepSeek Harness rejects this option.
   * - ``--no-images``
     - ``false``
     - ``api`` only: omit images from model requests.
   * - ``--dsh-home``
     - ``DSH_HOME``
     - DeepSeek Harness credential home. Set the option or environment variable;
       use the same directory with ``rpent-dsh-login``.
   * - ``--claude-code-max-budget-usd``
     - ``—``
     - Claude Code dollar budget; defaults to ``MAX_BUDGET_USD`` or 10.
   * - ``--output-dir``
     - ``—``
     - Artifact directory; the environment generates the default name.
   * - ``--memory-profile``
     - ``—``
     - Defaults to ``hf`` for evaluation and ``local`` for exploration.
   * - ``--memory-dir``
     - ``—``
     - Local memory directory; use with the local profile or exploration. For dsh
       exploration, selects the seed copied into the run's working memory;
       automatic merging modifies only that copy.
   * - ``--memory-repo``
     - ``—``
     - HF dataset ID or URL with optional @revision; evaluation only.
   * - ``--explore``
     - ``false``
     - Enable exploration on supported environments.
   * - ``--tui``, ``-t``
     - ``false``
     - Enter instructions in the terminal; see :ref:`terminal-interaction`.
   * - ``--dashboard``
     - ``false``
     - Start a Dashboard session.
   * - ``--dashboard-host``
     - ``127.0.0.1``
     - Dashboard bind address.
   * - ``--dashboard-port``
     - ``0``
     - 0 selects an available port.
   * - ``--dashboard-language``
     - ``en``
     - ``en`` or ``zh-cn``.
   * - ``--verbose``
     - ``false``
     - Enable DEBUG logging.

Environment Options
-------------------

LIBERO selects tasks with ``--suite``, ``--task``, ``--seed``, and ``--libero-type``; the variant defaults to ``LIBERO_TYPE`` or ``pro``. RoboCasa365 and RoboTwin use ``--task-name``. Franka uses ``--task-id`` and ``--robot-config``. See :doc:`../simulators/libero`, :doc:`../simulators/robocasa`, :doc:`../simulators/robotwin`, :doc:`../real_world_robots/franka`, and :doc:`../real_world_robots/dual_franka` for details.

``--env-endpoint``, ``--vla-endpoint``, and LIBERO’s ``--sam3-endpoint`` connect to existing services using ``[protocol://]host:port``; the default protocol is ``http``. See :doc:`advanced_deployment` for GPU assignment and remote services.

The following defaults apply to LIBERO; check each other environment’s ``--help`` for its defaults.

.. list-table::
   :header-rows: 1

   * - LIBERO option
     - Default
     - Meaning
   * - ``--seed``
     - ``0``
     - Environment random seed.
   * - ``--memory-version``
     - ``auto``
     - HF corpus: ``auto``, ``GPT_5.5_xhigh``, or ``GPT_6_astra_low``. See :doc:`memory`; explicit versions cannot be combined with local memory or exploration.
   * - ``--max-episode-steps``
     - ``10000``
     - Environment step limit, separate from planner turns (``--max-turns``).

Model-Service Environment Variables
-----------------------------------

Set the credentials and endpoint for your chosen planner in the launch shell.

.. list-table::
   :header-rows: 1

   * - Planner
     - Authentication
     - Endpoint
   * - ``api`` / Anthropic; ``claude_code``
     - ``ANTHROPIC_API_KEY``
     - ``ANTHROPIC_BASE_URL``
   * - ``api`` / OpenAI
     - ``OPENAI_API_KEY``
     - ``OPENAI_BASE_URL``
   * - ``codex``
     - ``CODEX_API_KEY``
     - ``CODEX_BASE_URL``
   * - ``deepseek_harness``
     - ``rpent-dsh-login``; ``DSH_HOME`` or ``--dsh-home``
     - Fixed ``openai-codex`` subscription provider

Codex can also reuse existing Codex authentication. See :doc:`configure_planner` for model selection, endpoint precedence, and local server examples. Use ``rpent-check-llm`` for connectivity checks and ``rpent-memory`` for the operations in :doc:`memory`.

Each environment reads its own resource variables, including LIBERO’s ``PI05_CHECKPOINT_PATH`` and ``SAM3_CHECKPOINT_PATH``, RoboCasa365’s ``ROBOCASA_ASSETS_PATH``, and RoboTwin’s ``ROBOTWIN_ASSETS_PATH`` and ``LINGBOT_MODEL_PATH``.

.. _run-output-files:

Output Files
------------

``--output-dir`` selects the run directory. Without it, LIBERO writes to ``logs/<timestamp>_<suite>_t<task>_s<seed>/`` under the repository; other environments describe their directory names on their own pages.

The table uses LIBERO as an example. Files depend on the environment, recording, and run outcome.

.. list-table::
   :header-rows: 1

   * - File or path
     - Purpose
   * - ``transcript_*.json``
     - Planner conversation, tool calls, and completion information.
   * - ``result.json``
     - LIBERO native success, accepted finish, operator verdict, and stop status.
   * - ``dsh-*/``
     - DeepSeek Harness runtime events and archived observation images.
   * - ``memory-source.json`` / ``memory-merge.json`` / ``dsh-operator-context.json``
     - Exploration memory provenance, merge outcome, and consumed operator context.
   * - ``episode.mp4``
     - Episode video.
   * - ``run.log``
     - RPent process log.
   * - ``env_server.log`` / ``vla_server.log`` / ``sam3_server.log``
     - Logs from environment and model servers started by this run.
   * - ``*_recipe.jsonl``
     - Action sequence exported from a successful task; failed runs may not produce this file.
   * - ``states.json``
     - Internal ``EnvState`` manifest. Inspect states through the Dashboard or ``view_env_state``; callers should not parse this file directly.
   * - ``agentview_depth.npz/00.npz``
     - Example step artifact: the directory uses the logical artifact name and contains numbered steps such as ``00.npz`` and ``01.npz``.

Run-level files live at the output root; step observations live in their own subdirectories. Use the native success criterion documented for each environment; a recipe, video, or planner summary does not replace it.
