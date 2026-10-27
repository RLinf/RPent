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
     - ``api``, ``claude_code``, ``codex``, or ``flash``.
   * - ``--model``
     - ``—``
     - ``api`` requires a provider prefix; SDK planners use their backend defaults.
   * - ``--max-turns``
     - ``100``
     - Planner turn limit. For ``api``, counts model requests across the conversation, including retries and follow-ups.
   * - ``--max-tokens``
     - ``8192``
     - ``api`` only: token limit per model response.
   * - ``--reasoning-effort``
     - ``api``: ``high``; ``claude_code``/``codex``: ``none``
     - Reasoning effort for ``api``, ``claude_code``, and ``codex``:
       ``none``, ``low``, ``medium``, ``high``, or ``xhigh``. Disabling
       reasoning reduced the average runtime from approximately 13.2 to
       7.9 minutes (about 40%) in our LIBERO Pro Long evaluations.
       Higher effort may improve task success rate. Supported levels
       ultimately depend on the selected model.
   * - ``--planner-timeout-s``
     - ``—``
     - Defaults to ``CODEX_TIMEOUT_S`` (Codex only), then ``CELL_TIMEOUT_S``, then 1200 seconds; terminal interactive API/Claude sessions are exempt.
   * - ``--base-url``
     - ``—``
     - ``api`` only: override the model endpoint. SDK planners use their own environment variables.
   * - ``--no-images``
     - ``false``
     - ``api`` only: omit images from model requests.
   * - ``--claude-code-max-budget-usd``
     - ``—``
     - Claude Code dollar budget; defaults to ``MAX_BUDGET_USD`` or 10.
   * - ``--output-dir``
     - ``—``
     - Artifact directory; the environment generates the default name.
   * - ``--memory-profile``
     - ``—``
     - Evaluation normally defaults to ``hf``; exploration and WAM evaluation use ``local``.
   * - ``--memory-dir``
     - ``—``
     - Local memory directory; use with the local profile or exploration.
   * - ``--memory-repo``
     - ``—``
     - HF dataset ID or URL with optional @revision; evaluation only.
   * - ``--explore``
     - ``false``
     - Enable exploration on supported environments.
   * - ``--interactive``, ``-i``
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

``--env-endpoint``, ``--am-endpoint``, and LIBERO’s ``--sam3-endpoint`` connect to existing environment, action-model (VLA or WAM), and SAM3 services respectively. RPent RPC addresses use ``[protocol://]host:port``, defaulting to ``http``. LingBot uses native WebSocket; see :doc:`../simulators/robotwin`. See :doc:`advanced_deployment` for GPU assignment and remote services.

``--am-backend`` selects one action model supported by the platform. LIBERO
defaults to ``pi05`` and RoboTwin to ``lingbot``. VLA and WAM share
``--am-endpoint`` for existing workers, and ``--am-checkpoint``, ``--am-python``,
``--am-root`` and ``--am-cuda-device`` for local deployment. Explicit local
deployment options are rejected with an endpoint.
``--am-checkpoint`` supplies local weights or a service model identity as required
by the backend. ``--am-python`` selects the provisioned model interpreter;
VLA defaults to the current interpreter. ``--am-root`` selects the working
directory, defaulting to the RPent checkout; Cosmos Policy requires its source
directory.

WAM uses the ``local`` memory profile with memory file tools and a
backend-specific directory; ``--memory-dir`` overrides it. WAM currently rejects
HF memory, Flash and exploration. See
:doc:`../development/add_action_model` for model launch options and supported pairs.

``--predict-future`` enables future-state/value outputs during local deployment.
Currently only Cosmos Policy supports this option; other backends reject it.
Configure it when starting the worker if connecting with ``--am-endpoint``.
RPent does not yet use these outputs for planning.

``--am-cuda-device`` selects the local model GPU independently of the simulator.
Local model deployment is supported by LIBERO, RoboTwin, RoboCasa, RoboDojo and
dual-Franka. Franka and YAM connect to externally deployed VLA services;
select their GPU when deploying the service.
LIBERO, RoboTwin, RoboCasa and RoboDojo use ``--env-cuda-device`` for the environment.
LIBERO, RoboDojo and dual-Franka use ``--sam3-cuda-device`` for local SAM3.
These services may share a GPU when memory permits. External endpoints
use the GPU configuration set when their services were started.
The original ``--cuda-device`` option remains a shared default for local services.
Per-service options take precedence, independent of argument order; without either
option, services inherit ``CUDA_VISIBLE_DEVICES``. For example,
``--cuda-device 0 --am-cuda-device 1`` starts the environment and SAM3 on GPU 0
and the WAM worker on GPU 1.
Standalone service commands retain their own ``--cuda-device`` where supported.
``rpent --robot libero --am-backend fast-wam --help`` groups connection,
local deployment and selected-model options. Model-specific options retain
their model prefix; options for another model are rejected. The action-model guide separates
one-time worker deployment from repeated task commands and includes examples
for both external services and workers started by RPent.

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

Action-Model Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the current options in the left column for new commands. The right column
lists hidden aliases retained for existing commands; they do not appear in
``--help`` and write the same configuration fields as their current names.

Common deployment options use ``--am-*``. Model-specific options use
``--pi05-*``, ``--cosmos-*``, ``--cosmos3-*``, ``--fast-wam-*`` or ``--lingbot-*``.
Here ``fast-wam`` is the model name, distinct from the old generic ``--wam-*``
prefix.

.. list-table:: Model option names
   :header-rows: 1

   * - Current option
     - Compatibility alias (hidden)
   * - ``--am-backend``
     - ``--vla-backend``, ``--wam-backend``
   * - ``--am-backend pi05``
     - ``--policy-backend rlinf`` (RoboDojo)
   * - ``--am-endpoint``
     - ``--vla-endpoint``, ``--wam-endpoint``
   * - ``--am-checkpoint``
     - ``--vla-model-path``, ``--wam-checkpoint``
   * - ``--am-python``
     - ``--pi05-python``, ``--wam-python``
   * - ``--am-root``
     - ``--wam-root``
   * - ``--am-cuda-device``
     - ``--vla-cuda-device``, ``--wam-cuda-device``
   * - ``--pi05-repo-id``
     - ``--vla-repo-id``
   * - ``--cosmos-text-embeddings``
     - ``--wam-text-embeddings``
   * - ``--cosmos-cached-instructions-only``
     - ``--wam-cached-instructions-only``
   * - ``--fast-wam-config``
     - ``--wam-config``
   * - ``--fast-wam-dataset-stats``
     - ``--wam-dataset-stats``

Model-Service Environment Variables
-----------------------------------

Set the credentials and endpoint for your chosen planner in the launch shell.

.. list-table::
   :header-rows: 1

   * - Planner
     - API key
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
   * - ``episode.mp4``
     - Episode video.
   * - ``run.log``
     - RPent process log.
   * - ``env_server.log`` / ``vla_server.log`` / ``wam_server.log`` / ``sam3_server.log``
     - Logs from environment and model servers started by this run.
   * - ``*_recipe.jsonl``
     - Action sequence exported from a successful task; failed runs may not produce this file.
   * - ``states.json``
     - Internal ``EnvState`` manifest. Inspect states through the Dashboard or ``view_env_state``; callers should not parse this file directly.
   * - ``agentview_depth.npz/00.npz``
     - Example step artifact: the directory uses the logical artifact name and contains numbered steps such as ``00.npz`` and ``01.npz``.

Run-level files live at the output root; step observations live in their own subdirectories. Use the native success criterion documented for each environment; a recipe, video, or planner summary does not replace it.
