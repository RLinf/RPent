Memory Management
=================

.. _memory-management:

RPent memory is maintained per robot and lets runs reuse already-validated
task experience and operating strategy instead of rediscovering it from
scratch each time.

Run modes
---------

Memory is used differently in the two run modes:

- **Evaluation** reads existing memory but does not update it.
- **Exploration** generates and updates local memory. It is currently
  supported only by LIBERO.

See the :ref:`LIBERO exploration guide <libero-exploration>` for the detailed
Exploration and local-memory Evaluation workflow.

Directory layout
----------------

Memory published to Hugging Face and memory prepared locally for evaluation
use the same directory structure:

.. code-block:: text

   <memory-root>/
   |-- MEMORY.md
   |-- global/
   |-- suite/
   `-- task_only/
       |-- <cell>.json
       |-- <cell>_recipe.jsonl
       `-- <task_key>.md

The default local root is ``memory/<robot>/``. On Hugging Face, LIBERO has
model-specific roots, described below; other robots use ``<robot>/``. A custom
``--memory-dir`` may point at any directory laid out like the tree above.

Every subtree is optional; a robot ships only the directories it uses:

- ``global/`` holds cross-task lessons distilled from successful experience.
- ``suite/`` holds task-level experience accumulated during exploration,
  organised by suite and reusable across seeds of the same task.
- ``task_only/`` holds same-task references such as the audit and recipe
  produced by successful runs.
- ``MEMORY.md`` indexes ``global/`` and ``suite/``.

During evaluation the planner may read only the current robot's memory.
Missing a layer does not stop a task from running.

Using memory
------------

RPent downloads memory from the public ``RLinf/RPent-memory`` dataset.
LIBERO selects one version with ``--memory-version auto`` (the default):

.. list-table:: LIBERO memory versions
   :header-rows: 1
   :widths: 25 35 40

   * - Running model
     - Memory directory under ``libero/``
     - Exploration configuration
   * - ``gpt-5.5``
     - ``GPT_5.5_xhigh``
     - Codex, reasoning on, xhigh
   * - ``gpt-6-astra``
     - ``GPT_6_astra_low``
     - Codex, reasoning on, low

Provider prefixes such as ``openai:`` are recognized. Codex uses ``--model``
first, then ``CODEX_MODEL``. Unknown models, Claude, or an unknown backend
default fall back to ``GPT_5.5_xhigh`` with a warning. Flash replay defaults
to GPT-5.5. An explicit version overrides model selection; it does not change
the running model or reasoning effort. The effort in a directory name records
how that memory was generated.

.. code-block:: bash

   # Choose Astra memory automatically.
   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-6-astra --reasoning-effort low

   # Use the same model with the GPT-5.5 corpus.
   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-6-astra --reasoning-effort low \
     --memory-version GPT_5.5_xhigh

CLI and Dashboard resolve the root before each task. In Dashboard, **Next task
model** changes the model for the next task and reselects auto memory then;
the active task keeps its existing model and corpus. A manually selected
memory version remains selected across model changes.

Only the chosen version is downloaded. LIBERO caches are isolated by repository,
commit and version under ``memory/libero/.versions/``. Every file is verified
before cache reuse. ``HF_HUB_OFFLINE=1`` requires a complete, unchanged cache
for the selected version and revision; failed downloads never substitute
another model's corpus. Missing or incomplete caches fail explicitly.
Other robots retain their existing optional-memory sync behavior.

Standalone download and local evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   rpent-memory sync --robot libero --memory-version GPT_6_astra_low
   rpent-memory sync --robot libero --model gpt-6-astra \
     --revision <release-commit> --output-dir /path/to/new-astra-memory
   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-6-astra --reasoning-effort low \
     --memory-profile local --memory-dir /path/to/new-astra-memory

``sync`` prints the actual corpus root. ``--output-dir`` must not already
exist. It also accepts ``--planner`` (default ``codex``) to interpret model
defaults. ``--memory-profile local`` never downloads memory; combining it or
``--explore`` with an explicit remote ``--memory-version`` is an error.
Exploration uses a local corpus; use a separate empty ``--memory-dir`` for
each independent exploration.

Release provenance and compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset's ``libero/README.md`` and ``libero/manifest.json`` document the
versions, source snapshots and SHA-256 hashes. The GPT-5.5 corpus is moved
without changing file contents, including its original ``task_card/`` assets.
Flash reads either ``flash/`` or this legacy name within the selected corpus.
Astra has no replay assets; selecting it for Flash reports an error.

The Astra release merges Long and Spatial/Object/Goal exploration memory,
preserving both versions of three conflicting global notes with source
suffixes. Its 79 task-specific audit/recipe pairs retain their original
content; Long Swap task 6 has no task-specific pair. The historical **741/800**
result used the two original frozen snapshots separately by suite. **The merged
release has not been reevaluated.** Memory was generated at runtime commit
``014a0fa``, before the scene-seed fix. Original snapshots are retained as the
Hub tags ``libero-astra-long-frozen-20260917`` and
``libero-astra-spatial-object-goal-frozen-20260917``.

The loader also supports the old unversioned Hub layout, exclusively as GPT-5.5
memory. The old dataset revision is archived at
``libero-gpt5.5-xhigh-before-versions-20260917``. Clients predating version
selection must upgrade or download that revision and use local memory:

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
     --revision libero-gpt5.5-xhigh-before-versions-20260917 \
     --include 'libero/*' --local-dir /path/to/legacy-download
   # With an older RPent client:
   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-5.5 --memory-profile local \
     --memory-dir /path/to/legacy-download/libero

You can also prepare local memory yourself with the same directory structure
and point the run at it through the environment's ``--memory-dir`` option or
local memory configuration. Hugging Face memory and local memory use the
same directory layout, differing only in where they come from.

Contributing memory
-------------------

Memory on Hugging Face is reviewed and published by RPent maintainers; the
repository ships no self-serve upload path. To contribute a new or updated
memory note, open an RPent issue with the proposed memory file and its
provenance, and a maintainer will review and publish accepted files to
``RLinf/RPent-memory``.
