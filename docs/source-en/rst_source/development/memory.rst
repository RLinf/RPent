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
- **Exploration** generates and updates local memory. LIBERO, RoboCasa and
  RoboTwin support it; see the robot guide for its exploration workflow.

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
   |-- task-family/
   `-- task-specific/
       |-- <cell>.json
       |-- <cell>_recipe.jsonl
       `-- <task_key>.md

The default local root is ``memory/<robot>/``. On Hugging Face, LIBERO has
model-specific roots, described below; other robots use ``<robot>/``. A custom
``--memory-dir`` may point at any directory laid out like the tree above.

Each robot provides the memory layers it uses:

.. list-table:: Memory layers
   :header-rows: 1
   :widths: 25 45 30

   * - Layer
     - Stored content
     - Reuse scope
   * - Global Memory (``global/``)
     - Cross-task general rules and failure patterns
     - All tasks
   * - Task-family Memory (``task-family/``)
     - Strategies and precautions validated within a specific task family
     - Similar tasks and their variants
   * - Task-specific Memory (``task-specific/``)
     - Execution records and procedures from a single task run
     - Reference for the current task only

Apply a note only when its prerequisites and evidence limits match the current
task. ``MEMORY.md`` is an optional index for global and task-family notes;
it is not another memory layer. RoboCasa's published HF corpus uses
``global/GLOBAL_MEMORY.md``; a local corpus may provide multiple ``global/*.md`` files.

During evaluation the planner may read only the current robot's memory.
Robot-specific requirements still apply: RoboCasa always provides task-specific
and global memory and requires its global file. Task JSON/recipe files must be
present together or both absent. The planner consults relevant guidance on demand;
robot actions do not require complete reads.

Updating an older corpus
------------------------

The current paths are ``task-specific/`` and ``task-family/``. Download the
matching updated HF corpus into a fresh directory when updating RPent. Family
notes use ``scope: task-family`` and filenames such as
``task-family_libero10_task_t2.md``; their benchmark ``suite`` field is unchanged.
Update local indexes and links when migrating your own notes. An unmigrated
directory is reported as an error rather than treated as missing task memory;
obsolete cached paths are not exposed by the memory tools.

RoboCasa uses task-specific and global memory together, with no layer-selection
option. Keep historical results with the code and validator used to produce
them. The explicit historical v1 manifest remains available; this migration
does not change those records.

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
commit and version under ``memory/libero/.versions/``. Both the exact file set
and every file hash are verified before cache reuse. Extra files invalidate
the cache, including under a pinned revision. Online sync rebuilds an invalid
cache; ``HF_HUB_OFFLINE=1`` requires a complete, unchanged cache
for the selected version and revision; failed downloads never substitute
another model's corpus. Missing or incomplete caches fail explicitly.
Caches created before versioned-source receipts require one successful online
refresh; old unversioned caches are not reused.
Other robots retain their existing synchronization behavior.

Standalone download and local evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m robots.libero.memory sync --memory-version GPT_6_astra_low
   python -m robots.libero.memory sync --model gpt-6-astra \
     --revision <release-commit> --output-dir /path/to/new-astra-memory
   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-6-astra --reasoning-effort low \
     --memory-profile local --memory-dir /path/to/new-astra-memory

``sync`` prints the actual corpus root. ``--output-dir`` must not already
exist. ``--planner`` defaults to ``api``, matching ``rpent``; pass
``--planner codex`` to use ``CODEX_MODEL`` when ``--model`` is omitted. ``--memory-profile local`` never downloads memory; combining it or
``--explore`` with an explicit remote ``--memory-version`` is an error.
Exploration uses a local corpus; use a separate empty ``--memory-dir`` for
each independent exploration.

Versioned downloads are a LIBERO-specific command. The shared ``rpent-memory``
command continues to provide ``merge``, ``validate`` and ``build-index``.

Release provenance and compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset's ``libero/README.md`` and ``libero/manifest.json`` document the
versions, original source snapshots and published files. Each version's
``files`` mapping contains paths relative to that version's root and SHA-256
hashes of the published bytes; the loader verifies this mapping. Source
revisions and source hashes describe the original snapshots and remain separate
from published hashes after directory, index and reference changes.

Both version roots use ``MEMORY.md``, ``global/``, ``task-family/`` and
``task-specific/``. GPT-5.5 additionally includes its ``task_card/`` replay
assets. Flash reads generated ``flash/`` plans or published ``task_card/``
assets within the selected corpus. Astra has no replay assets; selecting it
for Flash reports an error.

The Astra release merges Long and Spatial/Object/Goal exploration memory,
preserving both versions of three conflicting global notes with source
suffixes. Its 79 task-specific audit/recipe pairs retain their original
content; Long Swap task 6 has no task-specific pair. The historical **741/800**
result used the two original frozen snapshots separately by suite. **The merged
release has not been reevaluated.** Memory was generated at runtime commit
``014a0fa``, before the scene-seed fix. Original snapshots are retained as the
Hub tags ``libero-astra-long-frozen-20260917`` and
``libero-astra-spatial-object-goal-frozen-20260917``.

The current loader requires a versioned Hub layout and does not convert or
fall back to the historical unversioned corpus. Update code and data together.
The current directory names require the shared memory naming update in
`RPent #190 <https://github.com/RLinf/RPent/pull/190>`_ and the corresponding
`dataset update <https://huggingface.co/datasets/RLinf/RPent-memory/discussions/13>`_.
Dataset ``main`` evolves; ``reproduce/memory`` keeps its historical contents
and layout for the matching historical robot reproduction branches.
Historical reproduction uses the matching historical client and dataset
revision, archived at ``libero-gpt5.5-xhigh-before-versions-20260917``:

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
