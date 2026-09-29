Memory Design
=============

RPent uses ``MemoryManager`` to manage each robot’s experience files, access permissions, and exploration drafts. See :doc:`../guides/memory` for commands; this page describes storage and publication.

Layout and Access Scope
-----------------------

Memory published to Hugging Face and memory prepared locally for evaluation
use the same directory structure:

.. code-block:: text

   <memory-root>/
   ├── MEMORY.md
   ├── global/
   ├── task-family/
   ├── task-specific/
   │   ├── <cell>.json
   │   ├── <cell>_recipe.jsonl
   │   └── <task_key>.md
   └── _internal/inbox/<cell>/

The default local root is ``memory/<robot>/``. On Hugging Face, LIBERO has
model-specific roots, described in :doc:`../guides/memory`; other robots use ``<robot>/``. A custom
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

Each toolkit constructs a ``MemoryManager`` from the run configuration: evaluation uses read-only access, while exploration allows writes to the current cell’s inbox. Environment tool permissions and task rules further constrain reads. LIBERO local evaluation checks that a corpus exists.

Updating an Older Corpus
~~~~~~~~~~~~~~~~~~~~~~~~

The current paths are ``task-specific/`` and ``task-family/``. Download the
matching updated HF corpus into a fresh directory when updating RPent. Family
notes use ``scope: task-family`` and filenames such as
``task-family_libero10_task_t2.md``; their benchmark ``suite`` field is unchanged.
Update local indexes and links when migrating your own notes. An unmigrated
directory is reported as an error rather than treated as missing task memory;
obsolete cached paths are not exposed by the memory tools.

RoboCasa uses task-specific and global memory together, with no layer-selection
option. Keep historical results with the code and validator used to produce
them. The historical v1 manifest remains available for validating compatible records.

Exploration and Merging
-----------------------

Exploration preserves state and tool records for its attempts and produces experience drafts with provenance. The runner determines success from the environment and calls ``MemoryManager.merge_memory`` to merge drafts and update the index; successful task audits and action sequences have separate publication conditions.

Simulation environments merge automatically by default; ``--no-auto-merge-memory`` retains drafts for review. Dual-arm Franka disables automatic merging by default and relies on operator judgments. In a manual ``rpent-memory merge``, ``--solved`` is a caller-supplied success flag; use it only after verifying the environment or operator result.

``rpent-memory validate`` checks file structure, not real task success. See ``rpent/memory/manager.py`` for permissions and merging, and each robot’s ``robot_spec.py`` plus ``rpent/cli/main.py`` for run modes and result handling.

Synchronization and Contributions
---------------------------------

The HF profile synchronizes memory from the public ``RLinf/RPent-memory`` dataset. LIBERO selects and verifies a model-specific corpus; offline use requires a complete cache for that version and revision. See :doc:`../guides/memory` for downloads, cache requirements, and historical code/data compatibility.

Maintainers review and publish public memory. To contribute, open an RPent issue with the memory files, code and model versions, task arguments, and success evidence. The repository has no automatic memory-upload entry point.
