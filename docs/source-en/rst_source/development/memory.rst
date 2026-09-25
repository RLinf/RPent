Memory Management
=================

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
   |-- task-family/
   `-- task-specific/
       |-- <cell>.json
       |-- <cell>_recipe.jsonl
       `-- <task_key>.md

The default local root is ``memory/<robot>/``; on the Hugging Face dataset
the same content lives under the ``<robot>/`` subdirectory. A custom
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
it is not another memory layer. RoboCasa reads its single global file directly.

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

By default RPent syncs the current robot's memory from the Hugging Face
dataset ``RLinf/RPent-memory`` into ``memory/<robot>/``. The dataset is
public, so a fresh clone downloads it without a token. Set
``HF_HUB_OFFLINE=1`` to skip the sync and use the local copy only. Memory is
optional: if a robot has none on the dataset, or the sync fails, the run
continues with whatever is on disk.

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
