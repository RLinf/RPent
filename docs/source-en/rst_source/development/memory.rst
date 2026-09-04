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
   |-- suite/
   `-- task_only/
       |-- <cell>.json
       |-- <cell>_recipe.jsonl
       `-- <task_key>.md

The default local root is ``memory/<robot>/``; on the Hugging Face dataset
the same content lives under the ``<robot>/`` subdirectory. A custom
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
