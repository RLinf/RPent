Memory Design
=============

RPent uses ``MemoryManager`` to manage each robot’s experience files, access permissions, and exploration drafts. See :doc:`../usage/memory` for commands; this page describes storage and publication.

Layout and Access Scope
-----------------------

Local memory normally lives in ``memory/<robot>/``, or a directory selected with ``--memory-dir``. Layered memory uses the following structure; layers are optional:

.. code-block:: text

   <memory-root>/
   ├── MEMORY.md
   ├── global/
   ├── suite/
   ├── task_only/
   └── _internal/inbox/<cell>/

``global/`` holds general lessons, ``suite/`` organizes experience by suite, ``task_only/`` stores same-task references, and ``MEMORY.md`` indexes searchable experience. Exploration drafts go to ``_internal/inbox/<cell>/``.

Published corpora can also use environment-specific layouts. RoboCasa Target50 uses ``results/<Task>_s0.json``, ``recipe_<Task>_s0.jsonl``, and optional task Markdown; see :doc:`../usage/robocasa`. Do not assume one environment’s layout applies to every corpus.

Each toolkit constructs a ``MemoryManager`` from the run configuration: evaluation uses read-only access, while exploration allows writes to the current cell’s inbox. Environment tool permissions and task rules further constrain reads. LIBERO local evaluation checks that a corpus exists.

Exploration and Merging
-----------------------

Exploration preserves state and tool records for its attempts and produces experience drafts with provenance. The runner determines success from the environment and calls ``MemoryManager.merge_memory`` to merge drafts and update the index; successful task audits and action sequences have separate publication conditions.

Simulation environments merge automatically by default; ``--no-auto-merge-memory`` retains drafts for review. Dual-arm Franka disables automatic merging by default and relies on operator judgments. In a manual ``rpent-memory merge``, ``--solved`` is a caller-supplied success flag; use it only after verifying the environment or operator result.

``rpent-memory validate`` checks file structure, not real task success. See ``rpent/memory/manager.py`` for permissions and merging, and each robot’s ``robot_spec.py`` plus ``rpent/cli/main.py`` for run modes and result handling.

Synchronization and Contributions
---------------------------------

The HF profile synchronizes the current robot’s directory from the public ``RLinf/RPent-memory`` dataset; ``HF_HUB_OFFLINE=1`` skips synchronization. For reproduction, download the revision specified in the environment guide and select the local profile.

Maintainers review and publish public memory. To contribute, open an RPent issue with the memory files, code and model versions, task arguments, and success evidence. The repository has no automatic memory-upload entry point.
