Memory and Exploration
======================

Memory lets runs reuse task experience; exploration builds local experience through repeated attempts. Complete :doc:`../quickstart` before following these LIBERO examples. See each environment’s page for its task arguments and success criterion.

Evaluation reads memory without updating it. Exploration can reset and retry. Formal reproduction must use the specified memory revision and evaluation mode.

Use Published Memory
--------------------

Evaluation defaults to the ``hf`` profile, which synchronizes the current robot’s public data from ``RLinf/RPent-memory``. The command makes that default explicit:

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner claude_code --model claude-opus-4-8 \
     --memory-profile hf

Downloads are stored in ``memory/libero/``. Set ``HF_HUB_OFFLINE=1`` to skip synchronization and use the local copy. Network failures are logged as warnings; verify that required data is complete before reproducing results.

Explore and Build Local Memory
------------------------------

This command uses a separate local directory, with up to 3 planner sessions and 5 attempts per session. The directory may start empty; ``--explore`` selects the local profile. Choose a new output directory for later runs:

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 0 \
     --planner claude_code --model claude-opus-4-8 \
     --explore --explore-sessions 3 --explore-attempts-per-session 5 \
     --memory-dir ./memory/libero-local \
     --output-dir logs/explore_10_task_t0_s0

Each session creates a fresh toolkit; attempts within that session start with an environment reset. State and observations are saved under ``<output-dir>/sessions/session_NNN/``, and experience drafts under ``<memory-dir>/_internal/inbox/<cell>/``.

On normal completion, the runner validates and merges drafts and updates the index. It publishes successful task audits and action sequences only when LIBERO reports success. Generated data stays local and is not uploaded automatically. Add ``--dashboard`` to watch exploration; see :doc:`dashboard` for controls.

Evaluate with Local Memory
--------------------------

After exploration, inspect the local corpus and run result, then evaluate another scene with that corpus:

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner claude_code --model claude-opus-4-8 \
     --memory-profile local --memory-dir ./memory/libero-local

The ``local`` profile neither downloads memory nor enables exploration. LIBERO checks that a corpus exists and rejects an empty directory. The ``hf`` profile cannot be combined with ``--memory-dir``.

Inspect memory-read tool calls in the transcript to confirm they use files permitted for the current task. Historical coordinates, pixels, and poses are references; actions still require localization against current observations.

Validate and Merge Manually
---------------------------

Check memory file structure and rebuild the index from existing experience:

.. code-block:: bash

   rpent-memory --memory-dir ./memory/libero-local validate
   rpent-memory --memory-dir ./memory/libero-local build-index

To review drafts before publishing, add ``--no-auto-merge-memory`` to exploration. After review and verification of LIBERO’s success result, merge that cell with the command below. ``--solved`` is a manually supplied success flag; the planner’s summary alone does not justify it:

.. code-block:: bash

   rpent-memory --memory-dir ./memory/libero-local merge \
     --cell 10_task_t0_s0 --output-dir logs/explore_10_task_t0_s0 --solved

Structural validation does not prove real task success. See :doc:`../development/memory` for layout, permissions, and publication rules.

Other Environments
------------------

- :doc:`robocasa` and :doc:`robotwin` support exploration; see their pages for task arguments, resets, and memory scope.
- :doc:`dual_franka` exploration requires an operator to reset the scene and judge results. Automatic merging is disabled by default. Complete deployment and motion checks first.
- :doc:`franka` does not currently support exploration.
