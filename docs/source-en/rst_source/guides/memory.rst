.. _memory-management:

Memory and Exploration
======================

Memory lets runs reuse task experience; exploration builds local experience through repeated attempts. Complete :doc:`../get_started/quickstart` before following these LIBERO examples. See each environment’s page for its task arguments and success criterion.

Evaluation reads memory without updating it. Exploration can reset and retry. Formal reproduction must use the specified memory revision and evaluation mode.

Use Published Memory
--------------------

Evaluation defaults to ``--memory-profile hf`` and downloads memory from the public
`RLinf/RPent-memory dataset <https://huggingface.co/datasets/RLinf/RPent-memory>`_.
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

For a reproducible LIBERO evaluation, pin the Hub commit without switching
to the local-memory profile:

.. code-block:: bash

   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-6-astra --reasoning-effort low \
     --memory-revision <commit-sha>

``--memory-revision`` accepts a commit, tag or branch. Omitting it follows
``main`` as before; use a full commit SHA to keep the source fixed across
runs. It applies to the selected model-specific corpus and to subsequent
Dashboard tasks, including tasks that select another model. Local-memory evaluation and exploration reject
this remote-only option.

Before each task starts, LIBERO writes a compact ``memory_source.json`` in the
output directory. Dashboard shared services may already be running at that point.
The record contains the robot, the HF dataset URL pinned to the resolved commit,
and the selected corpus version. File hashes remain in the verified cache receipt;
they are not duplicated in this portable record. Local runs record only a local
directory and cannot be replayed as a verified HF source. Use a fresh output
directory for each independent run.

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

When the Hub cannot be reached, a full commit SHA selects that exact cached
commit, including one previously downloaded through a branch. A branch or tag
selects the commit last successfully synchronized for that name, repository and
corpus version on this machine; it may be older than the current Hub revision.
Check ``memory_source.json`` for the commit actually used. To reuse that source in another run, pass the saved file with ``--memory-source``.

Shared Source Selection and Replay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All robot CLI and Dashboard entry points accept ``--memory-repo`` and
``--memory-source`` in HF evaluation mode. For example:

.. code-block:: bash

   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-6-astra --reasoning-effort low \
     --memory-repo https://huggingface.co/datasets/RLinf/RPent-memory@<commit-sha>

   rpent --robot libero --suite libero_goal_swap --task 1 --seed 2 \
     --planner codex --model gpt-6-astra --reasoning-effort low \
     --memory-source /path/to/previous-run/memory_source.json

``--memory-repo`` accepts a dataset ID or an HTTPS Hugging Face dataset URL,
optionally followed by ``@revision`` (default ``main``). The repository must keep
the selected robot's existing layout: LIBERO retains its versioned manifest and
checksum verification; other robots load their ``<robot>/**`` subtree from the
resolved Hub snapshot and retain their existing task/layout checks. Explicit
sources fail if the requested corpus is unavailable, rather than using the
robot's default memory directory. Arbitrary Git remotes are not supported.

``--memory-source`` (also accepted as ``--memory_source``) reads schema version 2
records. It pins the saved repository, commit and, for LIBERO, corpus version,
independently of the next task's model. It rejects a different robot, local-only
records, mutable revisions and conflicting source flags. Older schema version 1
records are not replay inputs; use their recorded repository and resolved commit
with ``--memory-repo``. Do not combine these flags with exploration, local memory,
``--memory-revision``, or each other; replay also rejects an explicit
``--memory-version``. An explicit source takes precedence over
``RPENT_MEMORY_HF_REPO``. Downloads follow the host's existing HF authentication.

The shared source parser and record format live in ``rpent/memory/source.py``.
Robot-specific corpus verification remains with each robot. A pinned URL fixes
source identity, not the rest of the execution environment.

Standalone Download and Local Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m robots.libero.memory sync --memory-version GPT_6_astra_low
   python -m robots.libero.memory sync --model gpt-6-astra \
     --revision <release-commit> --output-dir /path/to/new-astra-memory \
     --source-record /path/to/astra-source.json
   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-6-astra --reasoning-effort low \
     --memory-profile local --memory-dir /path/to/new-astra-memory

``sync`` prints the actual corpus root. ``--output-dir`` must not already
exist. Exported bytes are checked against the verified receipt before the new
directory is published. Optional ``--source-record`` writes the same HF source
record. Its remote source can be replayed on another machine; the exported
directory remains selectable separately with ``--memory-profile local --memory-dir``. It must be outside
both the cache and the output corpus so it cannot invalidate the file manifest.
``--planner`` defaults to ``api``, matching ``rpent``; pass
``--planner codex`` to use ``CODEX_MODEL`` when ``--model`` is omitted. ``--memory-profile local`` never downloads memory; combining it or
``--explore`` with an explicit remote ``--memory-version`` is an error.
Exploration uses a local corpus; use a separate empty ``--memory-dir`` for
each independent exploration.

Versioned downloads are a LIBERO-specific command. The shared ``rpent-memory``
command continues to provide ``merge``, ``validate`` and ``build-index``.

Release Provenance and Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The loader's ref pointers are scoped to repository, requested ref and corpus
version. Offline branch/tag resolution validates the pointer's repository,
version and commit format before verifying the snapshot. Full commit requests
address the snapshot directly and do not use ref pointers, so a stale or modified
pointer cannot redirect a pinned request. The compact ``memory_source.json`` stores the resolved immutable source URL.
The requested ref remains visible in the launch command and run log.

The dataset's ``libero/README.md`` and ``libero/manifest.json`` document the
versions, original source snapshots and published files. Each version's
``files`` mapping contains paths relative to that version's root and SHA-256
hashes of the published bytes; the loader verifies this mapping. Source
revisions and source hashes describe the original snapshots and remain separate
from published hashes after directory, index and reference changes.

Both version roots use ``MEMORY.md``, ``global/``, ``task-family/`` and
``task-specific/``. GPT-5.5 additionally includes 78 plan/anchor pairs in
``flash/``, copied unchanged from the merged Flash release. Flash reads that
directory within the selected corpus. Astra has no replay assets; selecting it
for Flash reports an error.

The Astra release merges Long and Spatial/Object/Goal exploration memory,
preserving both versions of three conflicting global notes with source
suffixes. Its 79 task-specific audit/recipe pairs retain their original
content; Long Swap task 6 has no task-specific pair. The historical **741/800**
result used the two original frozen snapshots separately by suite. **The merged
release has not been reevaluated.** Memory was generated at runtime commit
``014a0fa``, before the scene-seed fix. The original source revisions are
``cf5d14ce9b3ec6c72de5477ec0fea806a3884efa`` (Long) and
``984c57f7c6caf48b572f5926851dd9134ae041d8`` (Spatial/Object/Goal).

The current loader requires a versioned Hub layout and does not convert or
fall back to the historical unversioned corpus. Update code and data together.
The current directory names require the shared memory naming update in
`RPent #190 <https://github.com/RLinf/RPent/pull/190>`_ and the corresponding
`dataset update <https://huggingface.co/datasets/RLinf/RPent-memory/discussions/13>`_.
Dataset ``main`` evolves; ``reproduce/memory`` keeps its historical contents
and layout for the matching historical robot reproduction branches.
Historical GPT-5.5 reproduction uses the matching historical client and the
original dataset revision ``21a62795fe3b7e500c8381ac47938f6d713ebe18``:

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
     --revision 21a62795fe3b7e500c8381ac47938f6d713ebe18 \
     --include 'libero/*' --local-dir /path/to/legacy-download
   # With an older RPent client:
   rpent --robot libero --suite libero_goal_swap --task 1 --seed 1 \
     --planner codex --model gpt-5.5 --memory-profile local \
     --memory-dir /path/to/legacy-download/libero

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

- :doc:`../simulators/robocasa` and :doc:`../simulators/robotwin` support exploration; see their pages for task arguments, resets, and memory scope.
- :doc:`../real_world_robots/dual_franka` exploration requires an operator to reset the scene and judge results. Automatic merging is disabled by default. Complete deployment and motion checks first.
- :doc:`../real_world_robots/franka` does not currently support exploration.
