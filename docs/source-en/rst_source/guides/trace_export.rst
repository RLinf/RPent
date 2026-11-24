.. _code-policy-trace-export:

Code-Policy Trace Export
========================

``tool-trace_export`` validates recorded code-policy traces offline and exports
audited, model-agnostic dataset rows (``rows.jsonl``), content-addressed PNG
images, and an export summary (``export-audit.json``). It does not render
prompts.

.. note::

   RPent does not yet include an online recorder that writes
   ``rpent.code_policy_trace.v1`` traces. The exporter currently reads trace
   directories produced in that format externally or by future code-policy
   recording hooks.

Install
-------

``tool-trace_export`` installs as a standalone component containing core
``rpent`` and ``pillow>=10``, with no robot, simulator, or model service:

.. code-block:: bash

   rpent-install install tool-trace_export

From a fresh checkout where ``rpent-install`` is not yet on ``PATH``, run:

.. code-block:: bash

   ./rpent-install.sh install tool-trace_export --venv /path/to/venv

Trace Input Format
------------------

The exporter reads episode directories laid out as shown below, containing
``episode.json``, ``turns.jsonl``, and ``images/``:

.. code-block:: text

   /path/to/traces/<episode_id>/
   ├── episode.json
   ├── turns.jsonl
   └── images/
       └── <image_file>.png

``episode.json`` must contain the five required keys below and may optionally
include ``robo_methods``. Every object in ``turns.jsonl`` must match the exact
key set listed below. Missing required keys, unknown keys, or non-finite
numbers such as ``NaN`` or ``Infinity`` cause the exporter to reject the
episode.

.. list-table:: ``episode.json`` identifies the episode, allowed methods, and outcome
   :header-rows: 1

   * - Field
     - Meaning and requirement
   * - ``schema``
     - Must be ``rpent.code_policy_trace.v1``.
   * - ``episode_id``
     - Non-empty unique identifier for this episode.
   * - ``robot``
     - Non-empty robot identifier.
   * - ``model``
     - Non-empty model identifier.
   * - ``verdict``
     - Object with exact keys ``accepted``, ``truth_source``, and
       ``stop_reason``.
   * - ``robo_methods``
     - Optional non-empty list of unique public Python identifiers allowed on
       ``robo.<method>``. Required here unless ``--robo-method`` is passed as
       fallback.

.. list-table:: ``turns.jsonl`` (one JSON object per turn, in order)
   :header-rows: 1

   * - Field
     - Meaning and requirement
   * - ``turn``
     - 1-based contiguous integer matching the 1-based line position.
   * - ``system``
     - Non-empty rendered system prompt seen by the planner.
   * - ``user``
     - Non-empty list of ``{"type": "text", "text": "..."}`` and
       ``{"type": "image", "image": <index>}`` parts referencing every entry
       in ``images`` exactly once.
   * - ``images``
     - List of image references for this turn.
   * - ``code``
     - Non-empty Python cell source submitted by the model.
   * - ``executed_code``
     - Executed Python source; must equal ``code``.
   * - ``accepted``
     - Must be ``true``.
   * - ``interrupted``, ``timed_out``, ``namespace_reset``
     - All three flags must be ``false``.

.. list-table:: Each entry in ``turns.jsonl`` ``images``
   :header-rows: 1

   * - Field
     - Meaning and requirement
   * - ``path``
     - Non-empty relative path inside ``images/`` (paths escaping
       ``images/`` are rejected).
   * - ``png_sha256``
     - SHA-256 hex digest of the PNG file bytes.
   * - ``rgb_sha256``
     - SHA-256 hex digest of the decoded RGB pixel bytes.
   * - ``width``, ``height``
     - Positive integers matching the decoded RGB PNG dimensions.

Run the Exporter
----------------

From the environment where ``tool-trace_export`` is installed, pass a parent
directory containing episode subdirectories (or one or more individual episode
directories) together with a new ``-o`` / ``--output`` directory:

.. code-block:: bash

   python -m components.tools.trace_export /path/to/traces -o /path/to/trace-dataset

If the episodes in ``/path/to/traces`` do not record ``robo_methods`` in
``episode.json``, pass ``--robo-method`` as fallback (accepting comma-separated
names or repeated flags), and optionally pass multiple source paths directly:

.. code-block:: bash

   python -m components.tools.trace_export \
     /path/to/traces/episode_001 \
     /path/to/traces/episode_002 \
     -o /path/to/trace-dataset \
     --robo-method move_pose,set_gripper,finish

- ``SOURCE``: One or more episode directories or parent directories to search
  recursively for trace v1 episodes (required).
- ``-o``, ``--output``: New output directory outside every ``SOURCE`` path
  (required; fails if the directory already exists).
- ``--robo-method``: Optional fallback public ``robo`` method name(s) used when
  an episode's ``episode.json`` omits ``robo_methods``; accepts
  comma-separated values or repeated flags. When ``episode.json`` specifies
  ``robo_methods``, that episode's own list takes precedence.
- ``--include-unsuccessful``: Optional flag to also export episodes without an
  environment-confirmed success, marked with ``environment_success: false``
  instead of quarantining them.

The exporter applies fail-closed validation and AST projection before writing
output:

- **Episode-level atomicity**: Any error on any turn quarantines the entire
  episode without writing any rows or images for it. Duplicate
  ``episode_id`` values are also quarantined.
- **Success criterion**: By default, ``verdict.accepted`` must be ``true``
  and ``verdict.truth_source`` must be ``"environment"`` (exported with
  ``environment_success: true``). Other episodes are quarantined unless
  ``--include-unsuccessful`` is set (exported with
  ``environment_success: false``).
- **AST annotation projection** (``robo-explanation-strip.v1``): Strips
  ``explanation`` keyword arguments on unaliased public ``robo.<method>``
  calls (literal strings of 1–800 characters and forwarded stable function
  parameters, while preserving evaluation of f-string explanations) and literal
  ``explanation`` keys inside ``robo.parallel`` child action dictionaries.
  Rejects shadowed or aliased ``robo`` bindings, unlisted ``robo`` methods,
  unknown ``**kwargs``, dynamic calls (``exec``, ``eval``, ``globals``,
  ``locals``, ``compile``, ``__import__``), or subscript reads of
  ``["explanation"]``.
- **Persistent namespace check**: Verifies across turns that every referenced
  name is defined by an earlier cell, the current cell, Python builtins, or
  the injected ``robo`` name.
- **Exit status**: Returns ``0`` only when at least one row is exported and
  ``quarantined_episodes`` is empty; returns ``1`` on setup failure, zero
  exported rows, or any quarantined episode.

Output Artifacts
----------------

A completed export writes ``rows.jsonl``, ``export-audit.json``, and
deduplicated PNG files under ``--output``:

.. code-block:: text

   /path/to/trace-dataset/
   ├── rows.jsonl
   ├── export-audit.json
   └── images/
       └── <png_sha256>.png

- ``rows.jsonl``: One ``rpent.trace_export.v1`` JSON object per turn containing
  ``schema``, ``id`` (``<episode_id>:turn_NNNNNN``), ``episode_id``,
  ``split_group``, ``environment_success``, ``messages`` (``system`` and
  ``user`` with rewritten ``images/<png_sha256>.png`` paths), ``target``
  (assistant tool call to ``python`` with projected ``code``), ``images``,
  ``source`` (``episode_dir``, ``robot``, ``model``, ``turn``, and
  ``original_code``), and ``audit`` (projection counts and hashes,
  ``namespace_prefix_cells``, and ``native_verdict``).
- ``images/<png_sha256>.png``: Content-addressed, deduplicated PNG files;
  created only when at least one image is exported, and refuses collisions
  whose bytes differ.
- ``export-audit.json``: Export summary recording ``schema``,
  ``trace_schema``, ``projection``, ``accepted_episodes``, ``rows``,
  ``successful_rows``, ``unsuccessful_rows``, ``quarantined_episodes``,
  ``rows_sha256``, and ``split_policy``
  (``"episode-level only; no independent heldout implied"``).
