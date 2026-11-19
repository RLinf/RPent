Embodied First Exam
===================

Run `Embodied First Exam (EFE) <https://www.embodied-first-exam.ai/>`_ tasks
with RPent through `Harbor <https://github.com/laude-institute/harbor>`_. This
guide walks through a CPU-only AutoBio laboratory task on a Linux host with
Docker and a model API key; start with `Run a Task`_ to run one evaluation.

Overview
--------

EFE packages robot tasks with fixed scenes, an HTTP simulator service
(``rcb_client``), and an independent verifier. For each task run (a Harbor
**trial**), Harbor starts the simulator and agent containers, runs RPent,
records the verifier score, and removes the containers.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      None (analytical IK ``Skills`` or native ``Raw`` actions)

   .. grid-item-card:: Planners

      Codex

   .. grid-item-card:: Tasks

      1,066 ``standard``-mode tasks across 36 benchmarks (e.g., AutoBio, RoboARC)

   .. grid-item-card:: Hardware

      Linux, Docker; CPU or NVIDIA GPU (task-dependent)

Tasks
~~~~~

Browse tasks in the `EFE task browser <https://www.embodied-first-exam.ai/tasks/>`_.
Task definitions live under ``tasks/<benchmark>/<task>`` in the
`EFE repository <https://github.com/mll-lab-nu/embodied-first-exam>`_.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Mode
     - Tasks
     - Support and behavior
   * - ``standard``
     - 1066
     - Supported on ``eai-standard/2.2`` and ``2.4``. The simulator steps when the agent sends actions.
   * - ``standard-realtime``
     - 84
     - Not supported (uses ``eai-standard/2.3`` or ``eai-standard-realtime/1.0``).
   * - ``privileged``
     - 959
     - Not supported (expects a complete trajectory or policy file instead of step-by-step control).

.. note::

   Check ``metadata.protocol`` in the task's ``task.toml`` before running. Tasks
   outside ``eai-standard/2.2`` and ``2.4`` fail immediately at connection time.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

RPent provides two control modes via the ``skills`` option (see
`Compare Skills and Raw`_):

- **Skills** (``skills=true``, default): exposes ``move_to_pose`` and
  ``set_gripper``. Falls back to **Raw** automatically if the task's action
  layout or kinematics do not support IK skills.
- **Raw** (``skills=false``): exposes ``efe_submit`` to send native action
  vectors directly.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Item
     - Description
   * - Observation
     - Camera RGB images, optional depth, ``camera_calibration``, ``frame_extrinsics_cv`` (world-to-camera transforms), joint positions, tool pose, gripper opening, and action specification.
   * - Action
     - **Skills**: target tool position, optional ``wxyz`` orientation, and gripper opening. **Raw**: native action vector matching ``efe_spec``.
   * - Reward / success
     - Graded independently by EFE's verifier in ``verifier/reward.json`` and ``result.json`` (see `View Results`_).
   * - Task prompt
     - The task's original ``instruction.md`` passed by Harbor.

Installation and Resources
--------------------------

Prerequisites on the Linux host:

- ``git``, `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, and Docker Engine with Compose v2 and BuildKit (see `EFE host requirements <https://github.com/mll-lab-nu/embodied-first-exam#quick-start>`_).
- NVIDIA driver and NVIDIA Container Toolkit for GPU benchmarks (not needed for the CPU AutoBio example below).
- Python 3.12 and Harbor 0.23.0 (installed in step 2).

1. Clone both repositories (skip if already cloned):

   .. code-block:: bash

      git clone https://github.com/RLinf/RPent.git
      git clone --recurse-submodules https://github.com/mll-lab-nu/embodied-first-exam.git

2. Set repository paths and install EFE and Harbor (`0.23.0`):

   .. code-block:: bash

      export RPENT_ROOT=/absolute/path/to/RPent
      export EFE_ROOT=/absolute/path/to/embodied-first-exam
      uv sync --project "$EFE_ROOT" --python 3.12
      uv pip install --python "$EFE_ROOT/.venv/bin/python" -e "$RPENT_ROOT"
      docker info
      uv run --project "$EFE_ROOT" harbor --version
      test -f "$RPENT_ROOT/robots/efe/harbor_agent.py"

   - ``uv pip install ... -e "$RPENT_ROOT"`` is only needed for manual host runs (``rpent --robot efe``); Harbor trials build and install RPent inside the task container automatically.
   - Verify that ``docker info`` succeeds, ``harbor --version`` reports ``0.23.0``, and ``test -f`` exits with ``0``.

3. Select the example AutoBio centrifuge task and configure model credentials:

   .. code-block:: bash

      export EFE_TASK="$EFE_ROOT/tasks/autobio/autobio-insert-centrifuge-5430-i00-standard"
      export EFE_MODEL=openai/your-model
      export OPENAI_API_KEY=your-key
      export OPENAI_BASE_URL=https://api.openai.com/v1
      export MODEL_HOST=api.openai.com
      test -f "$EFE_TASK/task.toml"

   - ``MODEL_HOST``: bare hostname (no ``https://`` or path) passed to ``--allow-agent-host`` so the container can reach your model endpoint. Update both ``OPENAI_BASE_URL`` and ``MODEL_HOST`` when using a custom endpoint.
   - Optional pre-check: run ``rpent-check-llm --planner codex --model <model> --json`` on the host (requires ``CODEX_API_KEY`` on the host; inside the container, the adapter maps ``OPENAI_*`` to ``CODEX_*`` automatically unless ``CODEX_*`` is already set).

Run a Task
----------

Run a single Harbor trial with the default Skills tools:

.. code-block:: bash

   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" \
     --n-concurrent 1 \
     --allow-agent-host "$MODEL_HOST" \
     --yes \
     --jobs-dir "$RPENT_ROOT/logs/efe-rpent"

- ``--agent``: loads ``RPentEfeAgent`` from ``PYTHONPATH``.
- ``--n-concurrent 1``: runs one trial at a time; ``--yes`` skips interactive confirmation.
- The first run pulls or builds the task image and installs dependencies inside the container. Results are written to ``$RPENT_ROOT/logs/efe-rpent``.

View Results
------------

Each ``harbor run`` creates a timestamped job directory under ``--jobs-dir``:

.. code-block:: text

   <jobs-dir>/<job_name>/
   ├── result.json              # Job summary
   └── <trial_name>/
       ├── result.json          # Verifier result, token usage, wall time
       ├── agent/
       │   ├── setup/           # Container agent install log
       │   └── rpent/           # Mounted from /logs/agent/rpent/ in container
       │       ├── transcript_*.json # CLI transcript and planner stats
       │       ├── efe-result.json   # EFE metrics and finalization data
       │       ├── codex_*.txt       # Planner transcript
       │       ├── codex-home/       # Isolated CODEX_HOME for this trial
       │       └── memory/           # Memory directory for this trial
       ├── verifier/            # Verifier logs and reward.json
       ├── artifacts/           # Collected trial artifacts
       ├── config.json
       ├── lock.json
       └── trial.log

If a trial aborts before writing ``result.json`` or ``transcript_*.json``,
inspect ``trial.log`` and ``agent/rpent/run.log`` first.

Success Criterion
~~~~~~~~~~~~~~~~~

- **Native evaluation score**: read ``verifier_result.rewards.success`` in ``<trial_name>/result.json`` (copied from ``verifier/reward.json``). Tasks with retries also log ``first_success`` for the first attempt, while ``success`` reflects the final attempt.
- Neither ``finish(status="completed")`` nor a ``0`` CLI exit code indicates task success.
- A trial whose planner exhausts its time budget is still graded: the adapter
  treats budget exhaustion as a normal stop, so the verifier checks the final
  state of the last attempt as usual.

Run a Benchmark
---------------

Point ``--path`` at a benchmark directory containing supported ``standard``-mode
tasks (for example, ``$EFE_ROOT/tasks/autobio``):

.. code-block:: bash

   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_ROOT/tasks/<benchmark>" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" \
     --n-concurrent 8 \
     --allow-agent-host "$MODEL_HOST" \
     --yes \
     --jobs-dir "$RPENT_ROOT/logs/efe-benchmark"

- Verify that the benchmark directory contains only supported ``standard``-mode tasks (`eai-standard/2.2` or `2.4`) before launching a batch.
- Keep the model, endpoint, task revision, and timeout fixed when comparing runs.

Common Options
--------------

Pass adapter options via ``--agent-kwarg key=value``:

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Option
     - Default
     - Description
   * - ``skills=false``
     - ``true``
     - Disable IK skills and register only raw action tools (``efe_submit``); see `Compare Skills and Raw`_.
   * - ``reasoning_effort=low``
     - ``none``
     - Codex reasoning effort: ``none``, ``low``, ``medium``, ``high``, or ``xhigh``.
   * - ``max_turns=N``
     - ``200``
     - Maximum planner model turns (not shell command count).
   * - ``memory_source=/path``
     - Unset
     - Upload an existing RPent memory directory into the trial (starts with an empty ``memory/`` when unset).
   * - ``timeout_s=N``
     - Detected
     - RPent trial time budget (auto-detected from Harbor trial config or local ``task.toml``, reserving shutdown/finalization time).
   * - ``connect_timeout_s=N``
     - ``60``
     - Seconds to wait for the EFE simulator service (deducted from the trial budget).
   * - ``agent_timeout_s=N``
     - Detected
     - Explicit Harbor timeout override when task config is not a local directory; match ``--agent-timeout-sec``.
   * - ``sandbox=full-access``
     - ``read-only``
     - Shell sandbox mode. Keep ``read-only`` so the planner acts only through RPent tools; see `Isolation and Measurement Mode`_.
   * - ``workdir=/path``
     - ``/app``
     - Working directory inside the task container.
   * - ``source_root=/path``
     - Auto-detected
     - RPent checkout used to build the wheel (set explicitly if importing ``RPentEfeAgent`` from an installed package).

Task Memory
-----------

- Pass ``--agent-kwarg memory_source=/path`` to seed a trial with an existing
  RPent memory directory; otherwise the trial starts with an empty ``memory/``.
- Each trial runs in an isolated ``codex-home/`` directory using the model and
  credentials forwarded by Harbor.

Compare Skills and Raw
----------------------

Use ``--agent-kwarg skills=false`` to compare native action submission (**Raw**)
against the default IK motion tools (**Skills**):

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Category
     - Default (``skills=true``)
     - Raw (``skills=false``)
   * - Observation & state
     - ``efe_spec``, ``efe_observe``, ``efe_status``, ``efe_recover``, ``view_env_state``
     - Same
   * - Action
     - ``move_to_pose``, ``set_gripper``
     - ``efe_submit``
   * - Conditional
     - ``efe_reset`` (retry from the start state when the task allows retries), ``efe_end_episode`` (stop and submit the final state when the task does not auto-terminate)
     - Same

- **Skills** (``move_to_pose(x, y, z, orientation_wxyz=None)``, ``set_gripper(opening)``): solves inverse kinematics and sends closed-loop joint chunks until position error ``<= 1 cm`` and orientation error ``<= 0.05 rad`` (up to 20 iterations by default, max 50). Requires action columns ordered as ``[*spec.kinematics.joint_names, gripper]`` and a tool frame present in both ``tcp_frames`` and ``ik_targets``; otherwise the toolkit falls back to ``efe_submit`` and records ``skills_error`` in ``efe-result.json``.
- **Raw** (``efe_submit``): sends joint-target chunks matching the column layout returned by ``efe_spec``.

.. code-block:: bash

   # Raw mode (efe_submit)
   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 \
     --allow-agent-host "$MODEL_HOST" \
     --agent-kwarg skills=false \
     --yes \
     --jobs-dir "$RPENT_ROOT/logs/efe-raw"

   # Skills mode (default: move_to_pose, set_gripper)
   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 \
     --allow-agent-host "$MODEL_HOST" \
     --yes \
     --jobs-dir "$RPENT_ROOT/logs/efe-skills"

Manual Run (Debugging)
----------------------

Connect ``rpent`` directly to an already-running EFE task service to debug
prompts or tool calls outside Harbor (this does not start a simulator or run
the verifier):

.. code-block:: bash

   rpent --robot efe \
     --instruction-file "$EFE_TASK/instruction.md" \
     --planner codex --model <model> --reasoning-effort low \
     --codex-sandbox read-only \
     --memory-profile local --memory-dir ./memory/efe \
     --output-dir logs/efe-manual

- ``--instruction-file``: path to the task's ``instruction.md`` (derives ``--task-name`` from the parent directory when omitted).
- ``--model``: bare model name without the ``openai/`` prefix.
- ``--efe-base-url``: service URL (probes ``sim:8000`` then ``localhost:8000`` when omitted).
- ``--efe-connect-timeout-s`` / ``--efe-trial-timeout-s``: connection wait limit (default ``3600`` s) and total trial deadline.
- ``--efe-no-skills``: register only Raw tools.

Isolation and Measurement Mode
------------------------------

- **Harbor default (``sandbox=read-only``)**: blocks shell file writes and
  network access so all robot actions go through RPent tools and are counted in
  ``efe-result.json``.
- **Unrestricted shell (``sandbox=full-access``)**: allows shell scripts to call
  ``rcb_client`` directly, bypassing RPent tools and metrics. Do not mix with
  ``read-only`` runs in comparisons.
- When debugging via the generic ``rpent`` CLI, pass ``--planner codex
  --codex-sandbox read-only`` explicitly to match Harbor's defaults.

Reproduction
------------

To reproduce a comparison, pin both the RPent commit and the EFE repository
commit, keep the task directory, model, ``reasoning_effort``, timeout, and
attempt count identical across runs, and archive ``result.json`` and
``config.json`` from the Harbor job directory.

Next Steps
----------

- Choose and configure the model and planner: :doc:`../guides/configure_planner`
- Full generic CLI reference: :doc:`../guides/cli`
- Task memory and exploration: :doc:`../guides/memory`
- Add a new robot integration: :doc:`../development/add_robot`
