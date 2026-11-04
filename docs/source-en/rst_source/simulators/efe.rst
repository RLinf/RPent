EFE Benchmark
=============

EFE (Embodied First Exam) is a robot task benchmark. RPent joins EFE tasks
through Harbor's installed agent to evaluate planning and tool use without
changing EFE's tasks, simulator, or verifier. The integration is experimental
and EFE-compatible / non-official.

How a Trial Runs
----------------

Harbor calls one task run a trial: it starts one pair of containers for one
task, runs the agent once, and produces one result. Each trial uses two
containers:

.. code-block:: text

   Harbor trial
   ├── agent container (main)
   │   └── RPent / CodexPlanner ── rcb_client ──▶ sim:8000
   └── simulator container (sim)
       └── EFE task service

1. Harbor creates the containers and starts the task in the simulator container.
2. The adapter packages the current checkout's ``rpent/``, ``robots/``,
   installation metadata, and the package data declared in ``pyproject.toml``.
   It uploads them to ``/opt/rpent-efe/source`` in the agent container and
   creates a ``venv`` with the task image's Python. The container does not need
   ``uv``.
3. The runner starts the existing ``CodexPlanner`` inside the agent container as
   the trial's only agent process. ``robots/efe`` connects to the simulator
   service with the task image's ``rcb_client.Robot`` and registers it as
   RPent's robot.
4. The planner operates the episode only through RPent tools. Harbor owns
   container isolation, execution budgets, and teardown; EFE's verifier scores
   the episode independently.

RPent starts no simulator, reads no privileged state, and changes neither the
task nor the scoring. ``finish(status="completed")`` means the agent finished
its own work, not that the task succeeded. ``solved()`` trusts only the
service's own end reason, and the verifier still grades the episode. The CLI
exit code reports only whether the runner failed.

Tool Sets: Skills and Raw
-------------------------

Both arms use the same runner, model, budget, and task. Only the action tool
changes.

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Category
     - Default (``skills=true``)
     - Raw (``skills=false``)
   * - Reading and control
     - ``efe_spec``, ``efe_observe``, ``efe_status``, ``efe_recover``
     - Same
   * - Action
     - ``move_to_xyz``, ``set_gripper``
     - ``efe_submit``
   * - Conditional
     - ``efe_reset`` with a reset budget, ``efe_end_episode`` when success is hidden
     - Same

``efe_reset`` abandons the current episode, restarts from the task's frozen
initial state, and spends one reset from the budget. Only the last episode is
graded; RPent never resets on its own.

The default arm replaces direct joint-target submission with two high-level
skills:

- ``move_to_xyz(x, y, z)`` solves IK for a world-frame target, submits short
  action chunks, and re-reads proprioception until the tool frame is within
  1 cm. It keeps the current end-effector orientation, so the planner can
  reason in Cartesian space.
- ``set_gripper(opening)`` sets the finger opening while holding the current
  arm pose.

The raw arm uses the service interface directly: ``efe_spec`` reports the
meaning and order of each action row's columns (the action layout), and the
planner sends joint-target chunks with ``efe_submit``.

An action layout defines what each number in one action row means. The skills
handle exactly one layout: the leading columns are the joint targets, in
``spec.kinematics.joint_names`` order, and the last column is the gripper
opening. For example:

.. code-block:: text

   joint_names = ["joint1", "joint2", "joint3"]
   layout      = ["q_joint1", "q_joint2", "q_joint3", "gripper"]

After the ``q_`` prefix is removed, the first three columns follow
``joint_names``, and the last column is ``gripper``, so a skill can write its IK
result into these columns. When the layout reorders, omits, or adds columns, a
skill cannot tell which column belongs to which joint and is not registered: the
toolkit keeps ``efe_submit``, records the reason as ``skills_error`` in
``session.json``, and the prompt lists only the tools the session registers.

Supported Tasks
---------------

RPent supports EFE's ``standard`` and ``standard-realtime`` tasks, 1150 in
total:

- 1066 ``standard`` tasks (``eai-standard/2.1``, ``2.2``, ``2.4``).
- 84 ``standard-realtime`` tasks (``eai-standard/2.3``).

In ``standard-realtime``, simulator time keeps advancing while the planner
thinks; the service holds the last targets until the next action arrives, so the
same tools still apply.

The remaining 959 EFE tasks use ``privileged`` mode, which we do not support
yet: the agent shares a container with the simulator, reads the simulated state
directly, and delivers a complete control solution — a replayable
``trajectory.npz`` or a trained policy program/model. The verifier replays the
trajectory or runs the policy in the simulator to grade, with no step-by-step
observation or action. RPent observes and acts step by step through the client,
which does not fit this mode, so it is not integrated yet.

Setup
-----

The host needs the Docker environment required by EFE, cloud model credentials,
and an EFE checkout. Follow EFE's instructions to prepare the host and the task;
simulator requirements come from the task. The first run still downloads the
container and task assets. Keep the original EFE task, simulator, verifier, and
resource limits. The host ``PYTHONPATH`` must import the adapter from this
repository.

.. code-block:: bash

   export RPENT_ROOT=/absolute/path/to/RPent
   export EFE_ROOT=/absolute/path/to/embodied-first-exam
   export EFE_TASK="$EFE_ROOT/tasks/<family>/<task>"
   export EFE_MODEL=openai/your-model
   export OPENAI_API_KEY=your-key

Do not write credentials into the repository. The adapter forwards only model
environment variables and explicit Harbor ``--agent-env`` settings. It maps
``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` to RPent ``CODEX_API_KEY`` and
``CODEX_BASE_URL``; existing ``CODEX_*`` settings take precedence. The adapter
does not forward the host ``CODEX_HOME``. The runner creates that directory
inside the agent container for each trial.

Run One Trial
-------------

.. code-block:: bash

   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
     --agent-kwarg source_root="$RPENT_ROOT" \
     --agent-kwarg timeout_s=3500 --agent-kwarg max_turns=200 \
     --agent-kwarg reasoning_effort=none \
     --jobs-dir "$RPENT_ROOT/logs/efe-rpent"

``RPentEfeAgent`` installs RPent from the current checkout. ``source_root``
defaults to the repository root of the adapter's source checkout; pass it
explicitly when the adapter is imported from an installed wheel. The source
archive excludes the host ``.git``, tests, credential files, and run logs.
``robots/`` is part of the existing RPent distribution and starts no Robot
Runtime.

The default workdir is ``/app``; pass ``--agent-kwarg workdir=/path`` for a task
with a different workdir. Harbor's task budget is always the outer hard limit.
``timeout_s`` is the RPent inner limit and must be below the task's agent
timeout: an equal value lets Harbor time out first and lose the report. RPent
writes ``session.json`` at the end of a trial, so leave enough headroom.
``max_turns`` counts model responses in the planner, not shell commands.

This custom agent declares no model endpoint, so Harbor cannot infer the
inference host. Pass ``--allow-agent-host <model-host>``. An explicit allowlist
replaces the defaults, so also allow the package hosts the install needs:
``--allow-agent-host pypi.org --allow-agent-host files.pythonhosted.org``. The
first container install runs ``pip``, and Harbor's default setup timeout is
360 seconds; add ``--agent-setup-timeout-multiplier 4`` on hosts where a cold
install is slower.

Other options:

- ``--agent-kwarg memory_source=/path`` uploads an existing RPent memory corpus
  in place of the trial's empty memory directory.
- ``--agent-kwarg explore_sessions=N`` runs several independent sessions on the
  same episode.
- ``--agent-kwarg sandbox=full-access`` lets the planner script the simulator
  directly from the shell. The default ``read-only`` keeps shell commands away
  from the simulator, so only RPent tools act on the episode.

Compare Skills and Raw
----------------------

.. code-block:: bash

   # Raw: register the raw tools, including efe_submit
   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
     --agent-kwarg source_root="$RPENT_ROOT" \
     --agent-kwarg timeout_s=3500 --agent-kwarg max_turns=200 \
     --agent-kwarg skills=false \
     --jobs-dir "$RPENT_ROOT/logs/efe-raw"

   # Skills: the default arm, with move_to_xyz and set_gripper
   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
     --agent-kwarg source_root="$RPENT_ROOT" \
     --agent-kwarg timeout_s=3500 --agent-kwarg max_turns=200 \
     --jobs-dir "$RPENT_ROOT/logs/efe-skills"

Keep the model, endpoint, reasoning effort, task revision, time limit, and
attempt count fixed across arms. A single trial is not a comparison.

Result Locations
----------------

``--jobs-dir`` is Harbor's job root; by default each ``harbor run`` creates a
timestamped job directory below it.

.. code-block:: text

   <jobs-dir>/<job_name>/
   ├── result.json              # Job summary
   └── <trial_name>/
       ├── result.json          # Verifier result, tokens, wall time
       ├── agent/
       │   ├── setup/           # Agent install log
       │   └── rpent/
       │       ├── session.json # Agent execution metrics, errors, per-session stats
       │       ├── codex*.txt   # Planner transcript
       │       ├── codex-home/  # CODEX_HOME for this trial
       │       └── memory/      # Memory root for this trial
       ├── verifier/            # Verifier logs and reward.json
       ├── artifacts/           # Collected artifacts
       ├── config.json
       ├── lock.json
       └── trial.log

The container path ``/logs/agent/rpent/`` maps to
``<trial_name>/agent/rpent/`` on the host. EFE's verifier writes
``verifier/reward.json``, and Harbor records it as
``verifier_result.rewards`` in ``result.json``; ``success`` is the task success
flag and the other fields are diagnostics. Agent execution metrics are in
``stats`` and each session's ``metrics`` in ``agent/rpent/session.json``. An
incomplete trial may have no ``result.json`` or ``session.json``; check
``trial.log`` and ``agent/rpent/run.log`` first.

Isolation and Measurement Mode
------------------------------

- Each trial starts with an empty ``memory/`` and its own ``codex-home/``;
  ``memory_source`` replaces the empty memory. Model and credentials come only
  from explicit environment variables.
- The runner defaults to the Codex ``read-only`` sandbox, so only RPent tools act
  on the episode. ``full-access`` measures the agent's free-form behavior; shell
  actions do not appear in the agent execution metrics.
- The generic ``rpent --robot efe`` entry point passes no sandbox setting and
  always runs with ``full-access``; use the Harbor adapter for evaluation and
  comparisons.
