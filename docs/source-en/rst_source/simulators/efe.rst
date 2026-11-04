Embodied First Exam
===================

Run robot tasks with RPent in EFE (Embodied First Exam). EFE starts an agent
container and a simulator container through Harbor; RPent joins as an installed
agent and operates the robot through the task's ``rcb_client``.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Agent

      RPent ``CodexPlanner``, which acts only through RPent tools.

   .. grid-item-card:: Tool Layers

      ``move_to_xyz`` and ``set_gripper`` by default; ``efe_submit`` in the raw arm.

   .. grid-item-card:: Tasks

      EFE ``standard`` and ``standard-realtime``, 1150 tasks.

   .. grid-item-card:: Runtime

      Docker; Harbor 0.23.0; Python 3.12.

Harbor calls one task run a trial: it starts one pair of containers for one
task, runs the agent once, and produces one result.

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
task nor the scoring. ``finish(status="completed")`` means the agent finished its
own work, not that the task succeeded. ``solved()`` trusts only the service's own
end reason, and the verifier still grades the episode. The CLI exit code reports
only whether the runner failed.

Tasks
~~~~~

EFE groups its tasks into suites, each covering one kind of robot capability.
The official EFE content defines the task list, scenes, and scoring rules:
browse every task in the `EFE task browser
<https://www.embodied-first-exam.ai/tasks/>`_, and find each definition under
``tasks/<benchmark>/<task>`` in the `EFE repository
<https://github.com/mll-lab-nu/embodied-first-exam>`_. RPent integrates the
``standard`` and ``standard-realtime`` tasks.

.. list-table::
   :header-rows: 1

   * - Mode
     - Tasks
     - Notes
   * - ``standard``
     - 1066
     - ``eai-standard/2.1``, ``2.2``, ``2.4``; the simulator advances by control step.
   * - ``standard-realtime``
     - 84
     - ``eai-standard/2.3``; simulator time keeps advancing while the planner thinks.
   * - ``privileged``
     - 959
     - Not supported yet; the deliverable is a complete control solution.

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

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Item
     - Description
   * - Observation
     - Camera RGB frames (with depth on request), robot state (joint angles, tool position and orientation, gripper opening), and the action layout, joint names, cameras, and kinematics. In the default configuration the agent reaches the simulator only through the client.
   * - Action
     - The default arm calls ``move_to_xyz`` to move the tool tip (TCP) to a world position and ``set_gripper`` to set the finger opening; the raw arm sends joint-target chunks with ``efe_submit``. Both reach the simulator service through ``rcb_client``.
   * - Reward / success
     - EFE's verifier scores the episode independently and writes ``verifier/reward.json``; Harbor records it as ``verifier_result.rewards`` in ``result.json``. ``success`` is this trial's graded result; ``first_success`` records the first attempt; the other fields are diagnostics.
   * - Task instruction
     - Harbor reads the instruction from the EFE task and passes it to the agent; RPent does not change the task text.

A task that declares a reset budget registers ``efe_reset``: it abandons the
current episode, restarts from the task's frozen initial state, and spends one
reset from the budget. Only the last episode is graded; RPent never resets on
its own. A task that hides success also registers ``efe_end_episode``.

Installation and Resources
--------------------------

The host needs the Docker environment required by EFE, cloud model API keys,
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
   export EFE_MODEL_HOST=<model-host>
   export OPENAI_API_KEY=your-key

Do not write API keys into the repository. The adapter forwards only model
environment variables and explicit Harbor ``--agent-env`` settings. It maps
``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` to RPent ``CODEX_API_KEY`` and
``CODEX_BASE_URL``; existing ``CODEX_*`` settings take precedence. The adapter
does not forward the host ``CODEX_HOME``. The runner creates that directory
inside the agent container for each trial.

Run a Task
----------

Harbor's ``--model`` selects the model, and the API key is read from the
current environment. If the task restricts network access during agent
execution, add ``--allow-agent-host "$EFE_MODEL_HOST"``; this option extends the
existing allowlist during agent execution only. RPent installation runs during
setup and uses the task environment's baseline network policy. The first
container install runs ``pip``, and Harbor's default setup timeout is 360
seconds; add ``--agent-setup-timeout-multiplier 4`` on hosts where a cold
install is slower. Set ``EFE_MODEL_HOST`` to the host that serves
``EFE_MODEL``. This command runs one trial:

.. code-block:: bash

   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
     --allow-agent-host "$EFE_MODEL_HOST" \
     --agent-kwarg source_root="$RPENT_ROOT" \
     --agent-kwarg timeout_s=3500 --agent-kwarg max_turns=200 \
     --agent-kwarg reasoning_effort=none \
     --jobs-dir "$RPENT_ROOT/logs/efe-rpent"

``RPentEfeAgent`` installs RPent from the current checkout. ``source_root``
defaults to the repository root of the adapter's source checkout; pass it
explicitly when the adapter is imported from an installed wheel. The source
archive excludes the host ``.git``, tests, API key files, and run logs.
``robots/`` is part of the existing RPent distribution and starts no Robot
Runtime.

View Results
------------

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
``verifier_result.rewards`` in ``result.json``; ``success`` is this trial's
graded result, ``first_success`` records the first attempt, and the other
fields are diagnostics. Agent execution metrics are in
``stats`` and each session's ``metrics`` in ``agent/rpent/session.json``. An
incomplete trial may have no ``result.json`` or ``session.json``; check
``trial.log`` and ``agent/rpent/run.log`` first.

Common Options
--------------

- ``--agent-kwarg source_root=/path``: the RPent checkout to install; defaults
  to the adapter's repository.
- ``--agent-kwarg workdir=/path``: the task working directory; defaults to
  ``/app``.
- ``--agent-kwarg timeout_s=N``: the RPent inner time limit, which must be below
  the task's agent timeout. An equal value lets Harbor time out first and lose
  ``session.json``.
- ``--agent-kwarg max_turns=N``: the model-response limit in the planner, not a
  count of shell commands.
- ``--agent-kwarg sandbox=full-access``: lets the planner script the simulator
  from the shell. The default ``read-only`` keeps shell commands away from the
  simulator, so only RPent tools act on the episode.

Harbor's task budget is always the outer hard limit; RPent writes
``session.json`` at the end of a trial, so leave enough headroom.

Task Memory and Exploration
---------------------------

- ``--agent-kwarg memory_source=/path``: uploads an existing RPent memory corpus
  in place of the trial's empty memory directory.
- ``--agent-kwarg explore_sessions=N``: runs several independent sessions on the
  same episode.
- Each trial starts with an empty ``memory/`` and its own ``codex-home/``; model
  and API key come only from explicit environment variables.

Compare Skills and Raw
----------------------

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

The default arm replaces direct joint-target submission with two high-level
skills:

- ``move_to_xyz(x, y, z)`` moves the TCP to the world position: it solves IK,
  submits short action chunks, and re-reads the tool position until the error is
  within 1 cm. It keeps the current orientation, so the planner can reason in
  Cartesian space.
- ``set_gripper(opening)`` sets the finger opening while holding the current arm
  pose.

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

.. code-block:: bash

   # Raw: register the raw tools, including efe_submit
   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 --n-attempts 1 \
     --allow-agent-host "$EFE_MODEL_HOST" \
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
     --allow-agent-host "$EFE_MODEL_HOST" \
     --agent-kwarg source_root="$RPENT_ROOT" \
     --agent-kwarg timeout_s=3500 --agent-kwarg max_turns=200 \
     --jobs-dir "$RPENT_ROOT/logs/efe-skills"

Keep the model, endpoint, reasoning effort, task revision, time limit, and
attempt count fixed across arms. A single trial is not a comparison.

Isolation and Measurement Mode
------------------------------

- The runner defaults to the Codex ``read-only`` sandbox, so only RPent tools act
  on the episode. ``full-access`` measures the agent's free-form behavior; shell
  actions do not appear in the agent execution metrics.
- The generic ``rpent --robot efe`` entry point passes no sandbox setting and
  always runs with ``full-access``; use the Harbor adapter for evaluation and
  comparisons.
