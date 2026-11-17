Embodied First Exam
===================

Run one simulated robot task with RPent, then check whether the robot achieved
the goal. This guide uses an AutoBio laboratory task that runs without a GPU.
You need a Linux machine with Docker and access to a model API.

Overview
--------

EFE (Embodied First Exam) is a collection of robot tasks with predefined scenes
and scoring rules. RPent uses a language model to choose actions from camera
images and robot observations. Harbor is the program that starts the task,
runs RPent, saves the results, and removes the containers afterward.
Harbor calls one task run a **trial**.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      RPent motion tools or the task's native robot controls.

   .. grid-item-card:: Planners

      This adapter currently supports CodexPlanner. CodexPlanner connects to your model API and selects the next tool call.

   .. grid-item-card:: Tasks

      EFE's standard-mode tasks.

   .. grid-item-card:: Hardware

      Linux with Docker. GPU requirements depend on the task; AutoBio uses CPU.

Harbor runs RPent and the simulator in separate containers. RPent sends actions
to the simulator through EFE's client library, ``rcb_client``. EFE's evaluator
(the verifier) checks the outcome independently of the model's final answer.

Tasks
~~~~~

Choose a task from the `EFE task browser
<https://www.embodied-first-exam.ai/tasks/>`_. Its files are in
``tasks/<benchmark>/<task>`` in the `EFE repository
<https://github.com/mll-lab-nu/embodied-first-exam>`_. A benchmark groups related
tasks. The counts below are the inventory documented for this integration;
they are not counts of tasks evaluated with RPent, and upstream may add tasks.

.. list-table::
   :header-rows: 1

   * - Mode
     - Tasks
     - Behavior
   * - ``standard``
     - 1066
     - Supported when the service speaks ``eai-standard/2.2`` or ``2.4``. The simulator advances when the agent sends actions.
   * - ``standard-realtime``
     - 84
     - Not supported. These tasks use ``eai-standard/2.3`` or ``eai-standard-realtime/1.0``; this integration handles ``2.2`` and ``2.4``.
   * - ``privileged``
     - 959
     - Not supported. The agent delivers a complete trajectory or policy for evaluation.

This integration handles standard-mode tasks whose service speaks
``eai-standard/2.2`` or ``2.4``. Check ``metadata.protocol`` in the task's
``task.toml`` before selecting a benchmark; a task with another protocol fails
at connection time with an explicit error.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

RPent has two control modes. **Skills** is the default: the model requests a
tool pose with ``move_to_pose`` or a gripper opening with ``set_gripper``.
**Raw** sends the task's native action values with ``efe_submit``. A task whose
layout Skills cannot drive keeps the Raw tool instead. The table below
summarizes both; see `Compare Skills and Raw`_ for the tool list and
comparison commands.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Item
     - Description
   * - Observation
     - Camera RGB images, optional depth, camera calibration, robot joint
       positions, tool pose, gripper opening, and a description of the controls.
   * - Action
     - Skills requests tool position, optional tool orientation, and gripper
       opening. Raw submits native action values in the task's specified order.
   * - Reward / success
     - EFE's verifier grades the robot's outcome. Read ``verifier/reward.json`` or the trial's ``result.json`` as described in `View Results`_.
   * - Task instruction
     - Harbor passes the task's original instruction to RPent.

When the service provides calibration, ``efe_observe`` returns it in
``camera_calibration``. It returns per-frame world-to-camera transforms in
``frame_extrinsics_cv``. Use these values with the depth image when you need
to relate a pixel to the world frame.

EFE's optional reset budget lets a task allow up to N attempts. On those tasks,
the model can call ``efe_reset`` to abandon the current episode, return to the
frozen start state, and spend one reset; only the last episode is graded. This
is the EFE standard-mode reset rule in the
`official protocol <https://github.com/mll-lab-nu/embodied-first-exam/blob/main/docs/protocol/standard_mode.md>`_.

``explore_sessions`` is an RPent policy on top of that budget, not an EFE
feature. When it is greater than 1, RPent runs independent planner sessions
and, if a session ends early and resets remain, spends one reset before the
next session. An episode the agent ends with ``efe_end_episode`` is graded as
it stands and is not reset. The default is 1, so a single run never resets on
its own.

Other tasks withhold the success signal from the agent. For those tasks,
RPent provides ``efe_end_episode`` so the model can declare that it has finished
acting. The verifier then grades the final state. Calling this tool does not
declare success, and ending before the goal is achieved can receive a failure.

Installation and Resources
--------------------------

Run the commands below in a Linux host terminal. Use git,
`uv <https://docs.astral.sh/uv/getting-started/installation/>`_, and Docker
Engine with Compose v2 and BuildKit. Follow EFE's
`host requirements and installation guide
<https://github.com/mll-lab-nu/embodied-first-exam#quick-start>`_ first.
GPU tasks also need the NVIDIA driver and NVIDIA Container Toolkit.
This integration targets Harbor 0.23.0 and Python 3.12.

1. Obtain both source repositories. If you already have them, use their
   absolute paths in the next step instead of cloning again.

   .. code-block:: bash

      git clone https://github.com/RLinf/RPent.git
      git clone --recurse-submodules https://github.com/mll-lab-nu/embodied-first-exam.git

2. Set the paths to both checkouts and install EFE's dependencies, including Harbor.
   Replace the first two paths with your own. ``RPENT_ROOT`` is the root of
   the RPent repository; ``EFE_ROOT`` is the root of the Embodied First Exam repository.

   .. code-block:: bash

      export RPENT_ROOT=/absolute/path/to/RPent
      export EFE_ROOT=/absolute/path/to/embodied-first-exam
      uv sync --project "$EFE_ROOT" --python 3.12
      docker info
      uv run --project "$EFE_ROOT" harbor --version
      test -f "$RPENT_ROOT/robots/efe/harbor_agent.py"

   Docker must report a running service, Harbor must report version 0.23.0,
   and the file check must exit successfully. The first trial builds or pulls
   its task image and downloads resources.

3. Select the example task and configure the model. This task asks the robot
   to place a tube into a centrifuge rotor. Replace ``your-model`` with a model
   identifier supported by your API provider and CodexPlanner.

   .. code-block:: bash

      export EFE_TASK="$EFE_ROOT/tasks/autobio/autobio-insert-centrifuge-5430-i00-standard"
      export EFE_MODEL=openai/your-model
      export OPENAI_API_KEY=your-key
      export OPENAI_BASE_URL=https://api.openai.com/v1
      export MODEL_HOST=api.openai.com
      test -f "$EFE_TASK/task.toml"

   ``MODEL_HOST`` specifies the model API host the task container may reach;
   the run command passes it to ``--allow-agent-host``. For a compatible custom
   API, replace both ``OPENAI_BASE_URL`` and ``MODEL_HOST``. Use only the
   hostname for ``MODEL_HOST``, without ``https://`` or a path. For example,
   use ``api.example.com`` for the API URL ``https://api.example.com/v1``.
   Keep API keys out of version control. Existing ``CODEX_API_KEY`` and
   ``CODEX_BASE_URL`` override the corresponding ``OPENAI_*`` variables;
   unset them if you intend to use the settings above.

.. _run-a-trial:

Run a Task
----------

Use the same terminal so the variables above remain available. This
command starts one trial with the default Skills tools. Harbor installs RPent
in the task container automatically; you do not need to install RPent there.

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

``--agent`` tells Harbor which RPent entry point to load; keep its value as
shown. ``PYTHONPATH`` lets Harbor find it in your RPent checkout.
``--n-concurrent 1`` runs one task at a time. ``--yes`` skips Harbor's
confirmation prompt. Setup can take longer on the first run because it installs
dependencies. After execution and grading, look under ``logs/efe-rpent`` for
the result files below. If the trial fails, start with ``trial.log`` and
``agent/rpent/run.log``.

View Results
------------

Open ``$RPENT_ROOT/logs/efe-rpent`` on the host.
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
       │       ├── codex*.txt   # Planner transcript (under sessions/ when explore_sessions > 1)
       │       ├── sessions/    # Per-session artifacts when explore_sessions > 1
       │       ├── codex-home/  # CODEX_HOME for this trial
       │       └── memory/      # Memory root for this trial
       ├── verifier/            # Verifier logs and reward.json
       ├── artifacts/           # Collected artifacts
       ├── config.json
       ├── lock.json
       └── trial.log

The container path ``/logs/agent/rpent/`` maps to
``<trial_name>/agent/rpent/`` on the host. EFE's verifier writes
``verifier/reward.json``, and Harbor copies those scores to
``verifier_result.rewards`` in the trial's ``result.json``. Agent execution metrics are in
``stats`` and each session's ``metrics`` in ``agent/rpent/session.json``. An
incomplete trial may have no ``result.json`` or ``session.json``; check
``trial.log`` and ``agent/rpent/run.log`` first.

Read the trial's ``verifier_result.rewards.success`` to determine whether
the task succeeded. ``first_success`` describes the first attempt when a task
supports retries. A model response such as ``finish(status="completed")`` only
means that the agent stopped working. A zero CLI exit code only means the
execution code did not report an error; neither is the verifier's success score.

Run a Benchmark
---------------

To run a benchmark, replace ``<benchmark>`` below with its name. Check its
task modes first: use a directory containing only supported standard-mode
tasks, not realtime or privileged tasks.

``--path`` points at a benchmark directory. Each child directory is one task;
together, they form the task set to run. ``-d embodied-first-exam@all`` is
the full set that EFE registers. For example, EFE's own ``harbor run -d embodied-first-exam@all
-p tasks/roboarc`` runs only RoboARC.

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

Each task runs one trial and writes one ``result.json``. ``--n-concurrent``
defaults to 4. Run one trial first, then the batch, and keep the model,
endpoint, task revision, and time limit fixed before comparing results.

Common Options
--------------

- ``--agent-kwarg source_root=/path``: the RPent source directory to build the
  wheel from; defaults to the checkout containing ``RPentEfeAgent``. Set it
  explicitly when importing the agent from an installed wheel.
- ``--agent-kwarg workdir=/path``: the task working directory; defaults to
  ``/app``.
- ``--agent-kwarg timeout_s=N``: an optional RPent inner time limit. By default,
  RPent reads Harbor's resolved timeout from the trial configuration and, for a
  local task, ``task.toml``. It reserves time for shutdown and report writing so
  Harbor does not kill the trial before ``session.json`` is saved.
- ``--agent-kwarg agent_timeout_s=N``: explicitly override the detected Harbor
  timeout. Use this when the task configuration is not available as a local
  task directory. Set it to the same effective timeout Harbor uses. Harbor
  0.23 resolves this timeout internally but does not pass it to installed
  agents, so the adapter reads the trial configuration and, for local tasks,
  ``task.toml``.
- ``--agent-kwarg max_turns=N``: the model-response limit in the planner, not a
  count of shell commands.
- ``--agent-kwarg sandbox=full-access``: lets the planner script the simulator
  from the shell. The default ``read-only`` keeps shell commands away from the
  simulator, so only RPent tools act on the episode.

Task Memory and Exploration
---------------------------

- ``--agent-kwarg memory_source=/path``: uploads an existing RPent memory
  directory for the trial. Without it, the trial starts with an empty ``memory/``.
- ``--agent-kwarg explore_sessions=N``: runs up to ``N`` planner sessions,
  stopping early when the reset budget is spent or time runs out. A session
  that ends early hands off by resetting while the reset budget remains, so
  the next session starts a fresh attempt; an episode the agent ends itself is
  graded without a reset, and only the final episode is graded. Later sessions
  can read working notes from earlier sessions. This is an RPent policy, not
  an EFE requirement; the default is 1. Multi-session handoff has not been
  benchmarked against a single session at the same budget.
- Each trial uses its own ``codex-home/``; the model is selected by ``--model``,
  and credentials come from the forwarded environment variables.

Compare Skills and Raw
----------------------

Run the same task with Skills and Raw to compare the two tool configurations.
Keep the model and execution budget the same; only the action tools change.

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Category
     - Default (``skills=true``)
     - Raw (``skills=false``)
   * - Reading and control
     - ``efe_spec``, ``efe_observe``, ``efe_status``, ``efe_recover``,
       ``view_env_state``
     - Same
   * - Action
     - ``move_to_pose``, ``set_gripper``
     - ``efe_submit``
   * - Conditional
     - ``efe_reset`` with a reset budget, ``efe_end_episode`` when success is hidden
     - Same

The Skills configuration provides two high-level motion tools:

- ``move_to_pose(x, y, z, orientation_wxyz=None)`` moves the tool tip to a world
  position and, when supplied, a world orientation. The quaternion order is
  ``w, x, y, z``. Omit the orientation to keep the current one. The skill uses
  inverse kinematics (IK), sends short action chunks, and checks both position
  and orientation error. It stops at 1 cm position error and 0.05 rad orientation
  error, or earlier if the iteration limit, IK solver, or task ends the move.
- ``set_gripper(opening)`` sets the gripper opening.

The Raw configuration uses the service interface directly. ``efe_spec`` reports the
meaning and order of each action row's columns (the action layout), and the
planner sends joint-target chunks with ``efe_submit``.

An action layout defines what each number in one action row means. The skills
support one layout: the leading columns are the joint targets, in
``spec.kinematics.joint_names`` order, and the last column is the gripper
opening. When the layout reorders, omits, or adds columns, the skills are not
registered: the toolkit keeps ``efe_submit``, records the reason as
``skills_error`` in ``session.json``, and the prompt lists only the tools the
session registers.

.. code-block:: bash

   # Raw: register the raw tools, including efe_submit
   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 \
     --allow-agent-host "$MODEL_HOST" \
     --agent-kwarg skills=false \
     --yes \
     --jobs-dir "$RPENT_ROOT/logs/efe-raw"

   # Skills: the Skills configuration, with move_to_pose and set_gripper
   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
   uv run --project "$EFE_ROOT" harbor run \
     --path "$EFE_TASK" \
     --agent robots.efe.harbor_agent:RPentEfeAgent \
     --model "$EFE_MODEL" --n-concurrent 1 \
     --allow-agent-host "$MODEL_HOST" \
     --yes \
     --jobs-dir "$RPENT_ROOT/logs/efe-skills"

Keep the model, endpoint, reasoning effort, task revision, time limit, and
attempt count fixed across configurations. A single trial is not a comparison.

Isolation and Measurement Mode
------------------------------

- The Harbor runner defaults to Codex's ``read-only`` sandbox. It restricts
  shell commands, not RPent tools: shell commands cannot write files or access
  the network. RPent tools send the robot actions.
- With ``full-access``, the agent can write shell scripts that call ``rcb_client``
  directly and bypass the RPent tools. Those calls never pass through RPent's
  client, so ``metrics`` in ``session.json`` does not count them, and the run
  cannot be compared with one that drives the robot through the tools. At each
  planner-session boundary, the runner reads ``/status`` to sync the episode,
  end reason, and reset allowance before it starts another session.
- The generic ``rpent --robot efe`` entry point has no sandbox option and always
  runs with ``full-access``. Use it for manual and interactive debugging; use the
  Harbor adapter for evaluation and comparison.
- ``rpent`` defaults to ``--planner api``, which has no Codex shell sandbox. Pass
  ``--planner codex`` to a manual ``rpent --robot efe`` run when the trial should
  behave like the Harbor run.

Integration Details for Developers
----------------------------------

Harbor loads RPent's ``RPentEfeAgent`` through ``--agent``. This entry point
builds a wheel from the RPent checkout and uploads it. The wheel follows the
package discovery and package data declared in ``pyproject.toml``, so host Git
data, tests, caches, API key files, and run logs are not included. Harbor
installs the wheel into a virtual environment at ``/opt/rpent-efe/venv`` using
the task image's Python; the container does not need ``uv``. ``robots/`` is
part of the RPent distribution; installing it does not start other robot
services.

The Harbor adapter calls ``robots.efe.runner.run_efe`` directly. The generic
``rpent --robot efe`` CLI is a separate path for manual debugging. ``run_efe``
is headless and owns the trial-wide deadline, the planner sessions on one
episode, and the machine-readable ``session.json``; the generic CLI has no
Codex sandbox option and runs one session for EFE. ``run_efe`` starts
CodexPlanner, uses ``rcb_client.Robot`` to interact with the simulator, and
saves ``session.json``. Harbor controls container isolation, execution
budgets, and cleanup. ``solved()`` uses only the service's reported end
reason. RPent does not start a simulator, read privileged state, or change
EFE's scoring.

RPentEfeAgent forwards model environment variables and explicit Harbor
``--agent-env`` settings. It maps ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` to
``CODEX_API_KEY`` and ``CODEX_BASE_URL`` unless those are already set.
Host ``CODEX_HOME`` is not forwarded; each trial creates its own directory.

.. code-block:: text

   Harbor trial
   ├── agent container (main)
   │   └── RPent / CodexPlanner ── rcb_client ──▶ sim:8000
   └── simulator container (sim)
       └── EFE task service

Supported service versions are ``eai-standard/2.2`` and ``2.4`` for standard
mode.

``2.1`` and ``2.3`` publish ``control_dt`` instead of ``control_hz`` and ship
a different ``rcb_client`` (``get_obs(include_depth=...)``, ``Result.depth``,
no ``wait_ready``), so ``EfeEnvClient`` rejects them with an explicit error
rather than failing later on a missing spec field. Supporting them needs a
separate adapter, including for ``2.3`` and its ``standard-realtime`` tasks.

``privileged`` tasks require a replayable ``trajectory.npz`` or a policy
program/model. The verifier replays the trajectory or runs the policy, which
does not fit RPent's step-by-step observation and action.
