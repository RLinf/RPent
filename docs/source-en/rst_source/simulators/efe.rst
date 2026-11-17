Embodied First Exam
===================

This page runs EFE (Embodied First Exam) tasks with RPent through Harbor, using
an AutoBio laboratory task, and shows how to read the scores. The example needs
only a CPU; the host must run Docker and have access to a model API.

"Run a Task" gives the complete single-trial path. The sections after it cover
the result files, benchmark runs, tool configuration, manual debugging, and
common failures.

Overview
--------

EFE is a collection of robot tasks with predefined scenes and scoring rules.
RPent uses a language model to choose actions from camera images and robot
observations. Harbor starts the task container, runs RPent, saves the results,
and removes the containers afterward. Harbor calls one task run a **trial**,
and this page follows that usage.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      RPent motion tools (Skills), or the task's native action values (Raw).

   .. grid-item-card:: Planners

      The Harbor path always uses the Codex planner. It connects to the model API and selects the next tool call.

   .. grid-item-card:: Tasks

      EFE's standard-mode tasks.

   .. grid-item-card:: Hardware

      Linux with Docker. AutoBio uses CPU; see EFE's documentation for GPU requirements of other tasks.

Harbor runs RPent and the simulator in separate containers. RPent sends actions
to the simulator through EFE's client library, ``rcb_client``. EFE's evaluator
(the verifier) checks the outcome independently of the model's final answer.

Tasks
~~~~~

Choose a task from the `EFE task browser
<https://www.embodied-first-exam.ai/tasks/>`_. Its files are in
``tasks/<benchmark>/<task>`` in the `EFE repository
<https://github.com/mll-lab-nu/embodied-first-exam>`_, where a benchmark groups
related tasks. The counts below are the inventory documented for this
integration. They are not counts of tasks evaluated with RPent, and upstream
may add more.

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
at connection time with an explicit error instead of failing partway through a
run.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

RPent has two control modes, selected by the ``skills`` option. **Skills** is
the default: the model calls ``move_to_pose`` for a tool pose and
``set_gripper`` for a gripper opening. With **Raw**, the model sends the task's
native action values through ``efe_submit``. When a task's action layout is not
supported by Skills, the toolkit keeps the Raw tool. The table below summarizes
both modes; see `Compare Skills and Raw`_ for the tool list and the comparison
commands.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Item
     - Description
   * - Observation
     - Camera RGB images, optional depth, camera calibration, robot joint positions, tool pose, gripper opening, and a description of the available controls.
   * - Action
     - Skills requests a tool position, an optional orientation, and a gripper opening. Raw submits native action values in the task's specified order.
   * - Reward / success
     - EFE's verifier grades the robot's actual outcome against the task rules. Read ``verifier/reward.json`` or the trial's ``result.json`` as described in `View Results`_.
   * - Task prompt
     - Harbor passes the task's original instruction to RPent.

When the service provides calibration, ``efe_observe`` returns it in
``camera_calibration`` and returns per-frame world-to-camera transforms in
``frame_extrinsics_cv``. Use these values with the depth image when you need to
relate a pixel to the world frame.

EFE's optional reset budget lets a task allow up to N attempts. On those tasks,
the model can call ``efe_reset`` to abandon the current attempt, return to the
frozen start state, and spend one reset; only the last attempt is graded. This
is the EFE standard-mode reset rule in the
`official protocol <https://github.com/mll-lab-nu/embodied-first-exam/blob/main/docs/protocol/standard_mode.md>`_.

Other tasks withhold the success signal from the agent. RPent provides
``efe_end_episode`` for those tasks so the model can declare that it has
finished acting. The verifier then grades the final state. Calling this tool
does not declare success, and ending before the goal is reached can receive a
failure.

Installation and Resources
--------------------------

Run the commands below in a Linux host terminal. Use git,
`uv <https://docs.astral.sh/uv/getting-started/installation/>`_, and Docker
Engine with Compose v2 and BuildKit. Follow EFE's
`host requirements and installation guide
<https://github.com/mll-lab-nu/embodied-first-exam#quick-start>`_ first.
GPU tasks also need the NVIDIA driver and NVIDIA Container Toolkit.
This integration targets Harbor 0.23.0 and Python 3.12.

1. Obtain both source repositories. If you already have them, skip this step
   and use their absolute paths in the next one.

   .. code-block:: bash

      git clone https://github.com/RLinf/RPent.git
      git clone --recurse-submodules https://github.com/mll-lab-nu/embodied-first-exam.git

2. Set the paths to both checkouts and install EFE's dependencies, including
   Harbor. Replace the first two paths with your own. ``RPENT_ROOT`` is the root
   of the RPent repository; ``EFE_ROOT`` is the root of the EFE repository.

   .. code-block:: bash

      export RPENT_ROOT=/absolute/path/to/RPent
      export EFE_ROOT=/absolute/path/to/embodied-first-exam
      uv sync --project "$EFE_ROOT" --python 3.12
      uv pip install --python "$EFE_ROOT/.venv/bin/python" -e "$RPENT_ROOT"
      docker info
      uv run --project "$EFE_ROOT" harbor --version
      test -f "$RPENT_ROOT/robots/efe/harbor_agent.py"

   This command installs RPent into EFE's host Python environment for manual
   use of ``rpent --robot efe``. You do not need to install RPent on the host
   to run a Harbor trial; Harbor installs it in the task container.
   ``docker info`` must report a running service, ``harbor --version`` must
   report 0.23.0, and ``test -f`` must exit without an error. The first trial
   also builds or pulls its task image and downloads resources, so setup takes
   longer.

3. Select the example task and configure the model. This task asks the robot to
   place a tube into a centrifuge rotor. Replace ``your-model`` with a model
   identifier supported by your API provider and the Codex planner.

   .. code-block:: bash

      export EFE_TASK="$EFE_ROOT/tasks/autobio/autobio-insert-centrifuge-5430-i00-standard"
      export EFE_MODEL=openai/your-model
      export OPENAI_API_KEY=your-key
      export OPENAI_BASE_URL=https://api.openai.com/v1
      export MODEL_HOST=api.openai.com
      test -f "$EFE_TASK/task.toml"

   ``MODEL_HOST`` is the model API host the task container may reach; the run
   command passes it to ``--allow-agent-host``. For a compatible custom API,
   replace both ``OPENAI_BASE_URL`` and ``MODEL_HOST``. Use only the hostname
   for ``MODEL_HOST``, without ``https://`` or a path: for the API URL
   ``https://api.example.com/v1``, use ``api.example.com``.

   Keep API keys out of version control. Existing ``CODEX_API_KEY`` and
   ``CODEX_BASE_URL`` override the corresponding ``OPENAI_*`` variables; unset
   them if you intend to use the settings above.

   To check credentials and the endpoint first, run
   ``rpent-check-llm --planner codex --model <model> --json`` on the host. The
   Codex planner reads ``CODEX_API_KEY`` there and does not fall back to
   ``OPENAI_API_KEY``; inside the container, the adapter performs that
   mapping.

Run a Task
----------

Use the same terminal so the variables above remain available. This command
starts one trial with the default Skills tools. Harbor installs RPent in the
task container automatically; you do not need to install it there by hand.

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
``--n-concurrent 1`` runs one task at a time, and ``--yes`` skips Harbor's
confirmation prompt. Setup can take longer on the first run because it installs
dependencies. After execution and grading, look under ``logs/efe-rpent``; if
the trial fails, start with ``trial.log`` and ``agent/rpent/run.log``.

View Results
------------

Results are written under ``$RPENT_ROOT/logs/efe-rpent`` on the host.
``--jobs-dir`` is Harbor's job root; each ``harbor run`` creates a timestamped
job directory below it.

.. code-block:: text

   <jobs-dir>/<job_name>/
   ├── result.json              # Job summary
   └── <trial_name>/
       ├── result.json          # Verifier result, tokens, wall time
       ├── agent/
       │   ├── setup/           # Agent install log
       │   └── rpent/
       │       ├── transcript_*.json # Generic CLI transcript and planner stats
       │       ├── efe-result.json   # EFE metrics and finalization data
       │       ├── codex_*.txt       # Planner transcript
       │       ├── codex-home/  # CODEX_HOME for this trial
       │       └── memory/      # Memory root for this trial
       ├── verifier/            # Verifier logs and reward.json
       ├── artifacts/           # Collected artifacts
       ├── config.json
       ├── lock.json
       └── trial.log

The container path ``/logs/agent/rpent/`` maps to
``<trial_name>/agent/rpent/`` on the host. EFE's verifier writes
``verifier/reward.json``, and Harbor records those scores as
``verifier_result.rewards`` in the trial's ``result.json``. Planner statistics
are in ``agent/rpent/transcript_*.json`` and EFE metrics are in
``agent/rpent/efe-result.json``. An incomplete trial may lack these files;
check ``trial.log`` and ``agent/rpent/run.log`` first.

Success Criterion
~~~~~~~~~~~~~~~~~

A trial's outcome is ``verifier_result.rewards.success`` in its
``result.json``. Tasks that support retries also record ``first_success`` for
the first attempt; the final score comes from the last attempt.

A model response such as ``finish(status="completed")`` only means that the
agent stopped working, and a zero CLI exit code only means the execution code
did not report an error. Neither is the verifier's score.

Run a Benchmark
---------------

Replace ``<benchmark>`` below with the benchmark name. ``--path`` points at a
benchmark directory, where each child directory is one task, and together they
form the task set to run. Check the directory first: it must contain only
supported standard-mode tasks, not realtime or privileged ones.
``-d embodied-first-exam@all`` is the full set that EFE registers; for example,
EFE's own ``harbor run -d embodied-first-exam@all -p tasks/roboarc`` runs only
RoboARC.

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

Pass these options to the adapter as ``--agent-kwarg key=value``.

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Option
     - Default
     - Description
   * - ``source_root=/path``
     - The checkout containing RPentEfeAgent
     - RPent source directory to build the wheel from. Set it explicitly when importing the agent from an installed wheel.
   * - ``workdir=/path``
     - ``/app``
     - Working directory inside the task container.
   * - ``timeout_s=N``
     - Detected
     - RPent trial time limit. RPent reads it from the Harbor trial configuration and, for a local task, ``task.toml``; either source reserves time for planner shutdown and CLI finalization.
   * - ``connect_timeout_s=N``
     - 60
     - Maximum seconds to wait for the EFE service. This time is deducted from the RPent trial budget before the planner gets the remainder.
   * - ``agent_timeout_s=N``
     - Detected
     - Overrides the Harbor timeout when the task configuration is not available as a local directory; use the same value Harbor does. Harbor 0.23 resolves this timeout internally but does not pass it to installed agents, so the adapter reads the trial configuration and ``task.toml`` itself.
   * - ``max_turns=N``
     - 200
     - Model-response limit in the planner, not a count of shell commands.
   * - ``sandbox=full-access``
     - ``read-only``
     - With ``full-access``, the planner can script the simulator from the shell. The default ``read-only`` keeps shell commands away from the service, so only RPent tools act on the current attempt.
   * - ``skills=false``
     - ``true``
     - Registers only the raw tools such as ``efe_submit``; see `Compare Skills and Raw`_.
   * - ``memory_source=/path``
     - Unset
     - Uploads an existing memory directory for this trial. Without it, the trial starts with an empty ``memory/``.
   * - ``reasoning_effort=low``
     - ``none``
     - Reasoning effort passed to the Codex planner: ``none``, ``low``, ``medium``, ``high``, or ``xhigh``.

Task Memory
-----------

- ``--agent-kwarg memory_source=/path`` uploads an existing RPent memory
  directory for the trial. Without it, the trial starts with an empty
  ``memory/``.
- Each trial uses its own ``codex-home/``; ``--model`` selects the model, and
  credentials come from the forwarded environment variables.

Compare Skills and Raw
----------------------

Run the same task with Skills and Raw to compare the two tool configurations.
Keep the model and execution budget the same; only the action tools change.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Category
     - Default (``skills=true``)
     - Raw (``skills=false``)
   * - Reading and control
     - ``efe_spec``, ``efe_observe``, ``efe_status``, ``efe_recover``, ``view_env_state``
     - Same
   * - Action
     - ``move_to_pose``, ``set_gripper``
     - ``efe_submit``
   * - Conditional
     - ``efe_reset`` with a reset budget, ``efe_end_episode`` when success is hidden
     - Same

The Skills configuration provides two high-level motion tools:

- ``move_to_pose(x, y, z, orientation_wxyz=None)`` moves the tool tip to a
  world position and, when supplied, a world orientation. The quaternion order
  is ``w, x, y, z``; omit the orientation to keep the current one. The skill
  uses inverse kinematics (IK), sends short action chunks, and reads the pose
  back to correct the move. It stops at 1 cm position error and 0.05 rad
  orientation error, or earlier when the iteration limit is reached (20 by
  default, at most 50), the solver fails, or the task ends.
- ``set_gripper(opening)`` sets the gripper opening.

The Raw configuration uses the service interface directly. ``efe_spec``
reports the meaning and order of each action row's columns (the action layout),
and the planner sends joint-target chunks with ``efe_submit``.

The skills support one layout: the leading columns are the joint targets, in
``spec.kinematics.joint_names`` order, and the last column is the gripper
opening. The task also needs a tool frame from ``tcp_frames`` that appears in
``ik_targets``, and proprioception must include that frame's pose and every
joint column. When the layout reorders, omits, or adds columns, or no usable
tool frame exists, the skills are not registered: the toolkit keeps
``efe_submit``, records the reason as ``skills_error`` in ``efe-result.json``,
and the prompt lists only the tools the session registers.

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

   # Skills: the default configuration, with move_to_pose and set_gripper
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

Manual Run (Debugging)
----------------------

To debug tool calls or the prompt without Harbor, call the generic CLI. This
entry point only connects to a running task service; it starts no simulator
and does not grade the attempt.

.. code-block:: bash

   rpent --robot efe \
     --instruction-file "$EFE_TASK/instruction.md" \
     --planner codex --model <model> --reasoning-effort low \
     --codex-sandbox read-only \
     --memory-profile local --memory-dir ./memory/efe \
     --output-dir logs/efe-manual

``--instruction-file`` is required and points at the task's instruction file.
Without ``--task-name``, the task name comes from that file's parent directory;
the Harbor path uploads the instruction elsewhere, so it passes the task
directory name explicitly. ``--efe-base-url`` sets the service address; when
omitted, the client probes ``sim:8000`` and then ``localhost:8000``.
``--efe-connect-timeout-s`` defaults to 3600 seconds, and
``--efe-trial-timeout-s`` sets a deadline for the whole trial, including the
initial connection wait. ``--efe-no-skills`` registers only the raw tools.
``--memory-repo`` is not supported; use ``--memory-profile local`` with
``--memory-dir`` when you need memory.

The Harbor path passes ``--model`` without the ``openai/`` prefix after
stripping it; a manual run uses the same bare model name.

Isolation and Measurement Mode
------------------------------

- Harbor gives Codex a ``read-only`` sandbox by default. It restricts shell
  commands, not RPent tools: shell commands cannot write files or access the
  network, and robot actions go through RPent tools only.
- With ``full-access``, the agent can write shell scripts that call
  ``rcb_client`` directly and bypass the RPent tools. Those calls never pass
  through RPent's client, so ``metrics`` in ``efe-result.json`` does not count
  them, and the run cannot be compared with one that drives the robot through
  the tools.
- The generic CLI's ``--codex-sandbox`` defaults to ``full-access``; pass
  ``--codex-sandbox read-only`` to reproduce Harbor's default.
- ``rpent`` defaults to ``--planner api``, which has no Codex shell sandbox.
  Pass ``--planner codex`` to a manual ``rpent --robot efe`` run to match the
  Harbor path.

Reproduction
------------

Pin both code versions: the RPent commit and the EFE repository commit that
provides the task definitions and images. Also record the task directory name,
model identifier, reasoning effort, time limit, and attempt count, and keep
``result.json`` and ``config.json`` from the Harbor job directory.

A comparison needs all of these settings to match and an explicit aggregation
method. A single trial does not separate two configurations.

Common Failures
---------------

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Symptom
     - Cause
     - Fix
   * - Connection fails with an unsupported protocol
     - The task uses ``2.1``, ``2.3``, or the realtime protocol
     - Use a standard task on ``2.2`` or ``2.4``; other versions need a separate adapter.
   * - The trial fails soon after start and ``run.log`` reports a connection timeout
     - The task service is not up, or the port is wrong
     - Check ``trial.log`` and the simulator container logs; raise ``connect_timeout_s`` if needed.
   * - Model requests return 401 or cannot connect
     - Wrong credentials, endpoint, or host allowlist
     - Check ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, and ``MODEL_HOST``, and make sure no stale ``CODEX_*`` variables override them.
   * - Only ``efe_submit`` is registered, without Skills
     - The action layout or kinematics do not qualify
     - Read ``skills_error`` in ``efe-result.json`` and check the layout and tool frame against `Compare Skills and Raw`_.
   * - Harbor terminates the trial when the budget runs out
     - Not enough time was allocated
     - Raise ``--agent-timeout-sec`` and pass the same value to the adapter as ``agent_timeout_s``.
   * - ``transcript_*.json`` or ``result.json`` is missing
     - The trial did not finish
     - Start with ``trial.log`` and ``agent/rpent/run.log``; neither ``finish`` nor the exit code is the score.

Integration Details for Developers
----------------------------------

Harbor loads RPent's ``RPentEfeAgent`` through ``--agent``. This entry point
builds a wheel from the RPent checkout and uploads it. The wheel follows the
package discovery and package data declared in ``pyproject.toml``, so host Git
data, tests, caches, API key files, and run logs are not included. Harbor
installs the wheel into ``/opt/rpent-efe/venv`` with the task image's Python;
the container does not need ``uv``. ``robots/`` is part of the RPent
distribution; installing it does not start other robot services.

The adapter runs the installed ``rpent`` in the container with ``--robot efe``
and ``--planner codex``, and passes ``--memory-profile local``,
``--memory-dir``, ``--task-name``, and the run options explicitly. The EFE
``RobotSpec`` connects the generic CLI to ``EfeEnvClient`` and ``EfeToolkit``:
the CLI writes ``transcript_*.json``, and the ``finalize_run`` hook writes
``efe-result.json``. The adapter passes the trial deadline to
``EfeEnvClient`` and leaves time for planner shutdown and CLI finalization;
when the transcript records a planner budget stop, it treats it as a clean
agent exit so Harbor can grade the attempt. ``solved()`` uses only the
service's reported end reason. RPent does not start a simulator, read
privileged state, or change EFE's scoring.

RPentEfeAgent forwards model environment variables and explicit Harbor
``--agent-env`` settings. It maps ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` to
``CODEX_API_KEY`` and ``CODEX_BASE_URL`` unless those are already set. Host
``CODEX_HOME`` is not forwarded; each trial creates its own directory.

.. code-block:: text

   Harbor trial
   ├── agent container (main)
   │   └── RPent / Codex planner ── rcb_client ──▶ sim:8000
   └── simulator container (sim)
       └── EFE task service

Supported service versions are ``eai-standard/2.2`` and ``2.4`` for standard
mode. ``2.1`` and ``2.3`` publish ``control_dt`` instead of ``control_hz`` in
``/spec`` and ship a different ``rcb_client`` (``get_obs(include_depth=...)``,
``Result.depth``, no ``wait_ready``), so ``EfeEnvClient`` rejects them with an
explicit error rather than failing later on a missing spec field. Supporting
them needs a separate adapter, including for ``2.3`` and its
``standard-realtime`` tasks.

``privileged`` tasks require a replayable ``trajectory.npz`` or a policy
program or model. The verifier replays the trajectory or runs the policy, which
does not fit RPent's step-by-step observation and action.

Next Steps
----------

- Choose and configure the model and planner: :doc:`../guides/configure_planner`
- Full generic CLI reference: :doc:`../guides/cli`
- Task memory and exploration: :doc:`../guides/memory`
- Add a new robot integration: :doc:`../development/add_robot`
