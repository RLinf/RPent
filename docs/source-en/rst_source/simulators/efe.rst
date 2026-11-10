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

      CodexPlanner connects to your model API and selects the next tool call.

   .. grid-item-card:: Tasks

      EFE's standard and standard-realtime tasks.

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
     - Supported. The simulator advances when the agent sends actions.
   * - ``standard-realtime``
     - 84
     - Supported. The simulator keeps running while the model thinks.
   * - ``privileged``
     - 959
     - Not supported. The agent delivers a complete trajectory or policy for evaluation.

In ``standard-realtime``, the robot continues following the most recently
commanded joint positions or other control targets while waiting for a new
action. The scene keeps changing during this wait, so model response time
can affect the outcome.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

RPent provides two ways for the model to control the robot. **Skills** is the
default: the model requests a tool-tip position with ``move_to_xyz`` or a
gripper opening with ``set_gripper``. RPent converts these requests into robot
joint commands. **Raw** lets the model send the task's native action values
directly with ``efe_submit``. Start with the default; see `Compare Skills and Raw`_ for limits
and comparison commands. If a robot's controls are unsupported by Skills,
RPent automatically uses Raw instead.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Item
     - Description
   * - Observation
     - Camera RGB images, optional depth, robot joint positions, tool pose, gripper opening, and a description of the available controls.
   * - Action
     - Skills requests tool-tip motion and gripper opening. Raw submits native action values in the order specified by the task.
   * - Reward / success
     - EFE's verifier grades the robot's outcome. Read ``verifier/reward.json`` or the trial's ``result.json`` as described in `View Results`_.
   * - Task instruction
     - Harbor passes the task's original instruction to RPent.

Some tasks allow another attempt. For those tasks, the model can call
``efe_reset`` to abandon the current episode (one attempt), return to the same
initial scene, and spend one reset from the task's budget. Only the last
episode is graded; RPent does not reset automatically.

Other tasks withhold the success signal from the agent. For those tasks,
RPent provides ``efe_end_episode`` so the model can declare that it has finished
acting. The verifier then grades the final state. Calling this tool does not
declare success, and ending before the goal is achieved can receive a failure.

Installation and Resources
--------------------------

Run the commands below on the Linux host, not inside a container. Use git,
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

2. Set the paths to your checkouts and install EFE's Harbor integration.
   Replace the first two paths with your own. ``RPENT_ROOT`` is the root of
   the RPent repository; ``EFE_ROOT`` is the root of the Embodied First Exam repository.

   .. code-block:: bash

      export RPENT_ROOT=/absolute/path/to/RPent
      export EFE_ROOT=/absolute/path/to/embodied-first-exam
      uv sync --project "$EFE_ROOT" --python 3.12
      docker info
      uv run --project "$EFE_ROOT" harbor --version
      test -f "$RPENT_ROOT/robots/efe/harbor_agent.py"

   Docker must report a running daemon, Harbor must report version 0.23.0,
   and the file check must exit successfully. The first trial builds or pulls
   its task image and downloads resources. Check the
   `AutoBio setup requirements
   <https://github.com/mll-lab-nu/embodied-first-exam/tree/main/tasks/autobio#before-you-run>`_
   for disk space and source licensing before running this example.

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

Use the same host terminal so the variables above remain available. This
command starts one trial with the default Skills tools. Harbor installs RPent
in the task container automatically; you do not need to install RPent there.

.. code-block:: bash

   PYTHONPATH="$RPENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"    uv run --project "$EFE_ROOT" harbor run      --path "$EFE_TASK"      --agent robots.efe.harbor_agent:RPentEfeAgent      --model "$EFE_MODEL" --n-concurrent 1      --allow-agent-host "$MODEL_HOST"      --yes      --jobs-dir "$RPENT_ROOT/logs/efe-rpent"

``--agent`` tells Harbor which RPent integration to load; keep its value as
shown. ``PYTHONPATH`` lets Harbor find that integration in your RPent checkout.
``--n-concurrent 1`` runs one task at a time. ``--yes`` skips Harbor's
confirmation prompt. Setup can take longer on the first run because it installs
dependencies. After execution and grading, look under ``logs/efe-rpent`` for
the result files below. If setup or model access fails, go to
`Debug a Single Task`_.

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

After the single-task run, replace ``<benchmark>`` below with the name of
your chosen benchmark. Check its task modes first: use a directory containing
supported standard or standard-realtime tasks only, not privileged tasks.

``--path`` points at a benchmark directory; Harbor treats its child task
directories as a dataset. ``-d embodied-first-exam@all`` is the full set that
EFE registers. For example, EFE's own ``harbor run -d embodied-first-exam@all
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

Benchmark Results and Reproduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are no published full-benchmark results for this RPent integration yet.
Individual smoke trials check that the integration runs; they do not establish
a success rate, speed advantage, or Skills-versus-Raw comparison. EFE's own
leaderboard is not a result for RPent.

To make a batch result reproducible, record both repository revisions, the
task list, model identifier and endpoint, reasoning effort, time and turn limits,
reset and retry settings, and attempt count. Save Harbor's job and trial results,
``agent/rpent-source.json`` (source archive hash), and
``agent/setup/pip-freeze.txt`` (installed dependencies). Report successes over
all attempted trials and account for errors and timeouts explicitly.

Common Options
--------------

- ``--agent-kwarg source_root=/path``: the RPent checkout to install; defaults
  to the adapter's source checkout. Set it explicitly if the adapter is
  imported from an installed wheel.
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
- Each trial starts with an empty ``memory/`` and its own ``codex-home/``; the model is selected by ``--model``, and credentials come from the
  forwarded environment variables.

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
     - ``efe_spec``, ``efe_observe``, ``efe_status``, ``efe_recover``
     - Same
   * - Action
     - ``move_to_xyz``, ``set_gripper``
     - ``efe_submit``
   * - Conditional
     - ``efe_reset`` with a reset budget, ``efe_end_episode`` when success is hidden
     - Same

The Skills configuration provides two high-level motion tools:

- ``move_to_xyz(x, y, z)`` moves the tool tip to the world position: it uses
  inverse kinematics (IK) to calculate joint positions, sends short action
  sequences, and re-reads the tool position until
  the error is within 1 cm. It keeps the current orientation, so the planner can
  reason in Cartesian space.
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

   # Skills: the Skills configuration, with move_to_xyz and set_gripper
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

- The Harbor runner defaults to Codex's ``read-only`` sandbox. It restricts shell
  commands, not RPent tools: a shell command cannot write files or reach the
  network, so it cannot connect to the simulator. Only RPent tools operate the
  robot.
- With ``full-access``, the agent can write shell scripts that call ``rcb_client``
  directly and bypass the RPent tools. Those calls never pass through RPent's
  client, so ``metrics`` in ``session.json`` does not count them, and the run
  cannot be compared with one that drives the robot through the tools.
- The generic ``rpent --robot efe`` entry point has no sandbox option and always
  runs with ``full-access``. Use it for manual and interactive debugging; use the
  Harbor adapter for evaluation and comparison.

Integration Details for Developers
----------------------------------

The Harbor adapter is the class named by ``--agent``. It packages ``rpent/``,
``robots/``, installation metadata, and package data declared in
``pyproject.toml`` from the RPent checkout. It installs them into a virtual
environment at ``/opt/rpent-efe/venv`` using the task image's Python; the
container does not need ``uv``. The archive excludes host Git data, tests,
API key files, and run logs. ``robots/`` is part of the RPent distribution;
uploading it does not start other robot services.

The runner is RPent's in-container execution code. It starts CodexPlanner,
connects to the simulator with ``rcb_client.Robot``, and saves ``session.json``.
Harbor controls container isolation, execution budgets, and cleanup.
``solved()`` uses only the service's reported end reason. RPent
does not start a simulator, read privileged state, or change EFE's scoring.

The adapter forwards model environment variables and explicit Harbor
``--agent-env`` settings. It maps ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` to
``CODEX_API_KEY`` and ``CODEX_BASE_URL`` unless those are already set.
Host ``CODEX_HOME`` is not forwarded; each trial creates its own directory.

.. code-block:: text

   Harbor trial
   ├── agent container (main)
   │   └── RPent / CodexPlanner ── rcb_client ──▶ sim:8000
   └── simulator container (sim)
       └── EFE task service

Supported service versions are ``eai-standard/2.1``, ``2.2``, and ``2.4`` for
standard mode, and ``2.3`` for standard-realtime. Privileged tasks instead
deliver a replayable ``trajectory.npz`` or a policy program or model. Their
verifier replays the trajectory or runs the policy, which does not match this
integration's step-by-step observation and action loop.

Debug a Single Task
-------------------

First check the failed trial's logs. Use the symptom to choose the next step:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Symptom
     - Check
   * - Harbor cannot import ``robots.efe``
     - Check ``RPENT_ROOT``, that ``robots/efe/harbor_agent.py`` exists, and ``PYTHONPATH`` in the run command.
   * - Container build or simulator startup fails
     - Check ``trial.log``, Docker, task assets, and the task's EFE setup guide.
   * - RPent installation times out
     - Inspect ``agent/setup/``. Harbor's default setup limit is 360 seconds; add ``--agent-setup-timeout-multiplier 4`` for a slow cold install. Package downloads use the task's setup network policy.
   * - Model authentication or connection fails
     - Check the key, model name, API URL, ``MODEL_HOST``, and overriding ``CODEX_*`` variables. The hostname allowlist applies during agent execution.
   * - No ``session.json`` after a timeout
     - Keep RPent's ``timeout_s`` below the task's agent timeout so RPent can save results before Harbor stops it.

For interactive investigation, ``harbor task start-env`` starts the same task's
containers and opens a shell in the agent container. It does not run the
complete trial or verifier. The following commands mount RPent for installation
and the task's instruction at ``/task/instruction.md``.

1. On the host, start the task environment and open its container shell:

   .. code-block:: bash

      # On the host: start the environment and open the container shell
      uv run --project "$EFE_ROOT" harbor task start-env \
        --path "$EFE_TASK" \
        --mounts '[{"type":"bind","source":"'"$RPENT_ROOT"'","target":"/opt/rpent","read_only":true},{"type":"bind","source":"'"$EFE_TASK"'","target":"/task","read_only":true}]'

2. In the container shell, copy the mounted source to a writable directory
   and install RPent:

   .. code-block:: bash

      cp -a /opt/rpent /tmp/rpent-src
      python3 -m venv /tmp/rpent-venv
      /tmp/rpent-venv/bin/python -m pip install /tmp/rpent-src

3. In the same container shell, enter the model key and run RPent:

   In the Harbor run above, the adapter maps the host's ``OPENAI_API_KEY``
   and ``OPENAI_BASE_URL`` to ``CODEX_API_KEY`` and ``CODEX_BASE_URL``.
   This command runs RPent directly in the container, without the adapter,
   so set the two variables used by CodexPlanner in the container. Use the
   same key and API URL as above.

   ``read -rsp`` prompts for the key without displaying the input. After you
   press Enter, ``echo`` starts a new line and ``export`` passes the key to
   subsequent programs.

   .. code-block:: bash

      read -rsp "Model API key: " CODEX_API_KEY; echo
      export CODEX_API_KEY
      export CODEX_BASE_URL=https://api.openai.com/v1
      /tmp/rpent-venv/bin/rpent --robot efe --planner codex \
        --instruction-file /task/instruction.md \
        --output-dir /app/output \
        --model your-model

``/opt/rpent`` is mounted read-only, and the install writes build metadata into
the source tree, so copy it to ``/tmp/rpent-src`` first. ``--instruction-file``
points at the task's own ``instruction.md``. The generic CLI defaults to the
``api`` planner; ``--planner codex`` runs the same CodexPlanner as the adapter.
``rpent --robot efe --planner codex`` always runs with ``full-access``, so shell
commands can reach the simulator too; use it for debugging and the adapter for
comparisons. ``--efe-base-url`` probes ``sim:8000`` by default. ``start-env``
does not apply the trial's agent network policy (``--allow-agent-host`` works in
``harbor run`` only), so the container uses Docker's default network.

Inside the container, enter the key at the prompt. For a custom endpoint,
replace ``CODEX_BASE_URL`` with its full URL. Replace ``your-model`` with the
same model identifier used above, without Harbor's ``openai/`` prefix.
Debug output is written to ``/app/output`` in the container. This procedure
does not produce a Harbor verifier score. Exit the shell when finished.
For container lifecycle options, see ``harbor task start-env --help``.
