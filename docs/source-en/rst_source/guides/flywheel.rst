.. _libero-data-flywheel:

Data Flywheel
=============

The RPent data flywheel records trajectories from task planning through action
execution and turns successful trajectories into training data.

With the ``api``, ``claude_code``, or ``codex`` planner, ``--record-trajectory``
records vision-language model (VLM) requests and responses, along with tool calls
and their results. Environments that support action recording also capture the
subtask instructions given to the vision-language-action (VLA) model, the actions
the robot executes, and the corresponding observations.

``rpent-flywheel`` exports either VLA or VLM training data:

- ``--vla`` (default): Writes executed VLA action segments from confirmed
  successful trajectories to a LeRobot dataset for supervised fine-tuning with
  RLinf. In the evaluations below, replacing the original VLA with the fine-tuned
  model raised RPent's task success rate from 70% to 73% on LIBERO-PRO and from
  48% to 72% on RoboTwin.
- ``--vlm``: Exports recorded ``api`` planner turns to ``samples.jsonl`` and a
  deduplicated ``images/`` directory for VLM training. It selects episodes with
  confirmed environment success by default, or includes failed and unknown
  outcomes with ``--include-unsuccessful``.

Installation
------------

The example below uses LIBERO-PRO. Follow :doc:`../get_started/quickstart` for
environment setup and simulator assets, and :doc:`configure_planner` for API
credentials. From the RPent repository root, install LIBERO-PRO with the
optional export dependencies:

.. code-block:: bash

   uv pip install -e ".[libero-pro,flywheel]"

The optional ``flywheel`` extra installs LeRobot and is required only for VLA
export (``--vla``). Exporting VLM turns (``--vlm``) works with the base RPent
installation.

Record a Trajectory
-------------------

Recording is off by default. Recording with ``--planner claude_code`` requires
``claude-agent-sdk 0.2.154+`` and ``Claude Code 2.1.274+``.
LIBERO, RoboCasa365, RoboTwin, and RoboDojo record executed actions and observations
only in evaluation mode. To export VLM training samples later, record with
``--planner api``, which is the backend ``rpent-flywheel --vlm`` currently
supports.

Enable recording and choose a run directory:

.. code-block:: bash

   rpent --robot libero \
     --suite libero_goal --task 0 --seed 0 \
     --planner api --model anthropic:claude-opus-4-8 \
     --record-trajectory \
     --output-dir /path/to/run

- ``--record-trajectory``: Enable trajectory recording as described above.
- ``--output-dir /path/to/run``: Set the output root directory for this run.
  Replace ``/path/to/run`` with your desired save path.

The evaluation example saves its trajectory under
``/path/to/run/trajectories/<task_id>/`` and attachments in the ``artifacts/``
subdirectory.

Trajectory Data Format
----------------------

The directory tree shows how a trajectory is stored, and the tables describe
the data format of each file.

.. code-block:: text

   /path/to/run/trajectories/<task_id>/
   ├── task.json
   ├── planners.jsonl
   ├── predictions.jsonl
   ├── actions.jsonl
   └── artifacts/

.. list-table:: ``task.json`` records the instruction, configuration, and outcome of one task run
   :header-rows: 1

   * - Field
     - Meaning
   * - ``task.task_id``
     - Unique ID for this run, used as the trajectory directory name ``<task_id>``.
   * - ``task.instruction``
     - Task instruction for this run.
   * - ``task.config``
     - Run configuration stored in ``task.config.data``, including CLI and run
       settings, ``system_prompt``, and registered ``tools``.
   * - ``task.result``
     - Run outcome. ``task.result.data.environment_success`` indicates whether the task succeeded.

.. list-table:: ``planners.jsonl`` (one model call per line)
   :header-rows: 1

   * - Field
     - Meaning
   * - ``planner_id``
     - Unique ID of this model call within the trajectory.
   * - ``backend``
     - Planner backend that made the call (``api``, ``claude_code``, or ``codex``).
   * - ``status``
     - State of this model call (``open``, ``completed``, ``failed``, or ``cancelled``).
   * - ``request``
     - The request sent to the model, with conversation context and available
       tools in the backend's format.
   * - ``response``
     - Model output, which can include text or requests to call tools.

.. list-table:: ``predictions.jsonl`` (one VLA prediction per line)
   :header-rows: 1

   * - Field
     - Meaning
   * - ``prediction_id``
     - Unique ID of this prediction within the trajectory.
   * - ``instruction``
     - Instruction the VLA used for this prediction.
   * - ``action_ids``
     - The ``action_id`` values in ``actions.jsonl`` for actions executed from
       this prediction, in execution order.

.. list-table:: ``actions.jsonl`` (one executed environment action per line)
   :header-rows: 1

   * - Field
     - Meaning
   * - ``action_id``
     - Unique ID of this action within the trajectory.
   * - ``source``
     - ``vla`` for a policy prediction, ``tool_computed`` for an action the tool
       computes and executes, or ``direct`` for a control value submitted through
       the direct action tool.
   * - ``action.format``
     - Action format used by the robot.
   * - ``action.data``
     - Control command the environment executed at this step.
   * - ``before_observation``
     - Environment observation before this action.

Export Training Data
--------------------

``rpent-flywheel`` supports two mutually exclusive export modes: ``--vla``
(default) for LeRobot VLA datasets and ``--vlm`` for API planner turn samples.

In both modes, ``--input-root`` accepts either a single trajectory directory
containing ``task.json`` or a parent directory containing multiple runs.
``--output-dir`` must not exist, and original recordings are left unchanged.
When several runs are saved in separate directories like these, set
``--input-root`` to their parent directory and use ``--pattern`` (keeping
quotes around shell wildcards) to select matching run directories:

.. code-block:: text

   /path/to/
   ├── run_001/trajectories/<task_id_1>/
   └── run_002/trajectories/<task_id_2>/

Export VLA Training Data
~~~~~~~~~~~~~~~~~~~~~~~~

Export recorded trajectories from one robot to a new output directory:

.. code-block:: bash

   rpent-flywheel \
     --vla \
     --input-root /path/to/run \
     --output-dir /path/to/lerobot/goal-task-00

``--vla`` exports only VLA training data and is the default when omitted.
Export each robot separately. By default, the exporter compresses and saves one
image at a time (``--image-writer-threads 0``). To write images in parallel,
specify a thread count in the command above, for example
``--image-writer-threads 4``.

To export matching run directories whose names start with ``run_`` together:

.. code-block:: bash

   rpent-flywheel \
     --vla \
     --input-root /path/to \
     --pattern 'run_*' \
     --output-dir /path/to/lerobot/combined

The command searches these run directories and their subdirectories for
trajectories, applies the selection rules below, and writes the results to
``/path/to/lerobot/combined/train/``. Keep the quotes around ``run_*``.

The VLA exporter selects and organizes training data using the following rules:

- Success criterion: ``environment_success`` in the task result must explicitly
  be ``true``. Failed tasks, missing results, and tasks whose success cannot be
  confirmed are excluded.
- Action source: Only executed actions uniquely linked to a valid VLA prediction
  record are retained. The prediction must contain a non-empty instruction and
  belong to the same tool call as the action. The exporter skips non-VLA actions
  (such as actions the tool computes itself) and predicted actions that were
  never executed.
- Data organization: Each continuous segment is saved as a separate LeRobot
  episode. Each executed action corresponds to one frame containing the
  observation before execution, the action, and the corresponding VLA instruction.

The dataset is written to the ``train/`` subdirectory of ``--output-dir`` without
an automatic validation split. ``report.json`` in the output directory lists
export counts and exclusions. If no actions qualify, only the report is
written.

Export VLM Training Data
~~~~~~~~~~~~~~~~~~~~~~~~

Export recorded ``api`` planner turns to ``samples.jsonl`` and deduplicated
images:

.. code-block:: bash

   rpent-flywheel \
     --vlm \
     --input-root /path/to/run \
     --output-dir /path/to/vlm/goal-task-00

- ``--vlm``: Export recorded ``api`` planner turns as VLM training samples.
- ``--include-unsuccessful``: Optional (requires ``--vlm``). Also export turns
  from episodes whose environment outcome is failed or unknown, instead of
  requiring confirmed environment success.

The VLM exporter selects and formats samples using the following rules:

- Episode selection: By default, ``environment_success`` in the task result
  must explicitly be ``true``. Pass ``--include-unsuccessful`` to include failed
  runs or runs without an environment success verdict.
- Planner backend and turn validity: Only completed ``api`` planner calls with
  complete request and response payloads, a normal finish reason (excluding
  truncated, content-filtered, or errored replies), valid JSON-object tool
  arguments, and at least one visible text reply or tool call are exported.
  Turns from ``claude_code`` and ``codex`` are skipped.
- Sample organization and reasoning content: Each row pairs the recorded
  conversation history and tool definitions sent to the model with one assistant
  response to supervise. Results of tool calls made in that response belong to
  subsequent turns. When a model emits internal reasoning (thinking) content,
  that reasoning is stripped from both earlier assistant messages in the
  conversation history and the target response, leaving only visible text and
  tool calls.
- Image deduplication: Each distinct referenced image is copied once into
  ``images/`` (named by content hash) and referenced by its relative path in
  ``samples.jsonl``, so the exported directory is self-contained. Turns with
  missing image files are excluded.

The output directory has the following structure, without an automatic
validation split:

.. code-block:: text

   /path/to/vlm/goal-task-00/
   ├── samples.jsonl
   ├── images/
   └── report.json

``report.json`` in the output directory records export counts and excluded
turns. Each line in ``samples.jsonl`` is a JSON object with these main fields:

.. list-table:: ``samples.jsonl`` (one supervised planner turn per line)
   :header-rows: 1

   * - Field
     - Meaning
   * - ``episode_id``
     - Task ID (``task.task_id``) of the episode this turn comes from.
   * - ``turn_id``
     - Planner call ID (``planner_id``) within the episode.
   * - ``messages``
     - Conversation history sent to the model for this turn, containing
       ``system``, ``user``, ``assistant``, and ``tool`` messages with text,
       image references under ``images/``, and prior tool calls.
   * - ``tools``
     - Tool definitions exposed to the model on this turn.
   * - ``target``
     - The single ``assistant`` response to supervise, containing visible text
       and/or tool calls.

Train with RLinf
----------------

For the LIBERO example, use the LeRobot dataset exported by ``--vla`` for Pi0.5
supervised fine-tuning in a separate RLinf environment. RPent handles collection
and export; RLinf handles model training, logging, and checkpoint saving.

For environment setup and training configuration, see the
`RLinf OpenPI SFT guide <https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/sft_openpi.html>`_.
Set ``data.train_data_paths`` to the exported ``train/`` directory containing
``meta/info.json``.

Use an RLinf environment with LIBERO LeRobot data-loading support. Prepare a
Pi0.5 LIBERO SFT configuration at
``/path/to/sft-config/libero_pi05_sft.yaml``, then run:

.. code-block:: bash

   cd /path/to/RLinf
   unset PYTHONPATH
   export PYTHONPATH="$PWD"
   export EMBODIED_PATH="$PWD/examples/sft"

   .venv/bin/python examples/sft/train_vla_sft.py \
     --config-path /path/to/sft-config \
     --config-name libero_pi05_sft \
     data.train_data_paths=/path/to/lerobot/goal-task-00/train \
     actor.model.model_path=/path/to/pi05-checkpoint \
     runner.logger.log_path=/path/to/new-training-output

Match the configuration's resource placement and batch size to your available
GPUs, and use the checkpoint and normalization settings for the selected model
and dataset.

SFT Experiment Results
----------------------

The evaluations below currently report VLA fine-tuning results; comparative
evaluations for VLM fine-tuning are still in progress.

LIBERO-PRO VLA Fine-Tuning Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Successful trajectories collected with RPent were used to fine-tune
``RLinf-Pi05-LIBERO-130-fullshot-SFT``. The resulting VLA model was evaluated
against the original model within RPent.

Evaluation covered tasks 0–9 and seeds 25–34 in ``libero_goal_swap``, for 100 runs
per model. Both groups used the same DeepSeek-V4.1-Flash-1017 planner with reasoning
effort set to ``none`` and loaded task memory from ``libero/GPT_5.5_xhigh`` in the
``RLinf/RPent-memory`` dataset on Hugging Face.

Success rates:

.. list-table::
   :header-rows: 1

   * - Model
     - Successes / runs
     - Success rate
   * - Original model
     - 70/100
     - 70%
   * - SFT model (step 300)
     - 73/100
     - 73%

In this evaluation, the SFT model succeeded in three more runs, increasing the
success rate by three percentage points.

RoboTwin VLA Fine-Tuning Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Successful trajectories collected with RPent were used to fine-tune
``LingBot-VLA-RoboTwin-EEF-ckpt1500``. The resulting VLA model was evaluated
against the original model within RPent.

Evaluation covered the same 10 tasks used for training in RoboTwin's
``demo_randomized`` configuration, with five fixed test scenes per task and
50 runs per model. Both groups used the same DeepSeek-V4.1-Flash-1017 planner
with reasoning effort set to ``none`` and loaded the task memory generated by GPT-5.5
(``xhigh``) from ``robotwin/`` in the ``RLinf/RPent-memory`` dataset on Hugging Face.

Success rates:

.. list-table::
   :header-rows: 1

   * - Model
     - Successes / runs
     - Success rate
   * - Original model
     - 24/50
     - 48%
   * - SFT model (step 500)
     - 36/50
     - 72%

In this evaluation, the SFT model succeeded in 12 more runs, increasing the
task success rate from 48% to 72%, a gain of 24 percentage points.
