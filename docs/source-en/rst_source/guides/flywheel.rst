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

``rpent-flywheel`` exports VLA or VLM training data:

- ``--vla`` (default): Exports executed VLA action segments from confirmed
  successful trajectories to a LeRobot dataset for supervised fine-tuning with
  RLinf.
- ``--vlm``: Exports recorded ``api`` planner turns to ``samples.jsonl`` and a
  deduplicated ``images/`` directory for VLM training.

Fine-tuning on exported trajectories raised task success rates in both VLA
(LIBERO-PRO, RoboTwin) and VLM (RoboTwin) experiments (see SFT Experiment
Results below).

Installation
------------

Exporting VLM samples (``--vlm``) works with the base RPent installation.
Exporting VLA datasets (``--vla``) requires the optional ``flywheel`` extra,
which installs LeRobot.

The example below uses LIBERO-PRO. Follow :doc:`../get_started/quickstart` for
environment setup and simulator assets, and :doc:`configure_planner` for API
credentials. From the RPent repository root, install LIBERO-PRO with the
optional VLA export dependencies:

.. code-block:: bash

   uv pip install -e ".[libero-pro,flywheel]"

Record a Trajectory
-------------------

Recording is off by default. Recording with ``--planner claude_code`` requires
``claude-agent-sdk 0.2.154+`` and ``Claude Code 2.1.274+``.
LIBERO, RoboCasa365, RoboTwin, and RoboDojo record executed actions and observations
only in evaluation mode. To export VLM training samples, record with
``--planner api``; ``rpent-flywheel --vlm`` supports only this backend.

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
subdirectory. See :ref:`trajectory-data-format` for the file layout and fields.

Export Training Data
--------------------

``rpent-flywheel`` runs in either ``--vla`` or ``--vlm`` mode. Both modes share
these input and output rules:

- ``--input-root`` accepts a single trajectory directory containing
  ``task.json`` or a parent directory containing multiple runs. When runs live
  in separate subdirectories, pass their parent directory and filter with
  ``--pattern`` (quote shell wildcards, for example ``--pattern 'run_*'``):

  .. code-block:: text

     /path/to/
     ├── run_001/trajectories/<task_id_1>/
     └── run_002/trajectories/<task_id_2>/

- ``--output-dir`` must not exist, and original recordings remain unchanged.
- Neither mode creates an automatic validation split. Both write
  ``report.json`` in ``--output-dir`` with export counts and exclusions.

Export VLA Training Data
~~~~~~~~~~~~~~~~~~~~~~~~

Export recorded trajectories from one robot:

.. code-block:: bash

   rpent-flywheel \
     --vla \
     --input-root /path/to/run \
     --output-dir /path/to/lerobot/goal-task-00

Export each robot separately. By default, images are compressed and written one
at a time (``--image-writer-threads 0``); pass a thread count such as
``--image-writer-threads 4`` to write images in parallel.

To aggregate run directories whose names start with ``run_``:

.. code-block:: bash

   rpent-flywheel \
     --vla \
     --input-root /path/to \
     --pattern 'run_*' \
     --output-dir /path/to/lerobot/combined

This searches matching run directories and their subdirectories, applies the
rules below, and writes the dataset to ``/path/to/lerobot/combined/train/``.

VLA export rules:

- Success criterion: ``environment_success`` in the task result must explicitly
  be ``true``. Failed tasks, missing results, and tasks whose success cannot be
  confirmed are excluded.
- Action source: Only executed actions uniquely linked to a valid VLA prediction
  record are retained. The prediction must contain a non-empty instruction and
  belong to the same tool call as the action. Non-VLA actions (such as actions
  computed by tools) and predicted actions that were never executed are skipped.
- Data organization: Each continuous segment is saved as a separate LeRobot
  episode. Each executed action corresponds to one frame containing the
  observation before execution, the action, and the corresponding VLA instruction.

The dataset is written under ``<output_dir>/train/``. If no actions qualify,
only ``report.json`` is written.

Export VLM Training Data
~~~~~~~~~~~~~~~~~~~~~~~~

Export recorded ``api`` planner turns to ``samples.jsonl`` and deduplicated
images:

.. code-block:: bash

   rpent-flywheel \
     --vlm \
     --input-root /path/to/run \
     --output-dir /path/to/vlm/goal-task-00

VLM export rules:

- Trajectory selection: By default, ``environment_success`` in the task result
  must explicitly be ``true``. Pass ``--include-unsuccessful`` (requires
  ``--vlm``) to include failed runs or runs without an environment success
  verdict.
- Backend and turn validity: Export includes only completed ``api`` planner
  turns and skips ``claude_code`` and ``codex``. Each turn must have complete
  request and response payloads, a normal finish reason (not truncated,
  content-filtered, or errored), valid JSON-object tool arguments, and at least
  one visible text reply or tool call.
- Sample layout and reasoning: Each row pairs the recorded conversation history
  and tool definitions sent to the model with one assistant response to
  supervise. Internal reasoning content is removed from both earlier assistant
  messages and the target response, leaving only visible text and tool calls.
- Image deduplication: Each distinct referenced image is copied once into
  ``images/`` (named by content hash) and referenced by its relative path in
  ``samples.jsonl``, making the output directory self-contained. Turns with
  missing image files are excluded.

Output layout:

.. code-block:: text

   /path/to/vlm/goal-task-00/
   ├── samples.jsonl
   ├── images/
   └── report.json

.. dropdown:: ``samples.jsonl`` main fields (one supervised planner turn per line)

   .. list-table::
      :header-rows: 1

      * - Field
        - Meaning
      * - ``episode_id``
        - Task ID (``task.task_id``) of the trajectory this turn comes from.
      * - ``turn_id``
        - Planner turn ID (``planner_id``) within the trajectory.
      * - ``messages``
        - Conversation history sent to the model for this turn, containing
          ``system``, ``user``, ``assistant``, and ``tool`` messages with text,
          image references under ``images/``, and prior tool calls.
      * - ``tools``
        - Tool definitions exposed to the model on this turn.
      * - ``target``
        - The single ``assistant`` response to supervise, containing visible text
          or tool calls.

Load ``samples.jsonl`` and ``images/`` into any VLM fine-tuning framework that
supports multimodal conversations and tool calls.

.. _train-with-rlinf:

Train a VLA with RLinf
----------------------

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

RoboTwin VLM Fine-Tuning Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Successful trajectories collected with RPent and exported with
``rpent-flywheel --vlm`` were used to fine-tune ``Qwen3.8-27B``. The resulting
VLM planner was evaluated against the original model within RPent.

Evaluation covered the same 10 tasks used for training in RoboTwin's
``demo_randomized`` configuration, with five seeds per task and 50 runs per
model; 10 of the 50 task-seed pairs overlap with the training trajectories and
40 do not. Both groups used ``LingBot-VLA-RoboTwin-EEF-ckpt1500`` as the VLA
model, set planner reasoning effort to ``medium``, and loaded task memory from
``robotwin/`` in the ``RLinf/RPent-memory`` dataset on Hugging Face.

Success rates:

.. list-table::
   :header-rows: 1

   * - Evaluation slice
     - Original model
     - SFT model (step 48)
   * - All 50 task-seeds
     - 8/50 (16%)
     - 15/50 (30%)
   * - Overlapping with training (10)
     - 5/10 (50%)
     - 8/10 (80%)
   * - Non-overlapping (40)
     - 3/40 (7.5%)
     - 7/40 (17.5%)

.. _trajectory-data-format:

Trajectory Data Format
----------------------

Each recorded trajectory uses this directory layout:

.. code-block:: text

   /path/to/run/trajectories/<task_id>/
   ├── task.json
   ├── planners.jsonl
   ├── predictions.jsonl
   ├── actions.jsonl
   └── artifacts/

.. dropdown:: ``task.json``: instruction, configuration, and outcome of one task run

   .. list-table::
      :header-rows: 1

      * - Field
        - Meaning
      * - ``task.task_id``
        - Unique ID for this run, used as the trajectory directory name ``<task_id>``.
      * - ``task.instruction``
        - Task instruction for this run.
      * - ``task.config``
        - Run configuration stored in ``task.config.data``, including CLI and run
          settings, the system prompt, and registered tool definitions.
      * - ``task.result``
        - Run outcome. ``task.result.data.environment_success`` indicates whether the task succeeded.

.. dropdown:: ``planners.jsonl``: one model call per line

   .. list-table::
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

.. dropdown:: ``predictions.jsonl``: one VLA prediction per line

   .. list-table::
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

.. dropdown:: ``actions.jsonl``: one executed environment action per line

   .. list-table::
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
