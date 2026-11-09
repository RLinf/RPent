.. _libero-data-flywheel:

Data Flywheel
=============

Use ``--record-trajectory`` with the ``api``, ``claude_code``, or ``codex`` planner
to save task, planner, and tool-call records. Supported environments also record executed
actions and their observations for later export to LeRobot training data.

Installation
------------

The example below uses LIBERO-PRO. Follow :doc:`../get_started/quickstart` for
environment setup and simulator assets, and :doc:`configure_planner` for API
credentials. From the RPent repository root, install LIBERO-PRO with the
optional export dependencies:

.. code-block:: bash

   uv pip install -e ".[libero-pro,flywheel]"

Record a Trajectory
-------------------

Recording is off by default. Recording with ``--planner claude_code`` requires
``claude-agent-sdk`` 0.2.154 or later and Claude Code 2.1.274 or later.
LIBERO, RoboCasa365, RoboTwin, and RoboDojo record executed actions and observations
only in evaluation mode. Exploration still saves task,
planner, and tool-call records, but contains no actions eligible for VLA dataset
export.

RoboDojo's native motion tools record scripted actions. Its current planner
toolkit does not expose a VLA tool, so these runs do not produce VLA training
segments for ``rpent-flywheel``.

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

``task.json`` stores the task instruction, run configuration, start and end times,
and outcome. Its top-level ``schema_version`` is ``5``. Each line in the three
JSONL files holds one JSON object. The examples below come from a LIBERO
trajectory recorded with the Codex planner.

.. list-table:: ``planners.jsonl`` (one model call per line)
   :header-rows: 1

   * - Field
     - Meaning
   * - ``planner_id``
     - Unique ID of this model call within the trajectory, such as ``P14``.
   * - ``backend``
     - Planner backend that made the call (``api``, ``claude_code``, or ``codex``).
   * - ``status``
     - State of this model call (``open``, ``completed``, ``failed``, or ``cancelled``).
   * - ``request``
     - Input to the model, including the conversation context and available tools.
       The structure depends on the backend.
   * - ``response``
     - Model output, which can include text or requests to call tools.

.. list-table:: ``predictions.jsonl`` (one VLA prediction per line)
   :header-rows: 1

   * - Field
     - Meaning
   * - ``prediction_id``
     - Unique ID of this prediction within the trajectory, such as ``V1``.
   * - ``instruction``
     - Instruction the VLA used for this prediction, such as
       ``pick up the green salad dressing bottle``.
   * - ``action_ids``
     - IDs of the actions executed from this prediction, in execution order.
       For example, look up ``A226`` through ``A230`` in ``actions.jsonl`` to find
       the control values executed from prediction ``V1``.

.. list-table:: ``actions.jsonl`` (one executed environment action per line)
   :header-rows: 1

   * - Field
     - Meaning
   * - ``action_id``
     - Unique ID of this action within the trajectory, such as ``A226``.
   * - ``source``
     - ``vla`` for a policy prediction, ``tool_computed`` for an action computed
       by a tool or script, or ``direct`` for a control value submitted through
       the direct action tool.
   * - ``action.format``
     - Robot action format, here ``libero.action``.
   * - ``action.data``
     - Control values actually executed, here an array of seven ``float32`` values.
   * - ``before_observation``
     - Environment observation before this action. The example contains the robot
       state, task description, and image references for the main and wrist cameras.

Export Training Data
--------------------

Export recorded trajectories from one robot to a new output directory:

.. code-block:: bash

   rpent-flywheel \
     --vla \
     --input-root /path/to/run \
     --output-dir /path/to/lerobot/goal-task-00

``--vla`` exports only VLA training data. This is also the default when the
flag is omitted.

``--input-root`` accepts a trajectory directory containing ``task.json`` or a
parent directory containing multiple runs. Export each robot separately;
``--output-dir`` must not already exist.

RoboTwin supports both LingBot's ``robotwin.action.ee`` format and Fast-WAM's
``robotwin.action.qpos`` format. The exporter automatically selects the matching
state and action layout from the recorded action format. EE data uses 16-dimensional
state and action vectors; qpos data uses the 14-dimensional joint targets in
``robot_state.qpos_target14`` as state and preserves the 14-dimensional executed
actions.

If several runs are saved in separate directories like these, you can export
them together:

.. code-block:: text

   /path/to/
   ├── run_001/trajectories/<task_id_1>/
   └── run_002/trajectories/<task_id_2>/

Set ``--input-root`` to their parent directory and use ``--pattern`` to select
run directories whose names start with ``run_``:

.. code-block:: bash

   rpent-flywheel \
     --vla \
     --input-root /path/to \
     --pattern 'run_*' \
     --output-dir /path/to/lerobot/combined

The command searches these run directories and their subdirectories for
trajectories, applies the selection rules below, and writes the results to
``/path/to/lerobot/combined/train/``. Keep the quotes around ``run_*``.

Add ``--image-writer-threads 4`` to compress and save images with four threads.
The default, ``0``, writes images synchronously.

Only tasks confirmed successful by the environment are eligible; planner
completion alone is insufficient. The exporter selects and organizes training
data using the following rules:

- Success criterion: ``environment_success`` in the task result must explicitly
  be ``true``. Failed tasks, missing results, and tasks whose success cannot be
  confirmed are excluded.
- Action source: Only executed actions uniquely linked to a valid VLA prediction
  record are retained. The prediction must contain a non-empty instruction and
  belong to the same tool call as the action. Scripted actions, other non-VLA
  actions, and unexecuted predicted actions are excluded.
- Data organization: Each continuous segment is saved as a separate LeRobot
  episode. Each executed action corresponds to one frame containing the
  observation before execution, the action, and the corresponding VLA instruction.

The dataset is written to the ``train/`` subdirectory of ``--output-dir`` without
an automatic validation split. ``report.json`` in the output directory lists
export counts and exclusions. If no actions qualify, only the report is
written. Original recordings are left unchanged.

Train with RLinf
----------------

For the LIBERO example, use the exported LeRobot dataset for Pi0.5 supervised
fine-tuning in a separate RLinf environment. RPent handles collection and
export; RLinf handles model training, logging, and checkpoint saving.

For environment setup and training configuration, see the
`RLinf OpenPI SFT guide <https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/sft_openpi.html>`_.
Point the training data path to the exported LeRobot dataset directory containing
``meta/info.json``, not the raw episode directory.

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
per model. Both groups used the same DeepSeek-V4.1-Flash-1017 planner (reasoning
effort: ``none``) and official memory, with only the VLA model replaced.

Task success rates for the full RPent system were:

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
50 runs per model. Task success rates for the full RPent system were:

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
