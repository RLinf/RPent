LIBERO Data Flywheel
====================

The optional Flywheel records executed LIBERO evaluation trajectories without
changing the planner or action primitives. It stores immutable raw episodes;
conversion to a training format is a separate step.

Collect an episode
------------------

Enable collection on a normal LIBERO evaluation run and choose a data root:

.. code-block:: bash

   rpent --robot libero \
     --suite libero_goal --task 0 --seed 0 \
     --planner codex \
     --collect-flywheel-data \
     --flywheel-root /path/to/datacollection

The run writes one episode below
``/path/to/datacollection/raw/libero/<suite>/task_<id>/seed_<seed>/``. Each episode
contains the policy observations, executed actions, rewards, terminal flags,
primitive IDs, and VLA proposals. Collection is opt-in and is supported only
for evaluation mode.

Validate and export successful episodes
---------------------------------------

Validate one raw episode before using it:

.. code-block:: bash

   rpent-flywheel validate /path/to/raw/episode

Export every finalized successful episode for one suite and task to a LeRobot
dataset:

.. code-block:: bash

   rpent-flywheel export-lerobot \
     --data-root /path/to/datacollection \
     --suite libero_goal \
     --task 0 \
     --dataset-id goal-task-00 \
     --output-root /path/to/lerobot

``--data-root`` is the raw data root, while ``--output-root`` is the parent
directory of the exported dataset. This example writes the dataset to
``/path/to/lerobot/goal-task-00``.

The training prefix of each successful episode includes all executed actions
through the first successful action, including actions from scripted primitives
and VLA calls. Actions after the first success are excluded from the export.

The exporter never rewrites the raw episodes. Failed episodes remain available
for auditing but are not included in this supervised-training export.

Train with RLinf
----------------

After exporting successful trajectories to LeRobot, you can perform Pi0.5
supervised fine-tuning in a separate RLinf environment. RPent handles collection
and export; RLinf handles model training, logging, and checkpoint saving.

For environment setup and training configuration, see the
`RLinf OpenPI_RLinf SFT guide <https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/sft_openpi_rlinf.html>`_.
Point the training data path to the exported LeRobot dataset directory containing
``meta/info.json``, not the raw episode directory.

With a Pi0.5 LIBERO SFT configuration and the corresponding RLinf environment
prepared, run training directly:

.. code-block:: bash

   cd /path/to/RLinf
   unset PYTHONPATH
   export PYTHONPATH="$PWD"
   export EMBODIED_PATH="$PWD/examples/sft"

   .venv/bin/python examples/sft/train_vla_sft.py \
     --config-path /path/to/sft-config \
     --config-name libero_pi05_sft \
     data.train_data_paths=/path/to/lerobot/goal-task-00 \
     actor.model.model_path=/path/to/pi05-checkpoint \
     runner.logger.log_path=/path/to/new-training-output

``/path/to/sft-config/libero_pi05_sft.yaml`` is a user-prepared training
configuration, not a file bundled with RPent or provided by upstream RLinf.
Visible GPUs must match the resource placement in the configuration; batch size,
initial weights, and normalization settings must also be compatible.

Our validated environment used RLinf ``e3977b9`` with local adaptations for LIBERO
data loading and action-module SFT.
