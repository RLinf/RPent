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
``/path/to/datacollection/raw/libero/<suite>/<task>/<seed>/``. Each episode
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

The exporter never rewrites the raw episodes. Failed episodes remain available
for auditing but are not included in this supervised-training export.

Train with RLinf
----------------

Use the Flywheel launcher with an official RLinf checkout, the exported
dataset, the initial Pi0.5 checkpoint, and a new output directory:

.. code-block:: bash

   rpent-flywheel train-rlinf \
     --dataset /path/to/lerobot/goal-task-00 \
     --checkpoint /path/to/pi05-checkpoint \
     --rlinf-root /path/to/RLinf \
     --output-dir /path/to/new-training-output \
     --max-steps 1000 \
     --save-interval 100 \
     --cuda-device 0

The launcher records the RLinf commit and invokes RLinf's native VLA SFT entry
point with the bundled Pi0.5 configuration. The output directory must not
already exist.
