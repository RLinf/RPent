Task Cards
==========

A **task card** stores the action sequence for a LIBERO task and marks the key
objects or locations needed by those actions. During replay, RPent finds their
current coordinates in the camera images, updates the action coordinates, and
executes the recorded actions in order.

This keeps planning and perception separate:

1. The task card decides **what to do**.
2. SAM3 or Molmo finds **where to do it** in the current scene.
3. The LIBERO toolkit executes the actions at the updated coordinates.

``--planner task_card`` therefore does not call an LLM to make planning
decisions. Molmo is used only for visual localization: it points to a requested
object or location in a camera image so RPent can recover its current
coordinates.

LIBERO-PRO performance and execution time
-----------------------------------------

Across the complete 800-case LIBERO-PRO matrix (Spatial, Object, Goal, and Long;
task/swap; 10 seeds per task), Task Card solved 581 episodes (72.63%). Codex
without reasoning solved 500 (62.50%), while Codex with high reasoning solved
628 (78.50%). The two tasks without a successful source trace and therefore no
Task Card are conservatively counted as 0/10.

.. image:: ../../_static/task_card_libero_pro_performance_time.png
   :alt: Per-task success rate and execution-time comparison between Task Card and Codex on all LIBERO-PRO suites
   :width: 100%
   :align: center

The timing excludes model and service startup. Codex time is the mean planner
execution time over available records for each task. Task Card timing uses the
tool-execution time from the successful episode underlying each final card
(one timing sample per card). Success rates use the complete 800-case matrix for
every method. Both Codex baselines have planner duration records for all 800
cases.

How replay works
----------------

Each card contains an action plan and a set of anchors. An anchor describes a
task-relevant object or location, such as the object to pick or the destination
for a placement. Actions that depend on an anchor store their offset from that
anchor instead of relying only on an absolute coordinate.

At run time, RPent extracts the anchors required by the card and locates each
one with the interface recorded for it:

* **SAM3** locates segmentation anchors and returns an object mask and its
  position.
* **Molmo** locates point anchors by pointing to the requested object or
  location in the camera image.

RPent then combines each live anchor position with the offset stored in the
card and executes the resulting waypoint. This lets the same card run when
objects appear at different positions.

Task-card files
---------------

Task cards are distributed through the `RLinf/RPent-memory task-card directory
<https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/libero/task_card>`_
on Hugging Face rather than tracked in Git. RPent downloads them with the other
LIBERO memory and stores them locally under ``memory/libero/task_card``.
There is one card for each supported task.

.. code-block:: text

   memory/libero/task_card/
     object_swap_t3_anchors.json   objects and locations to locate at run time
     object_swap_t3_plan.json      actions and their anchor-relative coordinates

The task selects the card. The seed changes the environment layout, not the
card used for the task.

Generate a task card
--------------------

The generator creates one card from one simulator-verified successful episode.
Its two required inputs are the episode audit JSON and the matching primitive
recipe JSONL:

.. code-block:: bash

   python -m robots.libero.task_card.generate \
     --audit results/goal_swap_t3_s7.json \
     --recipe results/recipe_goal_swap_t3_s7.jsonl \
     --destination memory/libero/task_card

The audit must contain a non-empty ``task_language`` (or
``perturbed_task_language``) and ``libero_terminated: true``. The audit and
recipe filenames, plus the audit suite/task/seed fields when present, must
identify the same episode.

If ``segment_*.json`` readings were saved for the episode, pass their directory
with ``--segments``. Otherwise the generator derives semantic Molmo anchors from
the instruction and the recipe's ordered pick/release or articulation
transactions. Nearby ``move_to`` and ``move_pose`` coordinates are stored as XY
offsets from those anchors, in the format consumed by task-card replay.

The relation parser supports all 80 LIBERO-PRO tasks: Spatial, Object, Goal, and
Long (``10``), across both task and swap suites. Long instructions are preserved
as ordered transactions, including dependent actions such as turning on the
stove before placement or closing an appliance after insertion.

To download only the task cards manually, run:

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
     --include "libero/task_card/**" --local-dir memory

Run a task card
---------------

Start Molmo first, then pass its endpoint to RPent:

.. code-block:: bash

   rpent --robot libero --planner task_card \
     --suite libero_object_swap --task 3 --seed 0 \
     --molmo-endpoint http://127.0.0.1:20703

Task-card replay supports the task and swap suites for LIBERO-PRO Spatial,
Object, Goal, and Long (``10``), for 80 task identities in total.

The VLA and SAM3 services use the normal LIBERO runtime configuration. You can
also connect to services that are already running with ``--vla-endpoint`` and
``--sam3-endpoint``.

Molmo setup
-----------

Molmo requires a newer ``transformers`` version than the LIBERO policy
environment, so run it in a separate Python environment:

.. code-block:: bash

   uv venv --python 3.11 /path/to/molmo-venv
   /path/to/molmo-venv/bin/pip install -e ".[molmo]"

Download ``allenai/Molmo2-8B`` from `Hugging Face
<https://huggingface.co/allenai/Molmo2-8B>`_ or `ModelScope
<https://modelscope.cn/models/allenai/Molmo2-8B>`_, then start the service:

.. code-block:: bash

   export MOLMO_CHECKPOINT_PATH=/path/to/Molmo2-8B
   PYTHONPATH=/path/to/RPent /path/to/molmo-venv/bin/python \
     rpent/robots/components/molmo_server.py \
     --transport http --host 127.0.0.1 --port 20703

Replay multiple layouts
-----------------------

Run the same task card with different seeds to evaluate it on different
layouts. Reusing existing VLA, SAM3, and Molmo services avoids loading the
models again for every run:

.. code-block:: bash

   for seed in $(seq 0 9); do
     rpent --robot libero --planner task_card \
       --suite libero_object_swap --task 3 --seed "$seed" \
       --output-dir logs/sweep/swap_t3_s$seed \
       --vla-endpoint http://127.0.0.1:20701 \
       --sam3-endpoint http://127.0.0.1:20702 \
       --molmo-endpoint http://127.0.0.1:20703
   done
