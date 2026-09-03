Task Cards
==========

A plan recorded as absolute waypoints cannot follow an object that moved. The
same plan recorded together with *what was localized* and *how far each
waypoint sat from that reading* can: the offset is task logic and survives a
change of layout, the coordinate is not.

A **task card** is a plan in that second form, recorded once from a solved
episode. Replaying it substitutes only perception: the actions, their order,
the prompts given to the policy and the gripper commands are all the card's,
and the grounder supplies the coordinates they run at.

The corpus
----------

Cards live in ``resources/libero/task_card``, beside the curated memory under
``resources/libero/memory``, and travel with the rest of that payload -- synced
from the HuggingFace resources dataset, or read from a copy already on disk.
Like everything under ``resources/``, the directory is not tracked in git.

There is one card per task, so nothing is chosen at run time: the task names
the card, and the card plus live grounding produces the trajectory.

.. code-block:: text

   resources/libero/task_card/
     index.json                 every card: task, source episode, instruction
     object/swap_t3/
       anchors.json             phrases localized, their readings, the anchor each yields
       plan.json                every action, with the anchor and offset behind its coordinate
       trace.md                 a human-readable trace of the recorded actions

No seed appears in the corpus. A card serves its task whatever layout it is
replayed against; the episode it was recorded from is kept inside the card as
``source``, as provenance rather than as a knob.

Molmo configuration
-------------------

Replay grounds point-based anchors with **Molmo**, served by
``rpent/robots/components/molmo_server.py``. Where SAM3 answers "which pixels
are this phrase", Molmo answers "where would you put the gripper" -- an
open-vocabulary point, for phrases no mask proposal names.

Molmo installs into its own environment, separate from the LIBERO one. Create
it, install the ``molmo`` extra, then download the weights from
`Hugging Face: allenai/Molmo2-8B <https://huggingface.co/allenai/Molmo2-8B>`_
or `ModelScope: allenai/Molmo2-8B
<https://modelscope.cn/models/allenai/Molmo2-8B>`_ and point at them via
``MOLMO_CHECKPOINT_PATH``:

.. code-block:: bash

   uv venv --python 3.11 /path/to/molmo-venv
   /path/to/molmo-venv/bin/pip install -e ".[molmo]"

   # Hugging Face
   hf download allenai/Molmo2-8B --local-dir /path/to/Molmo2-8B

   # ModelScope (use this instead of the Hugging Face command above)
   modelscope download --model allenai/Molmo2-8B --local_dir /path/to/Molmo2-8B

   export MOLMO_CHECKPOINT_PATH=/path/to/Molmo2-8B

Serve it with that interpreter:

.. code-block:: bash

   PYTHONPATH=/path/to/RPent /path/to/molmo-venv/bin/python \
     rpent/robots/components/molmo_server.py \
     --transport http --host 127.0.0.1 --port 20703

Both entry points below take the server's address rather than starting it.

Replaying one episode
---------------------

``--planner task_card`` is a planner backend like ``api`` or ``codex``, except
that the card decides the actions and no model is called. Everything else about
the run is unchanged:

.. code-block:: bash

   rpent --robot libero --planner task_card \
     --suite libero_object_swap --task 3 --seed 0 \
     --molmo-endpoint http://127.0.0.1:20703

The corpus holds one card per task, so the seed selects the layout to solve,
never the plan used to solve it.

Replaying a whole sweep
-----------------------

A sweep is that same command over more cells. Point every endpoint at a
already-serving model so the sweep pays to load them once:

.. code-block:: bash

   for seed in $(seq 0 9); do
     rpent --robot libero --planner task_card \
       --suite libero_object_swap --task 3 --seed "$seed" \
       --output-dir logs/sweep/swap_t3_s$seed \
       --vla-endpoint http://127.0.0.1:20701 \
       --sam3-endpoint http://127.0.0.1:20702 \
       --molmo-endpoint http://127.0.0.1:20703
   done

   # how many solved
   grep -l '"status": "success"' logs/sweep/*/transcript_*.json | wc -l

Evaluation is single-attempt with no environment reset: a failed episode is
scored as failed, not retried from a clean state.

How a reading becomes a waypoint
--------------------------------

Each anchor is re-read **through the interface it was first read through**. A
``segment`` anchor is re-segmented, because its offsets are relative to a mask
centroid, and a wide container seen at an angle has that centroid some way from
where a pointing model points. Point-grounded anchors are answered by the
grounder.

Localization is two-stage: a coarse survey of the opening frame, then the arm
parks over each point-grounded anchor and asks again from the wrist, where the
object fills the view. The close reading is kept only when it agrees with the
coarse one within 5 cm, so a wrist view that found something else cannot
overwrite a correct answer.

A wrist reading of a *held* object lands short of its centre, further the
taller the object. The correction is linear in the object's measured height.

Configuration
-------------

There is nothing to configure beyond the usual LIBERO run. ``--suite`` /
``--task`` / ``--seed`` name the cell, and the card follows from the task.
The one required addition is ``--molmo-endpoint``: an already-serving
grounder. Molmo runs in a separate environment because its ``transformers``
requirement conflicts with the policy environment, so RPent does not start it
with the current Python interpreter.
