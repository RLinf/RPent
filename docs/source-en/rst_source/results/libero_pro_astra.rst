:html_theme.sidebar_secondary.remove:

.. _libero-pro-astra-results:

LIBERO-PRO: GPT-6 Astra
=======================

:doc:`Back to the leaderboard <../benchmarks>`

**Codex / GPT-6 Astra / reasoning on / effort low.**
Verified snapshot: **2026-09-14 14:00:55 UTC**.

.. note::

   This is a partial evaluation: **619/800 episodes completed**, with
   **564 successes, 55 failures and 181 pending**. Six of eight suites are
   complete. Goal Task, Goal Swap and the full Overall rate remain
   **Not reported**. Pending episodes are not failures, and the success
   rate of a completed subset is not the 800-episode Overall.

Suite results
-------------

.. csv-table::
   :name: astra-suite-progress
   :header: "Suite", "Completed", "Success", "Failure", "Rate", "Status"
   :widths: 25 15 10 10 15 25
   :class: table-sm

   "Spatial Task", "100/100", "100", "0", "100.00%", "Complete"
   "Spatial Swap", "100/100", "98", "2", "98.00%", "Complete"
   "Object Task", "100/100", "100", "0", "100.00%", "Complete"
   "Object Swap", "100/100", "99", "1", "99.00%", "Complete"
   "Goal Task", "19/100", "10", "9", "Not reported", "Partial"
   "Goal Swap", "0/100", "0", "0", "Not reported", "Not started"
   "Long Task", "100/100", "85", "15", "85.00%", "Complete"
   "Long Swap", "100/100", "72", "28", "72.00%", "Complete"
   "Overall", "619/800", "564", "55", "Not reported", "181 pending"

Each suite has 10 tasks (IDs 0-9) evaluated on seeds 1-10, or 100 episodes.
Seed 0 is used only for exploration and is excluded from these counts.

Task and seed results
---------------------

``S`` = success; ``F`` = failure; ``-`` = no scored result yet. A task rate
is shown only after all ten evaluation seeds are complete. Task descriptions
retain the original experiment's ``task_language`` and task ordering.

Spatial Task
~~~~~~~~~~~~

**100/100 complete; 100 successes, 0 failures, 0 pending.**

.. csv-table::
   :name: astra-seeds-spatial-task
   :header: "Task", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Task result"
   :class: table-sm

   "0", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "1", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "2", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "3", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "4", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "5", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "6", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "7", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "8", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "9", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"

.. dropdown:: Task definitions

   .. csv-table::
      :header: "Task", "task_language"
      :widths: 10 90

      "0", "Pick the akita black bowl not between the plate and the ramekin and place it on the plate"
      "1", "Pick the akita black bowl next to the cookie box and place it on the plate"
      "2", "Pick the akita black bowl next to the plate and place it on the plate"
      "3", "Pick the akita black bowl on the top of the cabinet and place it on the plate"
      "4", "Pick the akita black bowl on the top of the wooden cabinet and place it on the plate"
      "5", "Pick the akita black bowl on the cookie box and place it on the plate"
      "6", "Pick the akita black bowl on the stove and place it on the plate"
      "7", "Pick the akita black bowl on the top of the cabinet and place it on the plate"
      "8", "Pick the akita black bowl next to the ramekin and place it on the plate"
      "9", "Pick the akita black bowl on the stove and place it on the plate"

Spatial Swap
~~~~~~~~~~~~

**100/100 complete; 98 successes, 2 failures, 0 pending.**

.. csv-table::
   :name: astra-seeds-spatial-swap
   :header: "Task", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Task result"
   :class: table-sm

   "0", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "1", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "2", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "3", "S", "F", "F", "S", "S", "S", "S", "S", "S", "S", "8/10 (80%)"
   "4", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "5", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "6", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "7", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "8", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "9", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"

.. dropdown:: Task definitions

   .. csv-table::
      :header: "Task", "task_language"
      :widths: 10 90

      "0", "Pick the akita black bowl between the plate and the ramekin and place it on the plate"
      "1", "Pick the akita black bowl next to the ramekin and place it on the plate"
      "2", "Pick the akita black bowl from table center and place it on the plate"
      "3", "Pick the akita black bowl on the cookies box and place it on the plate"
      "4", "Pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate"
      "5", "Pick the akita black bowl on the ramekin and place it on the plate"
      "6", "Pick the akita black bowl next to the cookies box and place it on the plate"
      "7", "Pick the akita black bowl on the stove and place it on the plate"
      "8", "Pick the akita black bowl next to the plate and place it on the plate"
      "9", "Pick the akita black bowl on the wooden cabinet and place it on the plate"

Object Task
~~~~~~~~~~~

**100/100 complete; 100 successes, 0 failures, 0 pending.**

.. csv-table::
   :name: astra-seeds-object-task
   :header: "Task", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Task result"
   :class: table-sm

   "0", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "1", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "2", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "3", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "4", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "5", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "6", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "7", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "8", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "9", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"

.. dropdown:: Task definitions

   .. csv-table::
      :header: "Task", "task_language"
      :widths: 10 90

      "0", "Pick the cream cheese and place it in the basket"
      "1", "Pick the alphabet soup and place it in the basket"
      "2", "Pick the tomato sauce and place it in the basket"
      "3", "Pick the ketchup and place it in the basket"
      "4", "Pick the milk and place it in the basket"
      "5", "Pick the bbq sauce and place it in the basket"
      "6", "Pick the orange juice and place it in the basket"
      "7", "Pick the butter and place it in the basket"
      "8", "Pick the salad dressing and place it in the basket"
      "9", "Pick the chocolate pudding and place it in the basket"

Object Swap
~~~~~~~~~~~

**100/100 complete; 99 successes, 1 failures, 0 pending.**

.. csv-table::
   :name: astra-seeds-object-swap
   :header: "Task", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Task result"
   :class: table-sm

   "0", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "1", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "2", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "3", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "4", "S", "S", "S", "S", "S", "S", "S", "S", "F", "S", "9/10 (90%)"
   "5", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "6", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "7", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "8", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "9", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"

.. dropdown:: Task definitions

   .. csv-table::
      :header: "Task", "task_language"
      :widths: 10 90

      "0", "Pick the alphabet soup and place it in the basket"
      "1", "Pick the cream cheese and place it in the basket"
      "2", "Pick the salad dressing and place it in the basket"
      "3", "Pick the bbq sauce and place it in the basket"
      "4", "Pick the ketchup and place it in the basket"
      "5", "Pick the tomato sauce and place it in the basket"
      "6", "Pick the butter and place it in the basket"
      "7", "Pick the milk and place it in the basket"
      "8", "Pick the chocolate pudding and place it in the basket"
      "9", "Pick the orange juice and place it in the basket"

Goal Task
~~~~~~~~~

**19/100 complete; 10 successes, 9 failures, 81 pending.**

.. csv-table::
   :name: astra-seeds-goal-task
   :header: "Task", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Task result"
   :class: table-sm

   "0", "F", "F", "F", "S", "F", "F", "F", "F", "F", "F", "1/10 (10%)"
   "1", "S", "S", "S", "S", "S", "S", "S", "S", "S", "``-``", "9/10 complete"
   "2", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "3", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "4", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "5", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "6", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "7", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "8", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "9", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"

.. dropdown:: Task definitions

   .. csv-table::
      :header: "Task", "task_language"
      :widths: 10 90

      "0", "open the bottom drawer of the cabinet"
      "1", "Put the plate on the stove"
      "2", "put the wine bottle in the bowl"
      "3", "Open the top layer of the drawer and put the cream cheese inside"
      "4", "Put the plate on the top of the drawer"
      "5", "Push the cream cheese to the front of the stove"
      "6", "put the wine bottle in the bowl"
      "7", "Turn off the stove"
      "8", "Put the wine bottle on the plate"
      "9", "Put the cream cheese on the rack"

Goal Swap
~~~~~~~~~

**0/100 complete; 0 successes, 0 failures, 100 pending.**

.. csv-table::
   :name: astra-seeds-goal-swap
   :header: "Task", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Task result"
   :class: table-sm

   "0", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "1", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "2", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "3", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "4", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "5", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "6", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "7", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "8", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"
   "9", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "0/10 complete"

.. dropdown:: Task definitions

   .. csv-table::
      :header: "Task", "task_language"
      :widths: 10 90

      "0", "Open the middle layer of the drawer"
      "1", "Put the bowl on the stove"
      "2", "Put the wine bottle on the top of the drawer"
      "3", "Open the top layer of the drawer and put the bowl inside"
      "4", "Put the bowl on the top of the drawer"
      "5", "Push the plate to the front of the stove"
      "6", "Put the cream cheese on the bowl"
      "7", "Turn on the stove"
      "8", "Put the bowl on the plate"
      "9", "Put the wine bottle on the rack"

Long Task
~~~~~~~~~

**100/100 complete; 85 successes, 15 failures, 0 pending.**

.. csv-table::
   :name: astra-seeds-long-task
   :header: "Task", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Task result"
   :class: table-sm

   "0", "S", "F", "S", "S", "S", "S", "S", "S", "F", "S", "8/10 (80%)"
   "1", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "2", "S", "S", "F", "S", "S", "S", "S", "S", "S", "S", "9/10 (90%)"
   "3", "F", "S", "F", "F", "S", "F", "S", "S", "S", "F", "5/10 (50%)"
   "4", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "5", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "6", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "7", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "8", "F", "F", "S", "S", "S", "S", "S", "S", "S", "S", "8/10 (80%)"
   "9", "S", "F", "S", "S", "F", "F", "S", "S", "F", "F", "5/10 (50%)"

.. dropdown:: Task definitions

   .. csv-table::
      :header: "Task", "task_language"
      :widths: 10 90

      "0", "put both the cream cheese and the tomato sauce in the basket"
      "1", "put both the alphabet soup and the butter in the basket"
      "2", "turn on the stove and put the pan on it"
      "3", "put the bottle in the bottom drawer of the cabinet and close it"
      "4", "put the yellow and white mug on the left plate and put the white mug on the right plate"
      "5", "pick up the cup and place it in the back compartment of the caddy"
      "6", "put the red mug on the plate and put the chocolate pudding to the right of the plate"
      "7", "put both the ketchup and the cream cheese box in the basket"
      "8", "put the left moka pot on the stove"
      "9", "put the white mug in the microwave and close it"

Long Swap
~~~~~~~~~

**100/100 complete; 72 successes, 28 failures, 0 pending.**

.. csv-table::
   :name: astra-seeds-long-swap
   :header: "Task", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Task result"
   :class: table-sm

   "0", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "1", "F", "S", "S", "S", "S", "F", "S", "S", "F", "S", "7/10 (70%)"
   "2", "S", "S", "S", "S", "S", "S", "S", "F", "S", "S", "9/10 (90%)"
   "3", "F", "S", "S", "S", "S", "S", "F", "S", "S", "S", "8/10 (80%)"
   "4", "S", "S", "S", "S", "S", "S", "S", "F", "S", "S", "9/10 (90%)"
   "5", "S", "F", "S", "S", "S", "S", "S", "S", "S", "F", "8/10 (80%)"
   "6", "F", "S", "F", "S", "F", "S", "S", "F", "F", "F", "4/10 (40%)"
   "7", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "8", "S", "S", "F", "S", "F", "S", "S", "F", "S", "F", "6/10 (60%)"
   "9", "F", "F", "S", "F", "F", "F", "F", "F", "F", "F", "1/10 (10%)"

.. dropdown:: Task definitions

   .. csv-table::
      :header: "Task", "task_language"
      :widths: 10 90

      "0", "put both the alphabet soup and the tomato sauce in the basket"
      "1", "put both the cream cheese box and the butter in the basket"
      "2", "turn on the stove and put the moka pot on it"
      "3", "put the black bowl in the bottom drawer of the cabinet and close it"
      "4", "put the white mug on the left plate and put the yellow and white mug on the right plate"
      "5", "pick up the book and place it in the back compartment of the caddy"
      "6", "put the white mug on the plate and put the chocolate pudding to the right of the plate"
      "7", "put both the alphabet soup and the cream cheese box in the basket"
      "8", "put both moka pots on the stove"
      "9", "put the yellow and white mug in the microwave and close it"

Protocol and provenance
-----------------------

Success is the original environment's ``states.json`` signal
``terminated = true``, not the planner's final message. Each task/seed
contributes at most one final scored result; existing valid failures are retained.
The report preserves all 800 planned positions, including unscored positions.

Runtime revision: `014a0fa <https://github.com/RLinf/RPent/commit/014a0fa97f69c991ee5e0f62f14e1d6f89c3dcd7>`_.
The evaluation uses a 5000-second planner timeout and a 10000-step environment
horizon. Long belongs to campaign ``libero_long_gpt6_astra_20260907``;
Spatial/Object/Goal belong to ``libero_pro_remaining_gpt6_astra_20260913``.
Each campaign uses its own seed-0 memory frozen before evaluation; the eight
suites do not share one memory snapshot. Their scores are not a controlled
comparison changing only the planner model. These protocol details are
experimental provenance, not new RPent defaults.

The contributor-supplied report was cross-checked for suite totals and the
619 scored task/seed positions before publication. Its SHA-256 is
``e9880e58953af964edbcdd50d586b54c22a66f3a7fab54d919a4205be9436fdf``.
The public `sanitized result snapshot <https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/astra-pro-20260914.json>`_
contains the task/seed outcomes, pending positions, counts and protocol metadata;
it excludes server paths, credentials and raw traces. This page reports supplied
experimental evidence, not a new policy evaluation performed while updating docs.
