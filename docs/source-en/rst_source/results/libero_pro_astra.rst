:html_theme.sidebar_secondary.remove:

.. _libero-pro-astra-results:

LIBERO-PRO: GPT-6 Astra
=======================

:doc:`Back to the leaderboard <../benchmarks>`

**Codex / GPT-6 Astra / reasoning on / effort low.**
Verified snapshot: **2026-09-14 14:00:55 UTC**.

.. note::

   Only **six completed suites, 600 episodes** are published here:
   **554 successes and 46 failures**. Goal Task, Goal Swap and the full
   eight-suite Overall remain **Not reported** until completion. The partial
   progress of unfinished suites is excluded; this subset is not the full Overall.

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
   "Long Task", "100/100", "85", "15", "85.00%", "Complete"
   "Long Swap", "100/100", "72", "28", "72.00%", "Complete"
   "Six completed suites", "600/600", "554", "46", "Not full Overall", "Complete"

Each suite has 10 tasks (IDs 0-9) evaluated on seeds 1-10, or 100 episodes.
Seed 0 is used only for exploration and is excluded from these counts.

Task and seed results
---------------------

``S`` = success; ``F`` = failure. All ten evaluation seeds are complete for
every task shown here. Task descriptions
retain the original experiment's ``task_language`` and task ordering.

Spatial Task
~~~~~~~~~~~~

**100/100 complete; 100 successes, 0 failures.**

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

**100/100 complete; 98 successes, 2 failures.**

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

**100/100 complete; 100 successes, 0 failures.**

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

**100/100 complete; 99 successes, 1 failures.**

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

Long Task
~~~~~~~~~

**100/100 complete; 85 successes, 15 failures.**

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

**100/100 complete; 72 successes, 28 failures.**

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
This publication preserves exactly the 600 positions in the six completed suites.

Runtime revision: `014a0fa <https://github.com/RLinf/RPent/commit/014a0fa97f69c991ee5e0f62f14e1d6f89c3dcd7>`_.
The evaluation uses a 5000-second planner timeout and a 10000-step environment
horizon. Long belongs to campaign ``libero_long_gpt6_astra_20260907``;
Spatial/Object belong to ``libero_pro_remaining_gpt6_astra_20260913``.
Each campaign uses its own seed-0 memory frozen before evaluation; the six
published suites do not share one memory snapshot. Their scores are not a controlled
comparison changing only the planner model. These protocol details are
experimental provenance, not new RPent defaults.

The contributor-supplied report was cross-checked for suite totals and the
600 scored task/seed positions before publication. Its SHA-256 is
``e9880e58953af964edbcdd50d586b54c22a66f3a7fab54d919a4205be9436fdf``.
The public `sanitized result snapshot <https://raw.githubusercontent.com/RLinf/misc/e858f627dbcc1b35440c3cb5ecaaefd016eb8680/rpent/benchmarks/astra-pro-20260914.json>`_
contains the 600 task/seed outcomes, counts and protocol metadata;
it excludes server paths, credentials and raw traces. This page reports supplied
experimental evidence, not a new policy evaluation performed while updating docs.
