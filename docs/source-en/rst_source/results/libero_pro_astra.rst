:html_theme.sidebar_secondary.remove:

.. _libero-pro-astra-results:

LIBERO-PRO: GPT-6 Astra
==========================================================================================

.. raw:: html

   <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/RLinf/misc@c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/docs.css">

:doc:`Back to the leaderboard <../benchmarks>`

**Codex / GPT-6 Astra / low / reasoning.**

Verified 2026-09-15 09:32:29 UTC: **741 successes, 59 failures, 800 episodes; Overall 92.63% (741/800).**

Suite results
------------------------------------------------------------------------------------------

.. _astra-suite-progress:

.. csv-table::
   :header: "Suite", "Success / evaluated", "Failure", "Success rate"
   :class: table-sm

   "Object Task", "100/100", "0", "100.00%"
   "Spatial Task", "100/100", "0", "100.00%"
   "Goal Swap", "99/100", "1", "99.00%"
   "Object Swap", "99/100", "1", "99.00%"
   "Spatial Swap", "98/100", "2", "98.00%"
   "Goal Task", "88/100", "12", "88.00%"
   "Long Task", "85/100", "15", "85.00%"
   "Long Swap", "72/100", "28", "72.00%"
   "Overall", "741/800", "59", "92.63%"

Each suite has ten tasks (IDs 0–9) and seeds 1–10. Seed 0 is exploration only. S = success; F = failure. All 800 positions are scored once; valid failures are retained. Task rows follow ascending original task ID; seed columns keep their original order. Method leaderboards remain ranked by success rate.

GPT-6 Astra · low · reasoning · Object Task · 100/100
----------------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "Task", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "Success rate"
   :class: table-sm

   "0", "Pick the cream cheese and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "1", "Pick the alphabet soup and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "2", "Pick the tomato sauce and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "3", "Pick the ketchup and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "4", "Pick the milk and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "5", "Pick the bbq sauce and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "6", "Pick the orange juice and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "7", "Pick the butter and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "Pick the salad dressing and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "9", "Pick the chocolate pudding and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"

GPT-6 Astra · low · reasoning · Spatial Task · 100/100
------------------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "Task", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "Success rate"
   :class: table-sm

   "0", "Pick the akita black bowl not between the plate and the ramekin and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "1", "Pick the akita black bowl next to the cookie box and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "2", "Pick the akita black bowl next to the plate and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "3", "Pick the akita black bowl on the top of the cabinet and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "4", "Pick the akita black bowl on the top of the wooden cabinet and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "5", "Pick the akita black bowl on the cookie box and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "6", "Pick the akita black bowl on the stove and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "7", "Pick the akita black bowl on the top of the cabinet and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "Pick the akita black bowl next to the ramekin and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "9", "Pick the akita black bowl on the stove and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"

GPT-6 Astra · low · reasoning · Goal Swap · 99/100
----------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "Task", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "Success rate"
   :class: table-sm

   "0", "Open the middle layer of the drawer", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "1", "Put the bowl on the stove", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "2", "Put the wine bottle on the top of the drawer", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "3", "Open the top layer of the drawer and put the bowl inside", "F", "S", "S", "S", "S", "S", "S", "S", "S", "S", "90%"
   "4", "Put the bowl on the top of the drawer", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "5", "Push the plate to the front of the stove", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "6", "Put the cream cheese on the bowl", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "7", "Turn on the stove", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "Put the bowl on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "9", "Put the wine bottle on the rack", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"

GPT-6 Astra · low · reasoning · Object Swap · 99/100
--------------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "Task", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "Success rate"
   :class: table-sm

   "0", "Pick the alphabet soup and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "1", "Pick the cream cheese and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "2", "Pick the salad dressing and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "3", "Pick the bbq sauce and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "4", "Pick the ketchup and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "F", "S", "90%"
   "5", "Pick the tomato sauce and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "6", "Pick the butter and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "7", "Pick the milk and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "Pick the chocolate pudding and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "9", "Pick the orange juice and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"

GPT-6 Astra · low · reasoning · Spatial Swap · 98/100
----------------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "Task", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "Success rate"
   :class: table-sm

   "0", "Pick the akita black bowl between the plate and the ramekin and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "1", "Pick the akita black bowl next to the ramekin and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "2", "Pick the akita black bowl from table center and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "3", "Pick the akita black bowl on the cookies box and place it on the plate", "S", "F", "F", "S", "S", "S", "S", "S", "S", "S", "80%"
   "4", "Pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "5", "Pick the akita black bowl on the ramekin and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "6", "Pick the akita black bowl next to the cookies box and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "7", "Pick the akita black bowl on the stove and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "Pick the akita black bowl next to the plate and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "9", "Pick the akita black bowl on the wooden cabinet and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"

GPT-6 Astra · low · reasoning · Goal Task · 88/100
----------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "Task", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "Success rate"
   :class: table-sm

   "0", "open the bottom drawer of the cabinet", "F", "F", "F", "S", "F", "F", "F", "F", "F", "F", "10%"
   "1", "Put the plate on the stove", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "2", "put the wine bottle in the bowl", "F", "S", "S", "S", "S", "F", "S", "S", "S", "S", "80%"
   "3", "Open the top layer of the drawer and put the cream cheese inside", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "4", "Put the plate on the top of the drawer", "S", "S", "S", "S", "S", "F", "S", "S", "S", "S", "90%"
   "5", "Push the cream cheese to the front of the stove", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "6", "put the wine bottle in the bowl", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "7", "Turn off the stove", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "Put the wine bottle on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "9", "Put the cream cheese on the rack", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"

GPT-6 Astra · low · reasoning · Long Task · 85/100
----------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "Task", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "Success rate"
   :class: table-sm

   "0", "put both the cream cheese and the tomato sauce in the basket", "S", "F", "S", "S", "S", "S", "S", "S", "F", "S", "80%"
   "1", "put both the alphabet soup and the butter in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "2", "turn on the stove and put the pan on it", "S", "S", "F", "S", "S", "S", "S", "S", "S", "S", "90%"
   "3", "put the bottle in the bottom drawer of the cabinet and close it", "F", "S", "F", "F", "S", "F", "S", "S", "S", "F", "50%"
   "4", "put the yellow and white mug on the left plate and put the white mug on the right plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "5", "pick up the cup and place it in the back compartment of the caddy", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "6", "put the red mug on the plate and put the chocolate pudding to the right of the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "7", "put both the ketchup and the cream cheese box in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "put the left moka pot on the stove", "F", "F", "S", "S", "S", "S", "S", "S", "S", "S", "80%"
   "9", "put the white mug in the microwave and close it", "S", "F", "S", "S", "F", "F", "S", "S", "F", "F", "50%"

GPT-6 Astra · low · reasoning · Long Swap · 72/100
----------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "Task", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "Success rate"
   :class: table-sm

   "0", "put both the alphabet soup and the tomato sauce in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "1", "put both the cream cheese box and the butter in the basket", "F", "S", "S", "S", "S", "F", "S", "S", "F", "S", "70%"
   "2", "turn on the stove and put the moka pot on it", "S", "S", "S", "S", "S", "S", "S", "F", "S", "S", "90%"
   "3", "put the black bowl in the bottom drawer of the cabinet and close it", "F", "S", "S", "S", "S", "S", "F", "S", "S", "S", "80%"
   "4", "put the white mug on the left plate and put the yellow and white mug on the right plate", "S", "S", "S", "S", "S", "S", "S", "F", "S", "S", "90%"
   "5", "pick up the book and place it in the back compartment of the caddy", "S", "F", "S", "S", "S", "S", "S", "S", "S", "F", "80%"
   "6", "put the white mug on the plate and put the chocolate pudding to the right of the plate", "F", "S", "F", "S", "F", "S", "S", "F", "F", "F", "40%"
   "7", "put both the alphabet soup and the cream cheese box in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "put both moka pots on the stove", "S", "S", "F", "S", "F", "S", "S", "F", "S", "F", "60%"
   "9", "put the yellow and white mug in the microwave and close it", "F", "F", "S", "F", "F", "F", "F", "F", "F", "F", "10%"

Protocol and provenance
------------------------------------------------------------------------------------------

Success is the original environment trace signal ``terminated = true``, not the planner message. Runtime revision is ``014a0fa97f69c991ee5e0f62f14e1d6f89c3dcd7``; the planner limit is 5000 seconds and the environment horizon is 10000 steps. These describe the experiment, not new runtime defaults.

Long belongs to ``libero_long_gpt6_astra_20260907``; Spatial/Object/Goal belong to ``libero_pro_remaining_gpt6_astra_20260913``. The two batches use separate seed-0 memory corpora frozen before evaluation; they do not share a single snapshot.

All 800 unique outcomes and task instructions were cross-checked against the contributor report and its verified JSON. The previously published 600 outcomes are unchanged. Percentages use exact counts; 741/800 is rounded half up to 92.63%, not reconstructed from a one-decimal summary.

Report SHA-256: ``627d6d8a95fb1e10357961694867cd0bf0a2aa9011c96c8a33504b904cb96c80``.

Verified JSON SHA-256: ``60b675da35557900f921befd84db306514723ebe5a9db73af2c72ae38bc46578``.

`Sanitized public snapshot <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_.
It contains task instructions, seed outcomes, counts and public protocol metadata, without server paths, credentials or raw traces. No experiments were rerun for this publication.
