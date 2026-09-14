:html_theme.sidebar_secondary.remove:

.. _benchmark-results:
.. _benchmark-leaderboard:
.. _leaderboard:

RPent Leaderboard
=================

.. raw:: html

   <div id="rpent-interactive-leaderboard" data-language="en"
        data-results-url="https://raw.githubusercontent.com/RLinf/misc/e858f627dbcc1b35440c3cb5ecaaefd016eb8680/rpent/benchmarks/results.json">
     <div class="rpent-static-leaderboard">
       <p>Full result tables are available below.</p>
     </div>
   </div>

**GPT-6 Astra: six completed suites, 600 episodes.**
:doc:`Task/seed results <results/libero_pro_astra>`; Goal and full Overall will be added after completion.

Model configurations
--------------------


.. list-table:: RPent planner configurations
   :header-rows: 1
   :widths: 25 30 25 20

   * - Backend
     - Model
     - Reasoning
     - Effort
   * - Codex
     - GPT-5.5
     - On
     - ``xhigh``
   * - Claude Code
     - Opus-4.8
     - On
     - ``max``
   * - Codex
     - GPT-6 Astra
     - On
     - ``low``


``Reasoning`` is the model's native reasoning mode; ``Effort`` is its configured
level. ``xhigh`` and ``max`` are provider-specific settings, not equivalent compute
budgets. Experiment contributors confirmed the model identities, backend mappings,
and reasoning settings.

.. _libero-series:

LIBERO series
----------------

Standard LIBERO
~~~~~~~~~~~~~~~

Standard LIBERO uses the original Spatial, Object, Goal, and Long suites
without PRO perturbations. Its Long suite is standard LIBERO-10, evaluated
separately from PRO Long Task/Swap below.


.. list-table:: RPent success rates
   :header-rows: 1
   :widths: 31 23 23 23

   * - Evaluation item
     - GPT-5.5 / ``xhigh``
     - Opus-4.8 / ``max``
     - GPT-6 Astra / ``low``
   * - Spatial
     - Not reported
     - 97.0%
     - Not reported
   * - Object
     - Not reported
     - 100.0%
     - Not reported
   * - Goal
     - Not reported
     - 94.0%
     - Not reported
   * - Long
     - Not reported
     - 93.0%
     - Not reported
   * - Overall
     - Not reported
     - 96.0% (384/400)
     - Not reported


.. _libero-pro-long:

.. _libero-pro-across-task-families:

LIBERO-PRO
~~~~~~~~~~

Task redirects instructions; Swap exchanges object positions. Overall covers
all eight Task/Swap cells across Spatial, Object, Goal, and Long. Long-only results
do not define this aggregate.


.. list-table:: RPent success rates
   :header-rows: 1
   :widths: 31 23 23 23

   * - Evaluation item
     - GPT-5.5 / ``xhigh``
     - Opus-4.8 / ``max``
     - GPT-6 Astra / ``low``
   * - Spatial Task
     - 81.0%
     - 94.0%
     - 100% (100/100)
   * - Spatial Swap
     - 69.0%
     - 80.0%
     - 98% (98/100)
   * - Object Task
     - 94.0%
     - 88.0%
     - 100% (100/100)
   * - Object Swap
     - 91.0%
     - 90.0%
     - 99% (99/100)
   * - Goal Task
     - 75.0%
     - 87.0%
     - Not reported
   * - Goal Swap
     - 66.0%
     - 87.0%
     - Not reported
   * - Long Task
     - 52.0%
     - 71.0%
     - 85% (85/100)
   * - Long Swap
     - 49.0%
     - 62.0%
     - 72% (72/100)
   * - Overall
     - 72.1%
     - 82.4%
     - Not reported



GPT-6 Astra publishes **six completed suites, 600 episodes** (554 successes,
46 failures). Goal Task, Goal Swap and eight-item Overall remain unreported;
uncompleted suites will be added after completion.

LIBERO-PRO Goal: zero-shot
~~~~~~~~~~~~~~~~~~~~~~~~~~

This ablation removes target-setting Task Specific Memory and Global Memory.
Its results are separate from the memory-backed PRO results above.


.. list-table:: RPent success rates
   :header-rows: 1
   :widths: 31 23 23 23

   * - Evaluation item
     - GPT-5.5 / ``xhigh``
     - Opus-4.8 / ``max``
     - GPT-6 Astra / ``low``
   * - Goal Task
     - Not reported
     - 79.0%
     - Not reported
   * - Goal Swap
     - Not reported
     - 31.0%
     - Not reported


RoboCasa365 Target50
--------------------

All three splits and Overall are shown. Overall weights each of the 50 tasks
equally; the splits have different episode counts, so pooling all episodes
would give a different metric.


.. list-table:: RPent success rates
   :header-rows: 1
   :widths: 31 23 23 23

   * - Evaluation item
     - GPT-5.5 / ``xhigh``
     - Opus-4.8 / ``max``
     - GPT-6 Astra / ``low``
   * - Atomic-Seen
     - 92.0%
     - 79.4%
     - Not reported
   * - Composite-Seen
     - 61.0%
     - 47.5%
     - Not reported
   * - Composite-Unseen
     - 13.8%
     - 15.0%
     - Not reported
   * - Overall (task-weighted)
     - 57.1%
     - 48.6%
     - Not reported


RoboTwin C2R
------------

C2R evaluates transfer from the clean setting to the randomized setting.


.. list-table:: RPent success rates
   :header-rows: 1
   :widths: 31 23 23 23

   * - Evaluation item
     - GPT-5.5 / ``xhigh``
     - Opus-4.8 / ``max``
     - GPT-6 Astra / ``low``
   * - C2R
     - 58.0%
     - 58.4%
     - Not reported


Complete paper result tables
----------------------------

The following tables retain every score in Harness VLA v4 Tables 2–6, including
unreported entries and source precision. RPent corresponds to Harness VLA in the
paper; Codex and CC correspond to GPT-5.5 and Opus-4.8. External trial counts
follow their source reports; rounded percentages are not converted to counts.

Standard LIBERO
~~~~~~~~~~~~~~~~

:ref:`Table 2 <benchmark-source-p2>`

.. csv-table::
   :name: paper-table-2
   :header: "Method", "Spatial", "Object", "Goal", "Long", "Overall"
   :class: table-sm

   "OpenVLA", "84.7", "88.4", "79.2", "53.7", "76.5"
   "NORA", "85.6", "89.4", "80.0", "63.0", "79.5"
   "π0", "96.8", "98.8", "95.8", "85.2", "94.2"
   "π_RLinf", "99.0", "96.0", "97.0", "89.0", "95.3"
   "AtomVLA", "96.4", "99.6", "97.6", "94.4", "97.0"
   "RPent / Opus-4.8", "97.0", "100.0", "94.0", "93.0", "96.0"

LIBERO-PRO
~~~~~~~~~~~~~~~~

:ref:`Table 3 <benchmark-source-p3>`

.. csv-table::
   :name: paper-table-3
   :header: "Method", "Spatial Task", "Spatial Swap", "Object Task", "Object Swap", "Goal Task", "Goal Swap", "Long Task", "Long Swap", "Overall"
   :class: table-sm

   "OpenVLA", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"
   "π0", "0.0", "0.0", "0.0", "2.0", "0.0", "0.0", "0.0", "0.0", "0.3"
   "π0.5", "1.0", "20.0", "1.0", "17.0", "2.0", "38.0", "1.0", "8.0", "11.0"
   "MolmoAct", "0.0", "0.0", "0.0", "6.0", "0.0", "0.0", "6.0", "0.0", "1.5"
   "NORA", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"
   "X-VLA", "0.0", "0.0", "8.0", "2.0", "9.0", "1.0", "10.0", "0.0", "3.8"
   "AtomVLA", "1.0", "16.0", "0.0", "10.0", "11.0", "2.0", "9.0", "1.0", "6.3"
   "Cap-X", "14.0", "12.0", "18.0", "22.0", "17.0", "26.0", "``-``", "``-``", "18.2 (6)"
   "RATS", "31.0", "29.0", "63.0", "61.0", "36.0", "43.0", "``-``", "``-``", "43.8 (6)"
   "π_RLinf", "42.0", "59.0", "71.0", "78.0", "45.0", "42.0", "49.0", "14.0", "50.0"
   "RPent / GPT-5.5", "81.0", "69.0", "94.0", "91.0", "75.0", "66.0", "52.0", "49.0", "72.1"
   "RPent / Opus-4.8", "94.0", "80.0", "88.0", "90.0", "87.0", "87.0", "71.0", "62.0", "82.4"

``(6)``: Cap-X and RATS report only the six Spatial/Object/Goal items; their
Overall is not ranked against eight-item aggregates. ``-`` is unreported, not
zero. Astra's six completed suites are a separate contribution, not paper results.

RoboCasa365 Target50
~~~~~~~~~~~~~~~~~~~~

:ref:`Table 4 <benchmark-source-p4>`

.. csv-table::
   :name: paper-table-4
   :header: "Method", "Atomic-Seen", "Composite-Seen", "Composite-Unseen", "Overall (task-weighted)"
   :class: table-sm

   "RLDX-1", "60.0", "21.3", "5.0", "30.0"
   "WorldDreamer", "66.3", "26.7", "9.0", "35.3"
   "π0.5", "39.6", "7.1", "1.2", "16.9"
   "π0", "34.6", "6.1", "1.1", "14.8"
   "RPent / GPT-5.5", "92.0", "61.0", "13.8", "57.1"
   "RPent / Opus-4.8", "79.4", "47.5", "15.0", "48.6"

LIBERO-PRO Goal zero-shot: per task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`Table 5 <benchmark-source-p5>`

.. csv-table:: Task (T)
   :name: paper-table-5-task
   :header: "Method", "Task 0", "Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Task 8", "Task 9", "Average"
   :class: table-sm

   "Cap-X", "0.0", "0.0", "10.0", "38.0", "12.0", "4.0", "34.0", "12.0", "40.0", "18.0", "16.8"
   "RPent / Opus-4.8", "10.0", "100.0", "90.0", "100.0", "20.0", "80.0", "90.0", "100.0", "100.0", "100.0", "79.0"

.. csv-table:: Swap (S)
   :name: paper-table-5-swap
   :header: "Method", "Task 0", "Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Task 8", "Task 9", "Average"
   :class: table-sm

   "Cap-X", "0.0", "4.0", "0.0", "36.0", "22.0", "60.0", "4.0", "2.0", "62.0", "66.0", "25.6"
   "RPent / Opus-4.8", "0.0", "10.0", "0.0", "20.0", "90.0", "0.0", "10.0", "80.0", "100.0", "0.0", "31.0"

RoboTwin C2R
~~~~~~~~~~~~~~~~

:ref:`Table 6 <benchmark-source-p6>`

.. csv-table::
   :name: paper-table-6
   :header: "Method", "Success rate (%)"
   :class: table-sm

   "GR00T-N1.7", "20.7"
   "π0.5", "47.9"
   "StarVLA", "10.6"
   "LingBot-VLA", "50.4"
   "RPent / GPT-5.5", "58.0"
   "RPent / Opus-4.8", "58.4"

Protocols and sources
---------------------

Result snapshot updated on 2026-09-14. RPent's reported GPT-5.5 and Opus-4.8 results
are aligned with Harness VLA paper v4 (2026-09-02); GPT-6 Astra adds a new
model evaluation. The paper labels the backends Codex and CC (Claude Code);
contributors supplied the exact model and reasoning settings.

Task success follows the benchmark predicate: LIBERO ``terminated`` in the
environment trace, RoboCasa ``state.success``, or RoboTwin
``TASK_ENV.eval_success``. A planner ``finish`` call or a primitive's local
success signal is not itself the task label. Exploration episodes used to
construct memory are excluded from evaluation scores.

The LIBERO-family VLA backend is the frozen RLinf π0.5 full-shot LIBERO
checkpoint; RoboCasa uses frozen RLDX-1; RoboTwin uses the frozen post-trained
LingBot-VLA checkpoint. Planner models, VLA backends, memory corpora, and
evaluation protocols describe different parts of each system; score differences
are not a controlled experiment changing only the planner model.

.. _benchmark-source-p2:

**Standard LIBERO.** `Harness VLA, Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_.
Four standard suites, 100 episodes per suite, and 400 episodes for Overall.

.. _benchmark-source-r2:
.. _benchmark-source-p3:

**LIBERO-PRO.** `Harness VLA, Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_.
Eight Task/Swap cells, each with 10 tasks and 10 evaluation seeds per task: 100
episodes per cell and 800 for Overall. Seed 0 is used to construct memory. Long Task
is ``libero_10_task``; Long Swap is ``libero_10_swap``.

.. _benchmark-source-r3:
.. _benchmark-source-p4:

**RoboCasa365 Target50.** `Harness VLA, Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_.
Atomic-Seen has 18 tasks with 10 seeds each; Composite-Seen and Composite-Unseen each
have 16 tasks with five seeds each: 180, 80, and 80 episodes. Seen/Unseen describes
task-template coverage in pretraining. Overall weights the 50 tasks equally.
Percentages retain the source precision; exact success counts are not inferred from
rounded percentages.

.. _benchmark-source-p5:

**LIBERO-PRO Goal zero-shot.** `Harness VLA, Table 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_.
Goal Task and Goal Swap each have 10 tasks with 10 seeds per task, or 100 episodes.
Target-setting Task Specific Memory and Global Memory are removed.

.. _benchmark-source-r4:
.. _benchmark-source-p6:

**RoboTwin C2R.** `Harness VLA, Table 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_.
50 tasks with five official expert-verified randomized seeds each, or 250 episodes.
Task memory comes from a verified ``demo_clean`` instance and transfers to
``demo_randomized`` without exploration in the randomized setting.

.. _benchmark-source-r1:
.. _benchmark-protocol-r1:

**GPT-6 Astra evaluation.** Campaign ``libero_long_gpt6_astra_20260907``
started on 2026-09-07 using runtime revision
`014a0fa <https://github.com/RLinf/RPent/commit/014a0fa97f69c991ee5e0f62f14e1d6f89c3dcd7>`_.
Long Task and Long Swap each contain 10 tasks. Task memory is independently
constructed on seed 0 and frozen before evaluation on seeds 1–10 for each task.
The evaluation uses a local memory corpus, a 5000-second planner time limit, and
a 10000-step environment horizon; the tables report environment success rates.

When adding models or scores, preserve evaluation rows and model-column order,
record the backend, model, reasoning settings, sample size, and protocol, and
update the source notes. Keep unavailable cells marked ``Not reported``.

.. astra-supplementary-source-begin

.. _benchmark-source-astra-pro:

**GPT-6 Astra supplementary LIBERO-PRO evaluation.** Campaign
``libero_pro_remaining_gpt6_astra_20260913`` uses runtime
`014a0fa <https://github.com/RLinf/RPent/commit/014a0fa97f69c991ee5e0f62f14e1d6f89c3dcd7>`_. Spatial, Object, and Goal each cover Task and Swap;
each reported item contains 10 tasks evaluated on seeds 1–10 (100 episodes).
Memory is built independently on seed 0, merged, and frozen before evaluation.

GPT-6 Astra's complete 800-episode Overall is reported only after all eight
Task/Swap items are complete. The full eight-item scope consists of 200 Long
and 600 supplementary episodes. Each reported batch uses its own memory frozen
before evaluation; the batches do not share a single memory snapshot.

From the **2026-09-14 14:00:55 UTC** verified report, this publication includes
only six completed suites, 600 episodes (554 successes, 46 failures).
The :doc:`task/seed result tables <results/libero_pro_astra>` preserve those
600 completed positions. Partial results from unfinished suites are not published.

.. astra-supplementary-source-end


.. _benchmark-demo:

Demo
----

**RPent simulation demo: GPT-6 Astra versus GPT-5.6 xhigh, shown at 4x speed.**
This is a separate demonstration from the GPT-5.5 evaluation records.

.. image:: https://raw.githubusercontent.com/RLinf/misc/e858f627dbcc1b35440c3cb5ecaaefd016eb8680/rpent/demo/demo-poster.jpg
   :alt: RPent simulation demo
   :width: 100%
   :target: https://raw.githubusercontent.com/RLinf/misc/e858f627dbcc1b35440c3cb5ecaaefd016eb8680/rpent/demo/demo.mp4

View or download the `complete MP4 (about 24 seconds, 5.4 MiB) <https://raw.githubusercontent.com/RLinf/misc/e858f627dbcc1b35440c3cb5ecaaefd016eb8680/rpent/demo/demo.mp4>`_.

``Not reported`` means no result is available; it is not a zero score.

Maintaining these results
-------------------------

Figures and demo media are maintained in `RLinf/misc <https://github.com/RLinf/misc>`_
under ``rpent/``. These pages reference immutable media commit
``e858f627dbcc1b35440c3cb5ecaaefd016eb8680``; the accompanying
`result snapshot <https://raw.githubusercontent.com/RLinf/misc/e858f627dbcc1b35440c3cb5ecaaefd016eb8680/rpent/benchmarks/results.json>`_ records their provenance.
The presentation code, ``leaderboard.js`` and ``leaderboard.css``, is also
maintained in misc. RPent embeds only pinned presentation assets. Interactive
charts read the same JSON snapshot. Native result tables remain available
without JavaScript or network access; building the docs needs no plotting tools.

When updating results, edit both language tables, retain evaluation scopes and
source precision, and submit matching figures to misc. Update the media commit
only after checking figure/table agreement. Do not infer success counts from
rounded percentages or use missing results as zero. Preserve the model-column
order, task-weighted RoboCasa aggregation and the distinction between full
LIBERO-PRO and six-item or zero-shot comparisons. Keep maintainer guidance on
this documentation page rather than in an additional code-directory README.

.. toctree::
   :hidden:

   results/libero_pro_astra
