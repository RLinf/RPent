.. _benchmark-results:

Benchmark Results
=================

Compare planner models, native reasoning settings, and success rates across
RPent evaluations and Harness VLA paper references. Results are grouped by
benchmark and evaluation setting; each source identifies a separate record.
The overview shows the best result within each listed source group, not a
claim of a current benchmark-wide SOTA.

Results checked on 2026-09-09; paper references use arXiv v4 (2026-09-02).

**Reading the tables.** ``Model`` is the planner model. ``Reasoning`` refers
to its native reasoning mode; ``Effort`` is its configured level. ``xhigh``
and ``max`` are provider-specific settings, not equivalent compute budgets.
``Not reported`` means the source has no available result or setting.
``N/A`` means a planner setting does not apply. Source IDs beginning with
**R** identify RPent evaluations; **P** identifies a versioned paper table.

Planner identities and reasoning settings were confirmed by the experiment
contributors: historical Codex records use GPT-5.5 with ``xhigh``, Claude Code
uses Opus-4.8 with ``max``, and Codex also has a GPT-6 Astra record with ``low``.
All three configurations enable native reasoning. The paper names the backends as Codex and CC
(Claude Code); its model and effort labels here are contributor-supplied
metadata.

Overview
--------

These summaries select the highest reported rate for the same benchmark and
setting within each source group. Different memory corpora, runtime revisions,
and evaluation protocols can affect scores; differences are not a controlled
comparison of planner models alone.

.. list-table:: Best recorded RPent evaluations
   :header-rows: 1
   :widths: 20 11 16 11 9 23 10

   * - Benchmark
     - Backend
     - Model
     - Reasoning
     - Effort
     - Success rate
     - Source
   * - PRO Long Task
     - Codex
     - GPT-6 Astra
     - On
     - ``low``
     - 85% (85/100)
     - :ref:`Eval R1 <benchmark-source-r1>`
   * - PRO Long Swap
     - Codex
     - GPT-6 Astra
     - On
     - ``low``
     - 72% (72/100)
     - :ref:`Eval R1 <benchmark-source-r1>`
   * - RoboCasa Target50
     - Codex
     - GPT-5.5
     - On
     - ``xhigh``
     - 57.00% (task-weighted)
     - :ref:`Repo R3 <benchmark-source-r3>`
   * - RoboTwin C2R
     - Codex
     - GPT-5.5
     - On
     - ``xhigh``
     - 58.0% (145/250)
     - :ref:`Repo R4 <benchmark-source-r4>`

.. list-table:: Best recorded paper references
   :header-rows: 1
   :widths: 17 12 11 13 10 8 20 9

   * - Benchmark
     - Method
     - Backend
     - Model
     - Reasoning
     - Effort
     - Success rate
     - Source
   * - Standard LIBERO
     - AtomVLA
     - N/A
     - N/A
     - N/A
     - N/A
     - 97.0%
     - :ref:`v4 P2 <benchmark-source-p2>`
   * - LIBERO-PRO overall
     - Harness VLA
     - Claude Code
     - Opus-4.8
     - On
     - ``max``
     - 82.4%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - PRO Long Task
     - Harness VLA
     - Claude Code
     - Opus-4.8
     - On
     - ``max``
     - 71.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - PRO Long Swap
     - Harness VLA
     - Claude Code
     - Opus-4.8
     - On
     - ``max``
     - 62.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - RoboCasa Target50
     - Harness VLA
     - Codex
     - GPT-5.5
     - On
     - ``xhigh``
     - 57.1% (task-weighted)
     - :ref:`v4 P4 <benchmark-source-p4>`
   * - RoboTwin C2R
     - Harness VLA
     - Claude Code
     - Opus-4.8
     - On
     - ``max``
     - 58.4%
     - :ref:`v4 P6 <benchmark-source-p6>`

Standard LIBERO and the eight-cell LIBERO-PRO aggregate have no separate
RPent evaluation record here. Paper references and RPent evaluations remain
distinct even when their model names or scores match.

LIBERO-PRO Long
---------------

Long Task is ``libero_10_task``; Long Swap is ``libero_10_swap``. Each
reported suite contains 100 evaluation episodes. These are two LIBERO-PRO
subsets, not the standard LIBERO-10 suite or the full LIBERO-PRO aggregate.

.. list-table:: Long Task and Long Swap
   :header-rows: 1
   :widths: 13 18 12 10 19 18 10

   * - Backend
     - Model
     - Reasoning
     - Effort
     - Task
     - Swap
     - Source
   * - Codex
     - GPT-5.5
     - On
     - ``xhigh``
     - 70% (70/100)
     - 55% (55/100)
     - :ref:`Repo R2 <benchmark-source-r2>`
   * - Codex
     - GPT-5.5
     - On
     - ``xhigh``
     - 52.0%
     - 49.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - Codex
     - GPT-6 Astra
     - On
     - ``low``
     - 85% (85/100)
     - 72% (72/100)
     - :ref:`Eval R1 <benchmark-source-r1>`
   * - Claude Code
     - Opus-4.8
     - On
     - ``max``
     - 71.0%
     - 62.0%
     - :ref:`v4 P3 <benchmark-source-p3>`

R2 uses the LIBERO reproduction branch and its published memory. R1 uses a
separately constructed, frozen local memory corpus. Both retain their own
evaluation context; see the source notes.

LIBERO-PRO across task families
-------------------------------

The paper evaluates Spatial, Object, Goal, and Long under Task (instruction
redirection) and Swap (position swaps). Each cell contains 10 tasks with
10 evaluation seeds, or 100 episodes. Seed 0 is reserved for memory
construction. These are the complete paper few-shot results.

.. list-table:: Task perturbations
   :header-rows: 1
   :widths: 12 15 10 8 11 11 11 12 10

   * - Backend
     - Model
     - Reasoning
     - Effort
     - Spatial
     - Object
     - Goal
     - Long
     - Source
   * - Codex
     - GPT-5.5
     - On
     - ``xhigh``
     - 81.0%
     - 94.0%
     - 75.0%
     - 52.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - Claude Code
     - Opus-4.8
     - On
     - ``max``
     - 94.0%
     - 88.0%
     - 87.0%
     - 71.0%
     - :ref:`v4 P3 <benchmark-source-p3>`

.. list-table:: Swap perturbations
   :header-rows: 1
   :widths: 12 15 10 8 11 11 11 12 10

   * - Backend
     - Model
     - Reasoning
     - Effort
     - Spatial
     - Object
     - Goal
     - Long
     - Source
   * - Codex
     - GPT-5.5
     - On
     - ``xhigh``
     - 69.0%
     - 91.0%
     - 66.0%
     - 49.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - Claude Code
     - Opus-4.8
     - On
     - ``max``
     - 80.0%
     - 90.0%
     - 87.0%
     - 62.0%
     - :ref:`v4 P3 <benchmark-source-p3>`

.. list-table:: Overall across all eight PRO cells
   :header-rows: 1
   :widths: 15 20 12 10 15 18 10

   * - Backend
     - Model
     - Reasoning
     - Effort
     - Overall
     - Evaluation episodes
     - Source
   * - Codex
     - GPT-5.5
     - On
     - ``xhigh``
     - 72.1%
     - 800
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - Claude Code
     - Opus-4.8
     - On
     - ``max``
     - 82.4%
     - 800
     - :ref:`v4 P3 <benchmark-source-p3>`

Overall covers all eight cells with equal evaluation sizes. It cannot be
replaced by the mean of Long Task and Long Swap alone. RATS and Cap-X report
only six non-Long cells; their coverage appears in the baseline references.

Standard LIBERO
---------------

Standard LIBERO evaluates the original Spatial, Object, Goal, and Long
(LIBERO-10) suites without PRO perturbations. Each suite has 100 episodes;
the overall record has 400. The paper reports Claude Code results only:
Codex results for all four standard suites and their overall aggregate
are not reported.

.. list-table:: Standard LIBERO paper results
   :header-rows: 1
   :widths: 12 14 10 8 10 10 10 10 16 10

   * - Backend
     - Model
     - Reasoning
     - Effort
     - Spatial
     - Object
     - Goal
     - Long
     - Overall
     - Source
   * - Claude Code
     - Opus-4.8
     - On
     - ``max``
     - 97.0%
     - 100.0%
     - 94.0%
     - 93.0%
     - 96.0% (384/400)
     - :ref:`v4 P2 <benchmark-source-p2>`

RoboCasa365 Target50
--------------------

Target50 has 18 Atomic-Seen tasks with 10 seeds per task, 16 Composite-Seen
tasks with five seeds per task, and 16 Composite-Unseen tasks with five seeds
per task: 180, 80, and 80 episodes. ``Seen`` and ``Unseen`` describe task-template
coverage in pretraining. All 50 tasks remain in the aggregate.

.. list-table:: RoboCasa split results
   :header-rows: 1
   :widths: 12 15 10 8 16 16 17 12 10

   * - Backend
     - Model
     - Reasoning
     - Effort
     - Atomic-Seen
     - Composite-Seen
     - Composite-Unseen
     - Overall
     - Source
   * - Codex
     - GPT-5.5
     - On
     - ``xhigh``
     - 90.56% (163/180)
     - 61.25% (49/80)
     - 15.00% (12/80)
     - 57.00%
     - :ref:`Repo R3 <benchmark-source-r3>`
   * - Codex
     - GPT-5.5
     - On
     - ``xhigh``
     - 92.0%
     - 61.0%
     - 13.8%
     - 57.1%
     - :ref:`v4 P4 <benchmark-source-p4>`
   * - Claude Code
     - Opus-4.8
     - On
     - ``max``
     - 79.4%
     - 47.5%
     - 15.0%
     - 48.6%
     - :ref:`v4 P4 <benchmark-source-p4>`

**Aggregation.** Overall is task-weighted: each task contributes equally,
although Atomic-Seen has twice as many seeds per task. For the RPent record:

.. math::

   \mathrm{Overall} = \frac{18(163/180) + 16(49/80) + 16(12/80)}{50}
   \times 100\% = 57.00\%.

The pooled episode fraction ``224/340`` is not this overall score. Paper
percentages retain their published precision; exact success counts are not
inferred from rounded percentages. R3 uses frozen same-task memory for
43 tasks, with seven tasks evaluated without task memory. See
:doc:`usage/robocasa` for the complete Target50 protocol.

RoboTwin C2R
------------

Clean-to-randomized evaluation covers 50 tasks with five official
expert-verified randomized seeds per task, or 250 episodes. Task memory
comes from a verified ``demo_clean`` instance and transfers to
``demo_randomized`` without exploration in the randomized setting.

.. list-table:: RoboTwin clean-to-randomized results
   :header-rows: 1
   :widths: 16 22 14 12 24 12

   * - Backend
     - Model
     - Reasoning
     - Effort
     - Success rate
     - Source
   * - Codex
     - GPT-5.5
     - On
     - ``xhigh``
     - 58.0% (145/250)
     - :ref:`Repo R4 <benchmark-source-r4>`
   * - Codex
     - GPT-5.5
     - On
     - ``xhigh``
     - 58.0%
     - :ref:`v4 P6 <benchmark-source-p6>`
   * - Claude Code
     - Opus-4.8
     - On
     - ``max``
     - 58.4%
     - :ref:`v4 P6 <benchmark-source-p6>`

R4 is the ``reproduce/robotwin`` branch result. The matching paper Codex
percentage remains its own source record; it is not an additional independent
evaluation count.

LIBERO-PRO Goal: zero-shot
--------------------------

This ablation removes target-setting Task Specific Memory and Global Memory.
Each perturbation has 10 tasks with 10 seeds each. These results are not
pooled with the memory-backed PRO tables. Codex zero-shot Goal results
are not reported.

.. list-table:: Goal zero-shot paper results
   :header-rows: 1
   :widths: 15 15 20 12 10 18 10

   * - Perturbation
     - Backend
     - Model
     - Reasoning
     - Effort
     - Success rate
     - Source
   * - Task
     - Claude Code
     - Opus-4.8
     - On
     - ``max``
     - 79.0%
     - :ref:`v4 P5 <benchmark-source-p5>`
   * - Swap
     - Claude Code
     - Opus-4.8
     - On
     - ``max``
     - 31.0%
     - :ref:`v4 P5 <benchmark-source-p5>`

Baseline references
-------------------

These are reference methods in the linked paper tables, not additional
RPent planner evaluations. ``Model`` and the reasoning columns describe a
separate planner: they are ``N/A`` for direct VLA methods and ``Not reported``
for agent baselines whose planner configuration is not specified in these
tables. A method name does not establish its native reasoning setting.

.. list-table:: Standard LIBERO baselines
   :header-rows: 1
   :widths: 23 13 13 13 10 16 12

   * - Method
     - Backend
     - Model
     - Reasoning
     - Effort
     - Overall
     - Source
   * - OpenVLA
     - N/A
     - N/A
     - N/A
     - N/A
     - 76.5%
     - :ref:`v4 P2 <benchmark-source-p2>`
   * - NORA
     - N/A
     - N/A
     - N/A
     - N/A
     - 79.5%
     - :ref:`v4 P2 <benchmark-source-p2>`
   * - π0
     - N/A
     - N/A
     - N/A
     - N/A
     - 94.2%
     - :ref:`v4 P2 <benchmark-source-p2>`
   * - π_RLinf
     - N/A
     - N/A
     - N/A
     - N/A
     - 95.3%
     - :ref:`v4 P2 <benchmark-source-p2>`
   * - AtomVLA
     - N/A
     - N/A
     - N/A
     - N/A
     - 97.0%
     - :ref:`v4 P2 <benchmark-source-p2>`

All standard-LIBERO rows cover Spatial, Object, Goal, and Long. The paper's
100-episode-per-suite count applies to π_RLinf and Harness VLA; it is not
imposed on external baseline reports. AtomVLA's 97.0% overall exceeds the
Harness VLA standard-LIBERO reference of 96.0%.

.. list-table:: LIBERO-PRO baselines
   :header-rows: 1
   :widths: 17 12 12 12 10 15 12 10

   * - Method
     - Backend
     - Model
     - Reasoning
     - Effort
     - Coverage
     - Overall
     - Source
   * - OpenVLA
     - N/A
     - N/A
     - N/A
     - N/A
     - Eight cells
     - 0.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - π0
     - N/A
     - N/A
     - N/A
     - N/A
     - Eight cells
     - 0.3%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - π0.5
     - N/A
     - N/A
     - N/A
     - N/A
     - Eight cells
     - 11.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - MolmoAct
     - N/A
     - N/A
     - N/A
     - N/A
     - Eight cells
     - 1.5%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - NORA
     - N/A
     - N/A
     - N/A
     - N/A
     - Eight cells
     - 0.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - X-VLA
     - N/A
     - N/A
     - N/A
     - N/A
     - Eight cells
     - 3.8%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - AtomVLA
     - N/A
     - N/A
     - N/A
     - N/A
     - Eight cells
     - 6.3%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - π_RLinf
     - N/A
     - N/A
     - N/A
     - N/A
     - Eight cells
     - 50.0%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - Cap-X
     - Not reported
     - Not reported
     - Not reported
     - Not reported
     - Six non-Long cells
     - 18.2%
     - :ref:`v4 P3 <benchmark-source-p3>`
   * - RATS
     - Not reported
     - Not reported
     - Not reported
     - Not reported
     - Six non-Long cells
     - 43.8%
     - :ref:`v4 P3 <benchmark-source-p3>`

Cap-X and RATS Overall averages Spatial, Object, and Goal under Task and
Swap. Neither reports Long Task or Long Swap in this table. Their six-cell
means must not be ranked as if they covered all eight PRO cells.

.. list-table:: RoboCasa365 baselines
   :header-rows: 1
   :widths: 23 13 13 13 10 16 12

   * - Method
     - Backend
     - Model
     - Reasoning
     - Effort
     - Overall
     - Source
   * - RLDX-1
     - N/A
     - N/A
     - N/A
     - N/A
     - 30.0%
     - :ref:`v4 P4 <benchmark-source-p4>`
   * - WorldDreamer
     - N/A
     - N/A
     - N/A
     - N/A
     - 35.3%
     - :ref:`v4 P4 <benchmark-source-p4>`
   * - π0.5
     - N/A
     - N/A
     - N/A
     - N/A
     - 16.9%
     - :ref:`v4 P4 <benchmark-source-p4>`
   * - π0
     - N/A
     - N/A
     - N/A
     - N/A
     - 14.8%
     - :ref:`v4 P4 <benchmark-source-p4>`

RoboCasa overall figures follow the paper's reported aggregation over
Atomic-Seen, Composite-Seen, and Composite-Unseen. RLDX-1 is the direct
frozen-VLA baseline; the other rows are external paper references.

.. list-table:: RoboTwin C2R baselines
   :header-rows: 1
   :widths: 23 13 13 13 10 16 12

   * - Method
     - Backend
     - Model
     - Reasoning
     - Effort
     - Overall
     - Source
   * - GR00T-N1.7
     - N/A
     - N/A
     - N/A
     - N/A
     - 20.7%
     - :ref:`v4 P6 <benchmark-source-p6>`
   * - π0.5
     - N/A
     - N/A
     - N/A
     - N/A
     - 47.9%
     - :ref:`v4 P6 <benchmark-source-p6>`
   * - StarVLA
     - N/A
     - N/A
     - N/A
     - N/A
     - 10.6%
     - :ref:`v4 P6 <benchmark-source-p6>`
   * - LingBot-VLA
     - N/A
     - N/A
     - N/A
     - N/A
     - 50.4%
     - :ref:`v4 P6 <benchmark-source-p6>`

LingBot-VLA is both the frozen contact-policy backend used by Harness VLA
and a direct evaluation baseline. The other rows are external reports;
Harness VLA's 250-episode count is not assigned to those rows.

.. list-table:: Goal zero-shot agent baseline
   :header-rows: 1
   :widths: 12 12 14 14 14 12 14 8

   * - Method
     - Perturbation
     - Backend
     - Model
     - Reasoning
     - Effort
     - Success rate
     - Source
   * - Cap-X
     - Task
     - Not reported
     - Not reported
     - Not reported
     - Not reported
     - 16.8%
     - :ref:`v4 P5 <benchmark-source-p5>`
   * - Cap-X
     - Swap
     - Not reported
     - Not reported
     - Not reported
     - Not reported
     - 25.6%
     - :ref:`v4 P5 <benchmark-source-p5>`

Protocols and sources
---------------------

Task success follows the benchmark predicate: LIBERO ``terminated`` in the
environment trace, RoboCasa ``state.success``, or RoboTwin
``TASK_ENV.eval_success``. A planner
``finish`` call or a primitive's local success signal is not itself the task
label. Exploration episodes used to construct memory are excluded from
evaluation scores.

The LIBERO-family VLA backend is the frozen RLinf π0.5 full-shot LIBERO
checkpoint; RoboCasa uses frozen RLDX-1; RoboTwin uses the frozen post-trained
LingBot-VLA checkpoint. Planner models, VLA backends, memory sources, and
benchmark settings describe distinct parts of each evaluated system.

.. _benchmark-source-r1:
.. _benchmark-protocol-r1:

**R1 — RPent LIBERO-PRO evaluation.** Campaign
``libero_long_gpt6_astra_20260907``, started 2026-09-07; results checked
2026-09-09. Runtime revision
`014a0fa <https://github.com/RLinf/RPent/commit/014a0fa97f69c991ee5e0f62f14e1d6f89c3dcd7>`_.
Codex with GPT-6 Astra, native reasoning enabled, ``low`` effort. Task memory
is constructed on seed 0 and frozen before evaluation on seeds 1–10 for each
of 10 tasks in each suite. Long Task achieves environment success in 85 of
100 episodes (85%); Long Swap achieves 72 of 100 (72%). This evaluation uses its own
local memory corpus, a 5000-second planner time limit, and a 10000-step
environment horizon.

.. _benchmark-source-r2:

**R2 — RPent LIBERO reproduction.** The
`published result and command <https://github.com/RLinf/RPent/blob/371ac90ece4f95676b7ea5ca05473986f134d5a4/docs/source-en/rst_source/usage/libero.rst#L278-L303>`_
record Long Task 70/100 and Long Swap 55/100 for ``reproduce/libero``.
The contributor-confirmed configuration is Codex, GPT-5.5, native reasoning
enabled, and ``xhigh`` effort. This record uses that branch's published memory
and runtime; its exact historical seed list is not specified in the cited
result.

.. _benchmark-source-r3:

**R3 — RPent RoboCasa Target50 reproduction.** The
`task-level results <https://github.com/RLinf/RPent/blob/371ac90ece4f95676b7ea5ca05473986f134d5a4/robots/robocasa/eval/target50_codex_results.md>`_
and
`Target50 manifest <https://github.com/RLinf/RPent/blob/371ac90ece4f95676b7ea5ca05473986f134d5a4/robots/robocasa/eval/target50.json>`_
record ``robocasa-harness-vla-v1``, its model configuration, fixed resources,
and 340-cell matrix. Published data provide task-level success counts, not
per-seed trajectories. The older 55.40% Harness VLA comparison in that record
refers to the
`v3 paper result <https://arxiv.org/html/2607.08448v3#S3.T4>`_.
This page's paper comparison uses v4's 57.1%, while preserving the RPent
score of 57.00%.

.. _benchmark-source-r4:

**R4 — RPent RoboTwin reproduction.** The
`result and protocol <https://github.com/RLinf/RPent/blob/371ac90ece4f95676b7ea5ca05473986f134d5a4/docs/source-en/rst_source/usage/robotwin.rst#L191-L228>`_
record 145/250 on ``reproduce/robotwin``, with Codex, GPT-5.5, and
``xhigh`` effort. The
`evaluation manifest <https://github.com/RLinf/RPent/blob/371ac90ece4f95676b7ea5ca05473986f134d5a4/robots/robotwin/eval/demo_randomized.json>`_
provides each task's five verified seeds; the seed list differs across tasks.
The reproduction-branch score does not establish full effect parity with
``main``.

.. _benchmark-source-p2:

**P2 — Standard LIBERO.** Harness VLA,
`arXiv v4, Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_
(2026-09-02). Four standard suites; 100 Harness VLA episodes per suite.

.. _benchmark-source-p3:

**P3 — LIBERO-PRO.** Harness VLA,
`arXiv v4, Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_
(2026-09-02). Eight Task/Swap cells; 100 episodes per Harness VLA cell.

.. _benchmark-source-p4:

**P4 — RoboCasa365.** Harness VLA,
`arXiv v4, Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_
(2026-09-02). Atomic-Seen, Composite-Seen, Composite-Unseen, and reported overall.

.. _benchmark-source-p5:

**P5 — LIBERO-PRO Goal zero-shot.** Harness VLA,
`arXiv v4, Table 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_
(2026-09-02). Goal Task and Swap without target-setting memory.

.. _benchmark-source-p6:

**P6 — RoboTwin C2R.** Harness VLA,
`arXiv v4, Table 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_
(2026-09-02). Clean-to-randomized transfer; 250 Harness VLA episodes.

When adding a result, keep the backend, exact planner model, native reasoning
setting, effort, benchmark split, evaluation size, memory protocol, and
versioned source together. Use ``Not reported`` for unavailable cells; add a
separate source record when the protocol or implementation changes.
