:html_theme.sidebar_secondary.remove:

.. _benchmark-results:

Benchmark Results
=================

Compare RPent success rates with selected reference methods across simulation benchmarks.

.. _benchmark-leaderboard:

Leaderboard
-----------

Select a benchmark to compare models in a sortable leaderboard. Click a score
heading to rank by that evaluation item, or switch to Charts for a visual
comparison. Overview retains the six benchmark panels. Purple denotes RPent;
gray denotes external reference methods.

Filter model configurations, inspect a score or bar with the mouse or keyboard,
and download the displayed records as CSV. Missing scores remain unranked;
Overall uses the complete reported evaluation, never a partial average.

.. raw:: html

   <div id="rpent-interactive-leaderboard" data-language="en"
        data-results-url="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/benchmarks/results.json" data-asset-base="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/benchmarks/">
   <noscript><p>Interactive filters require JavaScript. The static charts and full result tables remain available.</p></noscript>
   <div class="rpent-leaderboard-legend" aria-label="Chart legend">
     <span><i class="rpent-legend-swatch rpent-legend-rpent" aria-hidden="true"></i>RPent configurations</span>
     <span><i class="rpent-legend-swatch rpent-legend-reference" aria-hidden="true"></i>Selected external baselines</span>
   </div>
   <div class="rpent-leaderboard-grid rpent-static-leaderboard">
     <figure class="rpent-leaderboard-panel" data-benchmark="standard-libero">
       <img class="rpent-chart-light" src="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/benchmarks/standard-libero-en-light.png"
            alt="Standard LIBERO Overall success-rate comparison; full values appear in the result tables below." />
       <img class="rpent-chart-dark" src="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/benchmarks/standard-libero-en-dark.png"
            alt="Standard LIBERO Overall success-rate comparison; full values appear in the result tables below." />
       <figcaption><a href="#benchmark-source-p2">Standard LIBERO Overall: results and evaluation details</a></figcaption>
     </figure>
     <figure class="rpent-leaderboard-panel" data-benchmark="libero-pro">
       <img class="rpent-chart-light" src="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/benchmarks/libero-pro-en-light.png"
            alt="LIBERO-PRO Overall, all eight Task/Swap items: success-rate comparison; full values appear in the result tables below." />
       <img class="rpent-chart-dark" src="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/benchmarks/libero-pro-en-dark.png"
            alt="LIBERO-PRO Overall, all eight Task/Swap items: success-rate comparison; full values appear in the result tables below." />
       <figcaption><a href="#benchmark-source-p3">LIBERO-PRO Overall: results and evaluation details</a></figcaption>
     </figure>
     <figure class="rpent-leaderboard-panel" data-benchmark="robocasa">
       <img class="rpent-chart-light" src="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/benchmarks/robocasa-en-light.png"
            alt="RoboCasa365 Target50 task-weighted Overall success-rate comparison; full values appear in the result tables below." />
       <img class="rpent-chart-dark" src="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/benchmarks/robocasa-en-dark.png"
            alt="RoboCasa365 Target50 task-weighted Overall success-rate comparison; full values appear in the result tables below." />
       <figcaption><a href="#benchmark-source-p4">RoboCasa365 Target50 Overall: results and evaluation details</a></figcaption>
     </figure>
     <figure class="rpent-leaderboard-panel" data-benchmark="robotwin">
       <img class="rpent-chart-light" src="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/benchmarks/robotwin-en-light.png"
            alt="RoboTwin C2R success-rate comparison; full values appear in the result tables below." />
       <img class="rpent-chart-dark" src="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/benchmarks/robotwin-en-dark.png"
            alt="RoboTwin C2R success-rate comparison; full values appear in the result tables below." />
       <figcaption><a href="#benchmark-source-p6">RoboTwin C2R: results and evaluation details</a></figcaption>
     </figure>
     <figure class="rpent-leaderboard-panel" data-benchmark="long-task">
       <img class="rpent-chart-light" src="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/benchmarks/long-task-en-light.png"
            alt="LIBERO-PRO Long Task, RPent model success-rate comparison; full values appear in the result tables below." />
       <img class="rpent-chart-dark" src="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/benchmarks/long-task-en-dark.png"
            alt="LIBERO-PRO Long Task, RPent model success-rate comparison; full values appear in the result tables below." />
       <figcaption><a href="#libero-pro-across-task-families">LIBERO-PRO Long Task: results and evaluation details</a></figcaption>
     </figure>
     <figure class="rpent-leaderboard-panel" data-benchmark="long-swap">
       <img class="rpent-chart-light" src="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/benchmarks/long-swap-en-light.png"
            alt="LIBERO-PRO Long Swap, RPent model success-rate comparison; full values appear in the result tables below." />
       <img class="rpent-chart-dark" src="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/benchmarks/long-swap-en-dark.png"
            alt="LIBERO-PRO Long Swap, RPent model success-rate comparison; full values appear in the result tables below." />
       <figcaption><a href="#libero-pro-across-task-families">LIBERO-PRO Long Swap: results and evaluation details</a></figcaption>
     </figure>
   </div>
   </div>

Rankings apply to the coverage shown in each panel. Every vertical axis starts
at zero; its upper limit is the highest reported score for that setting rounded up
to a multiple of 10%, and remains fixed when configurations are hidden.
Compare the labeled percentages across panels. Full results,
model configurations, and protocols follow below.
Unreported configurations have no bar. Long Task and Long Swap compare RPent
planners and do not replace the full PRO Overall.

.. _benchmark-demo:

Demo
----

**RPent simulation demo: GPT-6 Astra versus GPT-5.6 xhigh, shown at 4× speed.**
This is a separate demonstration from the GPT-5.5 evaluation records above.

.. raw:: html

   <video class="rpent-benchmark-video" controls playsinline preload="metadata"
          poster="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/demo/demo-poster.jpg" width="2048" height="1024"
          aria-label="RPent simulation demo: GPT-6 Astra versus GPT-5.6 xhigh, 4x speed">
     <source src="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/demo/demo.mp4" type="video/mp4" />
     <a href="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/demo/demo.mp4">Download the complete demo video</a>
   </video>
   <p class="rpent-demo-download"><a href="https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/demo/demo.mp4" download>Download complete MP4 (about 24 seconds, 5.4 MiB)</a></p>

``Not reported`` means no result is available for that model and evaluation item;
it is not a zero score.

.. _overview:

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


Baseline references
-------------------

These reference methods retain their reported evaluation coverage. They are
not RPent planner configurations and are not pooled with the main results.
Direct VLA methods have no separate planner Reasoning/Effort settings; the cited
sources do not report the planner models or reasoning settings for Cap-X and RATS.

.. dropdown:: Standard LIBERO


   .. list-table:: Reference methods
      :header-rows: 1
      :widths: 24 36 20 20

      * - Method
        - Coverage
        - Success rate
        - Source
      * - OpenVLA
        - Four standard suites
        - 76.5%
        - :ref:`Table 2 <benchmark-source-p2>`
      * - NORA
        - Four standard suites
        - 79.5%
        - :ref:`Table 2 <benchmark-source-p2>`
      * - π0
        - Four standard suites
        - 94.2%
        - :ref:`Table 2 <benchmark-source-p2>`
      * - π_RLinf
        - Four standard suites
        - 95.3%
        - :ref:`Table 2 <benchmark-source-p2>`
      * - AtomVLA
        - Four standard suites
        - 97.0%
        - :ref:`Table 2 <benchmark-source-p2>`


   External baseline sample counts follow their own reports; RPent's 100 episodes per
   suite are not imposed on those records.

.. dropdown:: LIBERO-PRO


   .. list-table:: Reference methods
      :header-rows: 1
      :widths: 24 36 20 20

      * - Method
        - Coverage
        - Success rate
        - Source
      * - OpenVLA
        - Eight Task/Swap cells
        - 0.0%
        - :ref:`Table 3 <benchmark-source-p3>`
      * - π0
        - Eight Task/Swap cells
        - 0.3%
        - :ref:`Table 3 <benchmark-source-p3>`
      * - π0.5
        - Eight Task/Swap cells
        - 11.0%
        - :ref:`Table 3 <benchmark-source-p3>`
      * - MolmoAct
        - Eight Task/Swap cells
        - 1.5%
        - :ref:`Table 3 <benchmark-source-p3>`
      * - NORA
        - Eight Task/Swap cells
        - 0.0%
        - :ref:`Table 3 <benchmark-source-p3>`
      * - X-VLA
        - Eight Task/Swap cells
        - 3.8%
        - :ref:`Table 3 <benchmark-source-p3>`
      * - AtomVLA
        - Eight Task/Swap cells
        - 6.3%
        - :ref:`Table 3 <benchmark-source-p3>`
      * - π_RLinf
        - Eight Task/Swap cells
        - 50.0%
        - :ref:`Table 3 <benchmark-source-p3>`
      * - Cap-X
        - Six non-Long cells
        - 18.2%
        - :ref:`Table 3 <benchmark-source-p3>`
      * - RATS
        - Six non-Long cells
        - 43.8%
        - :ref:`Table 3 <benchmark-source-p3>`


   Cap-X and RATS cover only the six Spatial/Object/Goal Task/Swap cells. Their
   overall rates are not ranked against eight-cell aggregates.

.. dropdown:: RoboCasa365 Target50


   .. list-table:: Reference methods
      :header-rows: 1
      :widths: 24 36 20 20

      * - Method
        - Coverage
        - Success rate
        - Source
      * - RLDX-1
        - All three splits, task-weighted
        - 30.0%
        - :ref:`Table 4 <benchmark-source-p4>`
      * - WorldDreamer
        - All three splits, task-weighted
        - 35.3%
        - :ref:`Table 4 <benchmark-source-p4>`
      * - π0.5
        - All three splits, task-weighted
        - 16.9%
        - :ref:`Table 4 <benchmark-source-p4>`
      * - π0
        - All three splits, task-weighted
        - 14.8%
        - :ref:`Table 4 <benchmark-source-p4>`


   RLDX-1 is the direct frozen-VLA baseline; the other methods are external reports.
   Overall retains the aggregation reported by the source.

.. dropdown:: RoboTwin C2R


   .. list-table:: Reference methods
      :header-rows: 1
      :widths: 24 36 20 20

      * - Method
        - Coverage
        - Success rate
        - Source
      * - GR00T-N1.7
        - C2R
        - 20.7%
        - :ref:`Table 6 <benchmark-source-p6>`
      * - π0.5
        - C2R
        - 47.9%
        - :ref:`Table 6 <benchmark-source-p6>`
      * - StarVLA
        - C2R
        - 10.6%
        - :ref:`Table 6 <benchmark-source-p6>`
      * - LingBot-VLA
        - C2R
        - 50.4%
        - :ref:`Table 6 <benchmark-source-p6>`


   LingBot-VLA is both RPent's frozen contact-policy backend and a direct evaluation
   baseline. RPent's 250 episodes are not assigned to external reports.

.. dropdown:: LIBERO-PRO Goal zero-shot


   .. list-table:: Reference methods
      :header-rows: 1
      :widths: 24 36 20 20

      * - Method
        - Coverage
        - Success rate
        - Source
      * - Cap-X
        - Goal Task
        - 16.8%
        - :ref:`Table 5 <benchmark-source-p5>`
      * - Cap-X
        - Goal Swap
        - 25.6%
        - :ref:`Table 5 <benchmark-source-p5>`


   These Goal Task/Swap values belong to the ablation without target-setting memory.

Protocols and sources
---------------------

Results checked on 2026-09-13. RPent's reported GPT-5.5 and Opus-4.8 results
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

.. astra-supplementary-source-end


Maintaining these results
-------------------------

Figures and demo media are maintained in `RLinf/misc <https://github.com/RLinf/misc>`_
under ``rpent/``. These pages reference immutable media commit
``3f4c2003590076982782b73bb4e191bdd6a52f39``; the accompanying
`result snapshot <https://raw.githubusercontent.com/RLinf/misc/3f4c2003590076982782b73bb4e191bdd6a52f39/rpent/benchmarks/results.json>`_ records their provenance.
The optional interactive view reads the same pinned result snapshot and SVGs.
The static overview and full tables remain available if JavaScript is unavailable.

When updating results, edit the result snapshot and both language tables, retain
evaluation scopes and source precision, and submit matching figures to misc. Update the media commit
only after checking figure/table agreement. Do not infer success counts from
rounded percentages or use missing results as zero. Preserve the model-column
order, task-weighted RoboCasa aggregation and the distinction between full
LIBERO-PRO and six-item or zero-shot comparisons. Keep maintainer guidance on
this documentation page rather than in an additional code-directory README.
