:html_theme.sidebar_secondary.remove:

.. _benchmark-results:
.. _benchmark-leaderboard:
.. _leaderboard:

RPent Leaderboard
==========================================================================================

.. raw:: html

   <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/RLinf/misc@c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/docs.css">
   <script defer src="https://cdn.jsdelivr.net/gh/RLinf/misc@c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/table-sort.js"></script>
   <script defer src="https://cdn.jsdelivr.net/gh/RLinf/misc@c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/leaderboard.js"></script>
   <script defer src="https://cdn.jsdelivr.net/gh/RLinf/misc@c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/embed.js"></script>
   <div id="rpent-interactive-leaderboard" data-language="en"
        data-results-url="https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/results.json">
     <div class="rpent-static-leaderboard">
       <p>Full result tables are available below.</p>
     </div>
   </div>

**Codex / GPT-6 Astra / low / reasoning: 92.63% Overall (741/800).**
:doc:`All 80 tasks and 800 seed outcomes <results/libero_pro_astra>`.

Paper v4 results, repository reproductions, and contributor evaluations retain separate model settings and provenance. They are not a controlled model-only comparison. Missing results are not zero. Tables are sorted by success rate descending; each evaluation scope remains separate.

``xhigh``, ``max`` and ``low`` are provider-specific effort settings, not equivalent compute budgets. All GPT-6 Astra entries use low effort with reasoning. The GPT-5.6 no-reasoning control remains separate.

.. _libero-series:

LIBERO series
------------------------------------------------------------------------------------------

LIBERO · Standard manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Original, unperturbed LIBERO. External methods retain the precision and evaluation scope of their paper reports.

.. csv-table:: Overall
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "AtomVLA", "97.0%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "96.0%", "384/400", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π_RLinf", "95.3%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π0", "94.2%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "NORA", "79.5%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "OpenVLA", "76.5%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "Not reported", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-6 Astra / low / reasoning", "Not reported", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"

.. csv-table:: Spatial
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "π_RLinf", "99.0%", "Evaluated: 100", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "97.0%", "Evaluated: 100", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π0", "96.8%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "AtomVLA", "96.4%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "NORA", "85.6%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "OpenVLA", "84.7%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "Not reported", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-6 Astra / low / reasoning", "Not reported", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"

.. csv-table:: Object
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Claude Code / Opus-4.8 / max / reasoning", "100.0%", "Evaluated: 100", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "AtomVLA", "99.6%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π0", "98.8%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π_RLinf", "96.0%", "Evaluated: 100", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "NORA", "89.4%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "OpenVLA", "88.4%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "Not reported", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-6 Astra / low / reasoning", "Not reported", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"

.. csv-table:: Goal
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "AtomVLA", "97.6%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π_RLinf", "97.0%", "Evaluated: 100", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π0", "95.8%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "94.0%", "Evaluated: 100", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "NORA", "80.0%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "OpenVLA", "79.2%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "Not reported", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-6 Astra / low / reasoning", "Not reported", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"

.. csv-table:: Long
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "AtomVLA", "94.4%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "93.0%", "Evaluated: 100", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π_RLinf", "89.0%", "Evaluated: 100", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "π0", "85.2%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "NORA", "63.0%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "OpenVLA", "53.7%", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "Not reported", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"
   "Codex / GPT-6 Astra / low / reasoning", "Not reported", "—", "`Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_"

.. _libero-pro-long:
.. _libero-pro-across-task-families:

LIBERO-PRO · Instruction & layout perturbations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Eight-suite Overall uses all Task/Swap items. New model reports use model-specific exploration and evaluation. Cap-X and RATS retain their reported sub-suite scores; missing Long results and eight-suite Overall are not reconstructed.

.. csv-table:: Overall
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "92.63%", "741/800", "`GPT-6 Astra Long evaluation <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; `GPT-6 Astra supplementary evaluation <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; Contributor-confirmed model settings"
   "Claude Code / Opus 4.7", "82.50%", "Evaluated: 800", "New evaluation · 15 Sep 2026"
   "Claude Code / Opus-4.8 / max / reasoning", "82.4%", "Evaluated: 800", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.6 / xhigh / reasoning", "78.50%", "Evaluated: 800", "New evaluation · 15 Sep 2026; Contributor-confirmed model settings"
   "Task card / Molmo", "72.63%", "Evaluated: 800", "New evaluation · 15 Sep 2026"
   "Codex / GPT-5.5 / xhigh / reasoning", "72.1%", "Evaluated: 800", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.6 / no reasoning", "62.50%", "Evaluated: 800", "New evaluation · 15 Sep 2026"
   "π_RLinf", "50.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "11.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "6.3%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "3.8%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "1.5%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.3%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Qwen", "Not reported", "—", "New evaluation · 15 Sep 2026"

.. csv-table:: Spatial Task
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "100%", "100/100", "`GPT-6 Astra supplementary evaluation <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; Contributor-confirmed model settings"
   "Claude Code / Opus-4.8 / max / reasoning", "94.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "81.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "42.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "31.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "14.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "1.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "1.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

.. csv-table:: Spatial Swap
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "98%", "98/100", "`GPT-6 Astra supplementary evaluation <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; Contributor-confirmed model settings"
   "Claude Code / Opus-4.8 / max / reasoning", "80.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "69.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "59.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "29.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "20.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "16.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "12.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

.. csv-table:: Object Task
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "100%", "100/100", "`GPT-6 Astra supplementary evaluation <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; Contributor-confirmed model settings"
   "Codex / GPT-5.5 / xhigh / reasoning", "94.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "88.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "71.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "63.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "18.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "8.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "1.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

.. csv-table:: Object Swap
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "99%", "99/100", "`GPT-6 Astra supplementary evaluation <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; Contributor-confirmed model settings"
   "Codex / GPT-5.5 / xhigh / reasoning", "91.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "90.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "78.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "61.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "22.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "17.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "10.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "6.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "2.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "2.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

.. csv-table:: Goal Task
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "88%", "88/100", "`GPT-6 Astra supplementary evaluation <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; Contributor-confirmed model settings"
   "Claude Code / Opus-4.8 / max / reasoning", "87.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "75.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "45.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "36.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "17.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "11.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "9.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "2.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

.. csv-table:: Goal Swap
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "99%", "99/100", "`GPT-6 Astra supplementary evaluation <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; Contributor-confirmed model settings"
   "Claude Code / Opus-4.8 / max / reasoning", "87.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "66.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "43.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "42.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "38.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "26.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "2.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "1.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

.. csv-table:: Long Task
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "85%", "85/100", "`GPT-6 Astra Long evaluation <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; Contributor-confirmed model settings"
   "Claude Code / Opus-4.8 / max / reasoning", "71.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "52.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "49.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "10.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "9.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "6.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "1.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "Not reported", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "Not reported", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

.. csv-table:: Long Swap
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "72%", "72/100", "`GPT-6 Astra Long evaluation <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_; Contributor-confirmed model settings"
   "Claude Code / Opus-4.8 / max / reasoning", "62.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "49.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π_RLinf", "14.0%", "Evaluated: 100", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0.5", "8.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "AtomVLA", "1.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "MolmoAct", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "NORA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "OpenVLA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "π0", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "X-VLA", "0.0%", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "Cap-X", "Not reported", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"
   "RATS", "Not reported", "—", "`Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_"

LIBERO-PRO · Goal · zero-shot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No target-setting Task Specific Memory or Global Memory. This ablation is separate from the memory-assisted eight-suite evaluation. Per-task results from Table 5 are included below.

.. csv-table:: Goal Task
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Claude Code / Opus-4.8 / max / reasoning", "79.0%", "Evaluated: 100", "`Table 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"
   "Cap-X", "16.8%", "—", "`Table 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "Not reported", "—", "`Table 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"
   "Codex / GPT-6 Astra / low / reasoning", "Not reported", "—", "`Table 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"

.. csv-table:: Goal Swap
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Claude Code / Opus-4.8 / max / reasoning", "31.0%", "Evaluated: 100", "`Table 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"
   "Cap-X", "25.6%", "—", "`Table 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "Not reported", "—", "`Table 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"
   "Codex / GPT-6 Astra / low / reasoning", "Not reported", "—", "`Table 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_"

.. csv-table:: LIBERO-PRO Goal Task · zero-shot, Table 5
   :header: "Method", "Task 0", "Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Task 8", "Task 9", "Mean"
   :class: table-sm

   "Opus-4.8", "10.0%", "100.0%", "90.0%", "100.0%", "20.0%", "80.0%", "90.0%", "100.0%", "100.0%", "100.0%", "79.0%"
   "Cap-X", "0.0%", "0.0%", "10.0%", "38.0%", "12.0%", "4.0%", "34.0%", "12.0%", "40.0%", "18.0%", "16.8%"

.. csv-table:: LIBERO-PRO Goal Swap · zero-shot, Table 5
   :header: "Method", "Task 0", "Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Task 8", "Task 9", "Mean"
   :class: table-sm

   "Opus-4.8", "0.0%", "10.0%", "0.0%", "20.0%", "90.0%", "0.0%", "10.0%", "80.0%", "100.0%", "0.0%", "31.0%"
   "Cap-X", "0.0%", "4.0%", "0.0%", "36.0%", "22.0%", "60.0%", "4.0%", "2.0%", "62.0%", "66.0%", "25.6%"

RoboCasa365 · Target50
------------------------------------------------------------------------------------------

RPent Overall weights 50 tasks equally, not episodes. GPT-6 Astra: all 340 episodes, Overall only. External baseline numbers are quoted from Table 4, not asserted to use identical Target50 coverage. The repository reproduction remains separate below.

.. csv-table:: Overall (source-reported)
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "59.20%", "Evaluated: 340", "New evaluation · 15 Sep 2026; Contributor-confirmed model settings"
   "Codex / GPT-5.5 / xhigh / reasoning", "57.1%", "Evaluated: 340", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "48.6%", "Evaluated: 340", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "WorldDreamer", "35.3%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "RLDX-1", "30.0%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0.5", "16.9%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0", "14.8%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Codex / GPT-5.6 / no reasoning", "Not reported", "—", "New evaluation · 15 Sep 2026"

.. csv-table:: Atomic-Seen
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-5.5 / xhigh / reasoning", "92.0%", "Evaluated: 180", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "79.4%", "Evaluated: 180", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "WorldDreamer", "66.3%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "RLDX-1", "60.0%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0.5", "39.6%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0", "34.6%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Codex / GPT-6 Astra / low / reasoning", "Not reported", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"

.. csv-table:: Composite-Seen
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-5.5 / xhigh / reasoning", "61.0%", "Evaluated: 80", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Claude Code / Opus-4.8 / max / reasoning", "47.5%", "Evaluated: 80", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "WorldDreamer", "26.7%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "RLDX-1", "21.3%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0.5", "7.1%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0", "6.1%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Codex / GPT-6 Astra / low / reasoning", "Not reported", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"

.. csv-table:: Composite-Unseen
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Claude Code / Opus-4.8 / max / reasoning", "15.0%", "Evaluated: 80", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Codex / GPT-5.5 / xhigh / reasoning", "13.8%", "Evaluated: 80", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "WorldDreamer", "9.0%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "RLDX-1", "5.0%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0.5", "1.2%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "π0", "1.1%", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"
   "Codex / GPT-6 Astra / low / reasoning", "Not reported", "—", "`Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_"

RoboTwin · Clean → Randomized
------------------------------------------------------------------------------------------

50 bimanual tasks × 5 randomized seeds for RPent. Clean-setting memory transfers without randomized-setting exploration. External rows retain their source evaluation scope.

.. csv-table:: C2R
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "72.00%", "Evaluated: 250", "Contributor · RoboTwin C2R results; Contributor-confirmed model settings"
   "Codex / GPT-5.5 / xhigh / reasoning · Repository · 15 Sep 2026", "62.4%", "156/250", "`RPent · RoboTwin reproduction, 15 Sep 2026 <https://github.com/RLinf/RPent/blob/849143bf740a562367345cec0de9ef4657dafd73/docs/source-en/rst_source/usage/robotwin.rst>`_"
   "Codex / GPT-5.6 / xhigh / reasoning", "61.20%", "Evaluated: 250", "Contributor · RoboTwin C2R results; Contributor-confirmed model settings"
   "Claude Code / Opus-4.8 / max / reasoning", "58.4%", "Evaluated: 250", "`Table 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_"
   "Codex / GPT-5.5 / xhigh / reasoning · Paper / earlier repository", "58.0%", "145/250", "`Table 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_; `RPent · RoboTwin reproduction <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/robotwin.rst>`_"
   "LingBot-VLA", "50.4%", "—", "`Table 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_"
   "π0.5", "47.9%", "—", "`Table 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_"
   "GR00T-N1.7", "20.7%", "—", "`Table 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_"
   "StarVLA", "10.6%", "—", "`Table 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_"

Task Card · Object · repository evaluation
------------------------------------------------------------------------------------------

Object Task and Swap only. This 200-episode report is distinct from the Task card / Molmo eight-suite 72.63% result.

.. csv-table:: Object · Task + Swap
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / no reasoning", "93.0%", "186/200", "`RPent · Task Card evaluation <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/task_card.rst>`_"
   "Task card", "89.5%", "179/200", "`RPent · Task Card evaluation <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/task_card.rst>`_"

RoboCasa365 · Repository reproduction
------------------------------------------------------------------------------------------

GPT-5.6 xhigh with reasoning: 163/180 Atomic, 49/80 Seen, 12/80 Unseen; task-weighted Overall 57.00%. Earlier GPT-5.5 xhigh with reasoning: 55.40%, separate from paper v4 at 57.1%. GPT-6 Astra low with reasoning: 59.20%, the same evaluation shown in the main chart. The 50 task aggregates below belong only to the 57.00% evaluation.

.. csv-table:: Overall
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-6 Astra / low / reasoning", "59.20%", "Evaluated: 340", "New evaluation · 15 Sep 2026; Contributor-confirmed model settings"
   "Codex / GPT-5.6 / xhigh / reasoning · Repository reproduction", "57.00%", "Evaluated: 340", "`RPent · Target50 reproduction <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/robots/robocasa/eval/target50_codex_results.md>`_; Contributor-confirmed model settings"
   "Codex / GPT-5.5 / xhigh / reasoning · Earlier reference", "55.40%", "Evaluated: 340", "`RPent · earlier reference column <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/robocasa.rst>`_; Contributor-confirmed model settings"

.. csv-table:: Atomic
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-5.5 / xhigh / reasoning · Earlier reference", "91.67%", "165/180", "`RPent · earlier reference column <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/robocasa.rst>`_; Contributor-confirmed model settings"
   "Codex / GPT-5.6 / xhigh / reasoning · Repository reproduction", "90.56%", "163/180", "`RPent · Target50 reproduction <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/robots/robocasa/eval/target50_codex_results.md>`_; Contributor-confirmed model settings"

.. csv-table:: Composite-Seen
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-5.6 / xhigh / reasoning · Repository reproduction", "61.25%", "49/80", "`RPent · Target50 reproduction <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/robots/robocasa/eval/target50_codex_results.md>`_; Contributor-confirmed model settings"
   "Codex / GPT-5.5 / xhigh / reasoning · Earlier reference", "56.25%", "45/80", "`RPent · earlier reference column <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/robocasa.rst>`_; Contributor-confirmed model settings"

.. csv-table:: Composite-Unseen
   :header: "Method / model / setting", "Success rate", "Successful / evaluated", "Source"
   :class: table-sm

   "Codex / GPT-5.6 / xhigh / reasoning · Repository reproduction", "15.00%", "12/80", "`RPent · Target50 reproduction <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/robots/robocasa/eval/target50_codex_results.md>`_; Contributor-confirmed model settings"
   "Codex / GPT-5.5 / xhigh / reasoning · Earlier reference", "13.75%", "11/80", "`RPent · earlier reference column <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/robocasa.rst>`_; Contributor-confirmed model settings"

Paper analyses
------------------------------------------------------------------------------------------

Invocation counts, completion attribution and primitive shares describe mechanisms; they are not success-rate rankings. Published figures are retained without estimating curve point values. Zhang et al., Harness VLA v4, CC BY 4.0.

.. container:: rpent-paper-figures

   .. figure:: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/invocations-libero.png
      :alt: LIBERO-PRO cumulative success by VLA invocation count
      :width: 520px
      :class: rpent-paper-figure
      :target: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/invocations-libero.png

      LIBERO-PRO

   .. figure:: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/invocations-robocasa.png
      :alt: RoboCasa365 cumulative success by VLA invocation count
      :width: 520px
      :class: rpent-paper-figure
      :target: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/invocations-robocasa.png

      RoboCasa365

   .. figure:: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/invocations-robotwin.png
      :alt: RoboTwin C2R cumulative success by VLA invocation count
      :width: 520px
      :class: rpent-paper-figure
      :target: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/invocations-robotwin.png

      RoboTwin C2R

.. image:: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/completion-attribution.png
   :alt: Harness VLA completion-attribution.png
   :width: 520px
   :align: center
   :class: rpent-paper-figure
   :target: https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/assets/completion-attribution.png

.. csv-table:: Primitive usage · Table 18
   :header: "Primitive", "LIBERO", "RoboTwin C2R", "RoboCasa365"
   :class: table-sm

   "MOVE_TO", "6263 (61.8%)", "685 (40.9%)", "3004 (38.7%)"
   "VLA_ACT", "1598 (15.8%)", "794 (47.4%)", "2746 (35.3%)"
   "SET_GRIPPER", "1137 (11.2%)", "71 (4.2%)", "371 (4.8%)"
   "RELEASE", "831 (8.2%)", "124 (7.4%)", "76 (1.0%)"
   "MOVE_POSE", "203 (2.0%)", "—", "—"
   "ROTATE_PITCH", "58 (0.6%)", "—", "66 (0.8%)"
   "ROTATE_WRIST", "44 (0.4%)", "1 (0.1%)", "—"
   "MOVE_BASE", "—", "—", "808 (10.4%)"
   "NAVIGATE_TO", "—", "—", "701 (9.0%)"
   "Total", "10134 (100.0%)", "1675 (100.0%)", "7772 (100.0%)"

.. csv-table:: Primitive classes · Table 19
   :header: "Class", "LIBERO", "RoboTwin C2R", "RoboCasa365"
   :class: table-sm

   "Analytic", "8536 (84.2%)", "881 (52.6%)", "5026 (64.7%)"
   "VLA", "1598 (15.8%)", "794 (47.4%)", "2746 (35.3%)"

Protocols and sources
------------------------------------------------------------------------------------------

Task success follows the environment: LIBERO ``terminated``, RoboCasa ``state.success``, or RoboTwin ``TASK_ENV.eval_success``. A planner finish call is not a task label. Exploration episodes are excluded from evaluation. The frozen VLA backends are RLinf π0.5 (LIBERO), RLDX-1 (RoboCasa), and post-trained LingBot-VLA (RoboTwin).

.. _benchmark-source-p2:
.. _benchmark-source-r2:
.. _benchmark-source-p3:
.. _benchmark-source-r3:
.. _benchmark-source-p4:
.. _benchmark-source-p5:
.. _benchmark-source-r4:
.. _benchmark-source-p6:
.. _benchmark-source-r1:
.. _benchmark-protocol-r1:
.. _benchmark-source-astra-pro:

* `Table 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_
* `Table 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_
* `Table 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_
* `Table 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_
* `Table 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_
* `GPT-6 Astra Long evaluation <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_
* `GPT-6 Astra supplementary evaluation <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_
* New evaluation · 15 Sep 2026 — Contributor-provided result image; scopes explicitly confirmed by the contributor. LIBERO-PRO: full eight-suite exploration and evaluation per model. RoboCasa: all 340 episodes, 50-task-weighted Overall. Individual outcomes and unreported model settings are not supplied.
* `RPent · Task Card evaluation <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/task_card.rst>`_ — Object Task and Swap, 20 tasks × 10 seeds. Aggregate counts are explicit; model versions are not specified for this evaluation. Distinct from the new eight-suite Molmo report.
* `RPent · RoboTwin reproduction <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/robotwin.rst>`_ — The repository also reports Codex / GPT-5.5 xhigh at 58.0%, with explicit counts 145/250. Both published sources are attached to this result; no per-cell identity is asserted.
* `RPent · Target50 reproduction <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/robots/robocasa/eval/target50_codex_results.md>`_ — Published 50-task aggregate record, not a per-cell trace. Kept separate from the paper v4 experiment.
* `RPent · earlier reference column <https://github.com/RLinf/RPent/blob/43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92/docs/source-en/rst_source/usage/robocasa.rst>`_ — Earlier reference counts retained in the repository documentation, distinguished from paper v4 and the repository reproduction.
* `Paper · Figure 4 <https://arxiv.org/html/2607.08448v4#S3.F4>`_
* `Paper · Figure 6 <https://arxiv.org/html/2607.08448v4#S3.F6>`_
* `Paper · Table 18 <https://arxiv.org/html/2607.08448v4#A6.T18>`_
* `Paper · Table 19 <https://arxiv.org/html/2607.08448v4#A6.T19>`_
* Contributor-confirmed model settings — Configuration evidence only: GPT-5.6 reasoning uses xhigh; GPT-6 Astra uses low with reasoning. The 62.50% no-reasoning control is unchanged. RoboCasa reproduction model labels were confirmed by the contributor. Result evidence is cited separately.
* Contributor · RoboTwin C2R results — RoboTwin C2R: GPT-5.6 xhigh 61.20% and GPT-6 Astra low 72.00%, both with reasoning, 250 episodes each. Exact success counts and task outcomes were not supplied.
* `RPent · RoboTwin reproduction, 15 Sep 2026 <https://github.com/RLinf/RPent/blob/849143bf740a562367345cec0de9ef4657dafd73/docs/source-en/rst_source/usage/robotwin.rst>`_ — Main #179 reports GPT-5.5 xhigh: 156 successes, 58 task failures and 36 episode timeouts over 250 episodes (62.4%). Listed task/seed pairs bind the evaluation table task language. This batch is distinct from the paper and earlier 58.0% result.

* Original LIBERO; four suites, 100 episodes each; frozen RLinf π0.5 policy.
* LIBERO-PRO with target-setting memory; eight Task/Swap items, 100 episodes each.
* Long evaluation; seed-0 memory frozen before seeds 1–10. Runtime 014a0fa. This batch is distinct from the Spatial/Object/Goal evaluation.
* Goal Task/Swap without target-setting Task Specific Memory and Global Memory; 100 episodes each.
* Target50: 18 atomic tasks × 10 seeds, 16 seen and 16 unseen composite tasks × 5 seeds; Overall weights 50 tasks equally.
* 50 tasks × 5 expert-verified randomized seeds; memory from demo_clean transfers to demo_randomized.
* Four standard suites; sample counts follow each external report.
* Eight LIBERO-PRO Task/Swap items; sample counts follow each external report.
* Six Spatial/Object/Goal Task/Swap items only; excludes Long.
* Goal ablation without target-setting memory; sample counts follow the external report.
* Table 4 reported baseline values. RLDX-1 is the direct frozen-policy baseline under the paper protocol; other baseline rows come from prior papers. Identical task coverage and aggregation are not established for those prior reports.
* Clean-to-randomized transfer; sample counts follow each external report.
* Spatial/Object/Goal Task and Swap; each item has 10 tasks × seeds 1–10. Seed-0 task memories are merged and frozen before evaluation.
* 800 episodes: 200 Long and 600 Spatial/Object/Goal episodes. Each batch uses its own pre-frozen memory corpus; all eight items have 100 episodes.
* Eight LIBERO-PRO suites. Each model completed its own exploration and full evaluation. Counts and sub-suite scores are not inferred from the reported Overall.
* Full Target50 evaluation: 340 episodes; Overall averages 50 tasks equally. GPT-6 Astra uses low effort with reasoning enabled. Split scores and success counts are not supplied.
* Object Task + Object Swap; 200 episodes, 20 tasks. Task Card 179/200; Codex without reasoning 186/200. This is not the eight-suite Overall.
* Target50 repository reproduction: 18 Atomic tasks × 10 seeds; 16 Seen and 16 Unseen tasks × 5 seeds. Overall is task-weighted.
* Earlier reference column from the repository, not the v4 paper experiment. Split counts and task-weighted Overall are retained verbatim.
* RoboTwin C2R: 50 tasks × 5 seeds, 250 episodes per configuration. GPT-5.6 xhigh with reasoning and GPT-6 Astra low with reasoning. No success counts are inferred from percentages.
* 50 tasks, five verified expert seeds per task, 250 episodes. Evaluation-table task language is bound after scene reset; only final TASK_ENV.eval_success determines success. 4800-second planner limit and 10000 environment steps. Source: RPent #179.

The 2026-09-15 verified Astra snapshot contains all eight suites: 741 successes, 59 failures, 800 episodes. Long uses one frozen memory batch and Spatial/Object/Goal another; no common memory snapshot is implied. The earlier 600-episode snapshot remains archived, not a second ranked experiment.

.. _benchmark-demo:

Demo
------------------------------------------------------------------------------------------

**RPent simulation demo: GPT-6 Astra low with reasoning versus GPT-5.6 xhigh, at 4x speed.** A separate demonstration, not the GPT-5.5 evaluation.

.. raw:: html

   <video class="rpent-demo-video" controls playsinline preload="metadata" poster="https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/demo/demo-poster.jpg">
     <source src="https://cdn.jsdelivr.net/gh/RLinf/misc@c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/demo/demo.mp4" type="video/mp4">
     <a href="https://cdn.jsdelivr.net/gh/RLinf/misc@c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/demo/demo.mp4">Download the demo video</a>
   </video>

View or download the `complete MP4 (about 24 seconds, 5.4 MiB) <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/demo/demo.mp4>`_.

Maintaining these results
------------------------------------------------------------------------------------------

Presentation code, figures, data and demo media live in `RLinf/misc <https://github.com/RLinf/misc>`_, pinned to commit ``c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1``. Native tables remain readable without JavaScript or remote resources. The interactive mount uses the same JSON records and an isolated stylesheet; no plotting code is required to build RPent documentation.

Update both language tables from the verified data, preserve model and protocol boundaries, retain missing values, and verify figure/table agreement before changing the immutable resource revision. Do not infer unreported counts from rounded rates.

.. toctree::
   :hidden:

   results/libero_pro_astra
