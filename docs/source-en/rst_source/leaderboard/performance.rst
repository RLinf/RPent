:html_theme.sidebar_secondary.remove:

Performance
===========

.. rpent-leaderboard:: performance

.. _libero-pro-empty-memory-evaluation:

LIBERO-PRO Empty-Memory Evaluation
----------------------------------

This October 2026 experiment measures RPent with a frozen Pi0.5 policy and
an empty local experience corpus. Codex / ``gpt-6.1-sol`` / ``low`` plans
with the usual perception and manipulation tools. It records **76 successes
in 226 evaluable episodes (33.63%)** from 240 nominal task/seed pairs. The
leaderboard identifies the configuration as ``RPent / GPT-6.1 Sol / no memory
(stratified sample)`` and retains every suite's denominator.

This measures an agent using Pi0.5 without episodic memory, including planner
decisions, perception, and scripted manipulation tools. It is not direct
Pi0.5 policy inference alone. The policy is
``RLinf-Pi05-LIBERO-130-fullshot-SFT``, with frozen weights and no additional
LIBERO-PRO fine-tuning. The original Pi0.5 reference row and this SFT checkpoint
must not be treated as the same experimental condition.

Configuration and Coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~

Ordinary evaluation was configured for one planner session per task/seed,
with exploration disabled. Historical output directories contain repeated
invocations; their effect is described under Evidence and Limitations.

.. list-table:: Evaluation configuration
   :header-rows: 1

   * - Setting
     - Recorded condition
   * - Environment
     - LIBERO-PRO; Spatial, Object, Goal, Long, each with Task/Swap variants
   * - Planner
     - Codex, ``gpt-6.1-sol``, ``low`` reasoning effort
   * - Action policy
     - Frozen ``RLinf-Pi05-LIBERO-130-fullshot-SFT``
   * - Perception and tools
     - SAM3; normal RPent LIBERO perception, Pi0.5, and scripted action tools
   * - Memory
     - ``--memory-profile local``; an independent corpus with only an empty ``MEMORY.md``; no HF or historical task experience
   * - Static instructions
     - Repository perception, calibration, and tool instructions remain available
   * - Budget
     - 100 planner turns; 1,200-second planner timeout; no ``--explore``
   * - Execution
     - Four concurrent workers on an A800 80 GB; separate output directory per cell
   * - Sampling
     - Task IDs 0–9 in every suite-variant; Spatial seeds 0–5, other suites seeds 0–1

The manifest contains 80 distinct task/variant configurations and 240
nominal episodes, not 240 distinct task definitions. Spatial contributes 120
nominal episodes; the other six suite-variants contribute 20 each. Overall
is the micro-average over evaluable episodes, so Spatial has more weight
than in an eight-suite evaluation with equal sample sizes.

Eight-Suite Results
~~~~~~~~~~~~~~~~~~~

Success follows the recorded environment ``terminated`` /
``libero_terminated`` boolean. The 24 explicit planner timeouts count as
failures. Fourteen rows reported as unavailable because of missing official
PRO initial states are excluded from the denominator. Failure includes
timeouts in the table below.

.. list-table:: Empty-memory results by suite-variant
   :header-rows: 1

   * - Suite-variant
     - Seeds
     - Nominal
     - Evaluable
     - Success
     - Failure
     - Timeout
     - Unavailable
     - Success rate
   * - Spatial Task
     - 0–5
     - 60
     - 48
     - 19
     - 29
     - 4
     - 12
     - 39.58%
   * - Spatial Swap
     - 0–5
     - 60
     - 60
     - 28
     - 32
     - 4
     - 0
     - 46.67%
   * - Object Task
     - 0–1
     - 20
     - 20
     - 7
     - 13
     - 2
     - 0
     - 35.00%
   * - Object Swap
     - 0–1
     - 20
     - 20
     - 6
     - 14
     - 2
     - 0
     - 30.00%
   * - Goal Task
     - 0–1
     - 20
     - 20
     - 7
     - 13
     - 2
     - 0
     - 35.00%
   * - Goal Swap
     - 0–1
     - 20
     - 20
     - 5
     - 15
     - 7
     - 0
     - 25.00%
   * - Long Task
     - 0–1
     - 20
     - 18
     - 2
     - 16
     - 1
     - 2
     - 11.11%
   * - Long Swap
     - 0–1
     - 20
     - 20
     - 2
     - 18
     - 2
     - 0
     - 10.00%
   * - Overall
     - Stratified
     - 240
     - 226
     - 76
     - 150
     - 24
     - 14
     - 33.63%

Long Task/Swap correspond to ``libero_10_task`` / ``libero_10_swap``.
The unavailable rows are Spatial Task IDs 3 and 7 at seeds 0–5, and Long
Task ID 2 at seeds 0–1. They stay in the nominal manifest rather than being
counted as model failures. The non-Spatial batch is 29/118 (24.58%); its 89
failures include 16 timeouts.

Spatial Comparison
~~~~~~~~~~~~~~~~~~

Spatial Task and Spatial Swap have the broadest coverage: six seeds per
task, compared with two in the other suites. Together they give **47/108
(43.52%)**, with 61 failures including eight timeouts and 12 unavailable rows.
They are the most extensively sampled description of this empty-memory
configuration in the experiment.

.. list-table:: Spatial results and published memory-assisted references
   :header-rows: 1

   * - Configuration
     - Spatial Task
     - Spatial Swap
     - Coverage
   * - RPent / GPT-6.1 Sol / low / empty memory
     - 19/48 (39.58%)
     - 28/60 (46.67%)
     - Seeds 0–5; 12 unavailable Spatial Task rows
   * - RPent / GPT-5.5 / xhigh / published memory
     - 81.0%
     - 69.0%
     - Published 100 episodes per suite-variant
   * - RPent / GPT-6 Astra / low / published memory
     - 100/100 (100%)
     - 98/100 (98%)
     - Published 100 episodes per suite-variant

These use the reference results already published above. Planners, memory
sources, sampling, runtime revisions, and inference conditions differ.
Six seeds per task do not establish a paired memory ablation: a memory-on
run with the same GPT-6.1 Sol configuration and task/seed manifest has not
been established. Differences must not be attributed entirely to memory.
The stratified 33.63% Overall must not substitute for an equally sampled
800-episode score.

Evidence and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~

The contributor record is ``final_record_20261008/`` under
``03-memory-tuned-evaluation/``, particularly
``03-LIBERO_PRO_PI05_TEST_PROTOCOL.md``,
``07-PROGRESS_LIBERO_PRO_NO_MEMORY_240.md``, and
``COMPARISON_LIBERO_PRO_NO_MEMORY.md``. The leaderboard's
:download:`results.json <../../../_static/leaderboard/results.json>` stores
aggregate provenance under ``contributor-libero-pro-no-memory-20261008``,
including suite counts, coverage, timeout handling, and the audit manifest
SHA-256. Raw logs, checkpoints, and generated episode outputs are not vendored.

The source review accounts for 201 terminal JSON outcomes, 24 timeout-log
failures, and one transcript-only failure. Long Swap task 8 / seed 0 has no
canonical task JSON. The last parseable object of its concatenated transcript
records failure and ``terminated:false``; the episode remains a failure,
not an omitted row. Of the 14 unavailable rows, eight have retained diagnostic
logs in the reviewed tree; Spatial Task 7 / seeds 0–5 follow the consolidated
unavailability record.

The scheduler reran some completed cells when it did not find a canonical
JSON. Outputs contain repeated invocations and concatenated transcripts.
The archived terminal record, or the last parseable transcript in the
exception above, defines the outcome. A terminal JSON with a valid boolean
takes precedence over an earlier timeout string in the same log. This
selection history prevents interpreting the archive as strict single-run
or controlled memory-on/off evidence. Historical exploration results such
as 24/39 are separate and are not pooled into these 240 rows.

The exact historical source, simulator, and asset revisions have not been
independently established. An empty experience corpus still leaves policy
training and static instructions intact. No latency, token-cost, or
statistical memory-effect claim is inferred from this archive.

Run the Empty-Memory Condition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complete :doc:`../simulators/libero` and planner setup first. The following
shows the recorded command pattern with the planner budget made explicit.
It runs one cell; reproducing coverage requires the matrix above, separate
output directories, and recording actual code, resource, and checkpoint
revisions. It is not a claim of bit-for-bit historical reproduction.

.. code-block:: bash

   NO_MEMORY_DIR="$(mktemp -d)"
   printf '# Empty memory\n' > "$NO_MEMORY_DIR/MEMORY.md"

   rpent --robot libero \
     --suite libero_spatial_task --task 0 --seed 0 \
     --libero-type pro --planner codex \
     --model gpt-6.1-sol --reasoning-effort low \
     --memory-profile local --memory-dir "$NO_MEMORY_DIR" \
     --max-turns 100 --planner-timeout-s 1200 \
     --cuda-device 0 --output-dir ./outputs/no-memory/spatial_task_t0_s0

Keep the corpus empty, omit ``--explore``, and do not substitute the default
HF memory profile. Use the recorded frozen policy and perception checkpoints.
For further runs, fix task/seed coverage, planner, policy, prompts, tool
budgets, and failure accounting. Record new runs separately rather than
silently replacing historical outcomes.
