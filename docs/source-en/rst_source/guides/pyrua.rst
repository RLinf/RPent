
.. _pyrua-mode:

PyRUA Mode
==========

**PyRUA Mode** allows the planner to execute Python code cells that call
RPent's registered robot tools. A cell can use tool results for conditional
execution and loops. Variables, functions, and imports remain available
across cells within the same session. Enable PyRUA mode with
``--policy-mode code``.

The workflow consists of three steps:

1. The planner decides **what to do** and submits a Python code cell.
2. The Python runtime executes the cell, while the robot toolkit handles
   each ``robo.<tool_name>(...)`` call.
3. RPent returns the execution output and latest tool result to the
   planner, which then determines the next step.

First complete the LIBERO-PRO installation and model setup described in
:doc:`../get_started/quickstart`. The example below uses the same
environment and model services as a standard LIBERO-PRO run.


Run a Task
----------

Add ``--policy-mode code`` to a standard LIBERO-PRO task command to
enable PyRUA mode. The following example runs task 2 in the LIBERO-PRO
Object Swap suite with seed 0, using the Codex planner and GPT-6.1 Sol
at low reasoning effort:

.. code-block:: bash

   rpent --robot libero --libero-type pro \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner codex --model gpt-6.1-sol \
     --policy-mode code --reasoning-effort low

PyRUA mode supports the ``api``, ``claude_code``, and ``codex`` planners
via either the CLI or the Dashboard. It cannot be combined with
``--planner flash`` or ``--explore``. PyRUA mode does not load task
memory or generate an action recipe.


How Code Execution Works
------------------------

In PyRUA mode, RPent loads robot-specific prompts that guide the planner
in calling robot tools through Python. The planner submits code through
``python(code)`` and invokes robot tools using
``robo.<tool_name>(argument=value)``. Use ``print(...)`` to display
values; variables remain available in subsequent cells.

Each cell has a default timeout of 120 seconds, including tool calls,
and allows up to 100 robot tool calls. These limits can be adjusted
using ``--code-timeout-s`` and ``--code-max-calls``, respectively.
When ``robo.finish(...)`` successfully returns a completion marker,
RPent terminates the current code cell and ends the planning session.


Inspect a Run
-------------

Execution logs are saved in the ``code_policy/`` directory under the
task's output directory. Use ``--output-dir`` to specify the output
directory. The ``cell_*.py`` files contain code cells, ``cell_*.json``
files contain execution results, and ``calls.jsonl`` records robot
tool calls. Task success is determined by ``environment_success``
in the run's ``transcript_*.json`` file.


LIBERO-PRO Evaluation
---------------------

This evaluation used the Codex planner with GPT-6 Astra at high reasoning
effort. It covered the Spatial, Object, Goal, and Long LIBERO-PRO task
families, each with task and swap suites. Tasks 0–9 were evaluated with
seed 1 in each suite, for a total of 80 planned episodes and an overall
success rate of 80.00% (64/80).

The table summarizes results by suite. Token usage, elapsed time, and
planner turns are averaged over episodes with transcripts, including
both successful and unsuccessful episodes.

.. list-table:: PyRUA single-seed evaluation on LIBERO-PRO
   :header-rows: 1

   * - Suite
     - Success Rate (%)
     - Tokens (input/output)
     - Time (s)
     - Planner Turns
   * - Object Task
     - 100.00% (10/10)
     - 409,966 / 1,057
     - 152.8
     - 11.6
   * - Object Swap
     - 100.00% (10/10)
     - 396,309 / 1,083
     - 165.3
     - 11.5
   * - Spatial Task
     - 80.00% (8/10)
     - 385,299 / 1,017
     - 147.4
     - 11.3
   * - Spatial Swap
     - 90.00% (9/10)
     - 1,109,798 / 1,764
     - 278.9
     - 17.4
   * - Long Task
     - 50.00% (5/10)
     - 1,655,538 / 2,610
     - 481.6
     - 24.0
   * - Long Swap
     - 60.00% (6/10)
     - 2,150,170 / 3,361
     - 690.4
     - 29.5
   * - Goal Task
     - 80.00% (8/10)
     - 824,506 / 1,580
     - 258.6
     - 16.1
   * - Goal Swap
     - 80.00% (8/10)
     - 962,640 / 1,831
     - 300.7
     - 16.6

Success rates are calculated as the number of successful episodes divided
by the number of planned episodes for each suite. The overall success
rate is calculated across all 80 planned episodes.

Elapsed time uses ``elapsed_s`` from the run record, measured after the
environment runtime is ready. Planner turns use ``stats.turns_used``.
