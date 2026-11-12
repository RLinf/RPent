.. _pyrua-mode:

Use PyRUA Mode
==============

PyRUA Mode lets a planner run Python cells that call RPent's registered robot
tools. A cell can use tool results in conditions and bounded loops; variables,
functions, and imports remain available to later cells in the same session.
Enable it with ``--policy-mode code``.

The process can be summarized as follows:

1. The planner decides **what to do** and submits a Python cell.
2. The Python runtime executes the cell, and the robot toolkit handles each ``robo.<tool_name>(...)`` call.
3. RPent returns the execution output and latest tool result to the planner, which then decides the next step.

First complete the LIBERO-PRO installation and model setup in
:doc:`../get_started/quickstart`. The example below uses the same environment
and model services as an ordinary LIBERO-PRO run.


Run a Task
----------

Add ``--policy-mode code`` to a working task command. The following example
runs task 2 in the LIBERO-PRO Object Swap suite with seed 0:

.. code-block:: bash

   rpent --robot libero --libero-type pro \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8 \
     --policy-mode code

The ``api``, ``claude_code``, and ``codex`` planners support this mode.
``--planner flash`` and ``--explore`` cannot be combined with it. Code mode
does not load task memory or generate a recipe. Omit ``--policy-mode code``
to use the default direct tool-calling mode.

Both CLI and Dashboard task runs support code mode. A dedicated code-mode
prompt is loaded for the selected robot. The first user message sent to the
planner is JSON serialized directly from the run configuration's ``task_desc``;
it is not augmented with ``prompt_vars``, and the tools-mode user prompt is not
rendered. For example, the initial message for the LIBERO-PRO task above
contains only suite, task, and seed.


How Code Execution Works
------------------------

The planner submits a cell through the ``python(code)`` tool. Within that
cell, it calls registered robot tools with keyword arguments, for example
``robo.<tool_name>(argument=value)``. The available methods depend on the
selected robot. ``print(...)`` makes values visible in the returned output;
bare expressions do not. After inspecting the result, the planner may submit
another cell and reuse earlier variables.

Each cell has a default wall-clock limit of 120 seconds, including robot tool
calls, and a budget of 100 robot tool calls. Set ``--code-timeout-s`` and
``--code-max-calls`` to change these limits. A Python exception preserves the
session variables. On timeout or cancellation, in-flight robot tool calls
drain before the Python namespace is reset. Robot actions already executed
are not rolled back, so inspect the scene before continuing.

A successful ``robo.finish(...)`` ends both the cell and the task; the
``api``, ``claude_code``, and ``codex`` planners all receive the completion
result. 


Inspect a Run
-------------

Run records are saved under ``code_policy/`` in the task's output directory;
use ``--output-dir`` to specify that directory. ``cell_*.py`` files contain
code cells, ``cell_*.json`` files contain execution results, and
``calls.jsonl`` records robot tool calls. For LIBERO-PRO, use
``environment_success`` in the run's ``transcript_*.json`` to determine
whether the task succeeded.


LIBERO-PRO Evaluation
---------------------

This evaluation used the Codex planner with GPT-6 Astra at high reasoning
effort. It covered the Spatial, Object, Goal, and Long LIBERO-PRO task
families, each with task and swap suites. Tasks 0–9 were run at seed 0 in
each suite, for 80 episodes in total. The table summarizes results by suite
and reports overall performance; tokens, time, and turns are averages over
episodes with transcripts.

.. list-table:: PyRUA single-seed evaluation on LIBERO-PRO
   :header-rows: 1

   * - Suite
     - Success (of total)
     - Tokens (input/output)
     - Time (seconds)
     - Planner turns
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

Success rates divide successful episodes by all 80 planned episodes. Time
uses ``elapsed_s`` from the run record, measured after the environment runtime
is ready; planner turns use ``stats.turns_used``.
