.. _pyrua-mode:

Use PyRUA Mode
==============

PyRUA Mode lets a planner run Python cells that call RPent's registered robot
tools. A cell can use tool results in conditions and bounded loops; variables,
functions, and imports remain available to later cells in the same session.
Enable it with ``--policy-mode code``.

The planner chooses **what to do**, the Python worker executes the cell, and
the robot toolkit handles each ``robo.<tool_name>(...)`` call. RPent returns
the cell's output and latest tool result before the planner's next step.

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
session variables. Timeout or cancellation resets the Python namespace;
robot actions already executed are not rolled back, so inspect the scene
before continuing.

A successful ``robo.finish(...)`` ends both the cell and the task. Operator
requests such as ``request_scene_reset`` and ``request_operator_verdict`` are
separate planner tools because they may wait longer than a cell's time limit.


Inspect a Run
-------------

The run output contains ``code_policy/cell_000001.py`` for the first submitted
cell and ``code_policy/cell_000001.json`` for its result. The cell JSON records
output, any exception, elapsed time, and the last tool result.
``code_policy/calls.jsonl`` records each robot tool call. The run's
``transcript_*.json`` also records ``policy_mode`` and
``environment_success``. For LIBERO-PRO, use the environment success signal
to judge the episode; a successful tool call or ``finish`` alone does not
establish task success.


LIBERO-PRO Evaluation
---------------------

In a single-seed evaluation using Codex with GPT-6 Astra at high reasoning
effort, PyRUA Mode succeeded in 64 of 80 episodes (80.00%). The evaluation
used tasks 0–9 at seed 1 in each of the eight LIBERO-PRO suites:

.. list-table::
   :header-rows: 1

   * - Suite
     - Successful episodes
     - Success rate
   * - Spatial Task
     - 8/10
     - 80.00%
   * - Spatial Swap
     - 9/10
     - 90.00%
   * - Object Task
     - 10/10
     - 100.00%
   * - Object Swap
     - 10/10
     - 100.00%
   * - Goal Task
     - 8/10
     - 80.00%
   * - Goal Swap
     - 8/10
     - 80.00%
   * - Long Task
     - 5/10
     - 50.00%
   * - Long Swap
     - 6/10
     - 60.00%
   * - Overall
     - 64/80
     - 80.00%

These results cover one seed per task. See :doc:`Performance
<../leaderboard/performance>` for other published configurations and their
evaluation coverage.
