
.. _pyrua-mode:

PyRUA Mode
==========

**PyRUA Mode** allows the planner to execute Python code cells that call
RPent's registered robot tools. A cell can use tool results for conditional
execution and bounded loops. Variables, functions, and imports remain available
across cells within the same session. Enable PyRUA mode with
``--policy-mode code``.

The workflow consists of three steps:

1. The planner decides **what to do** and submits a Python code cell.
2. The Python runtime executes the cell, while the robot toolkit handles
   each ``robo.<tool_name>(...)`` call.
3. RPent returns the execution output and latest tool result to the
   planner, which then determines the next step.

Run a Task
----------

Use a Linux host with user namespaces enabled and Bubblewrap (``bwrap``)
installed. On Ubuntu, install it with ``sudo apt install bubblewrap``.
For the example below, complete the LIBERO-PRO and model setup in
:doc:`../get_started/quickstart` and configure access to the Codex model.

Add ``--policy-mode code`` to a standard LIBERO-PRO task command to
enable PyRUA mode. The following example runs task 2 in the LIBERO-PRO
Object Swap suite with seed 0, using the Codex planner and GPT-6.1 Sol
at low reasoning effort:

.. code-block:: bash

   rpent --robot libero --libero-type pro \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner codex --model gpt-6.1-sol \
     --policy-mode code --reasoning-effort low --output-dir ./runs/pyrua

Look in ``./runs/pyrua/code_policy/`` for the executed cells. Check
``environment_success`` in ``transcript_*.json`` for the native LIBERO result.
If the worker fails to start, check that ``bwrap`` is installed and user
namespaces are permitted. Once the worker starts launching, its startup
diagnostics appear in ``code_policy/worker.stderr.log``.

PyRUA mode supports the ``api``, ``claude_code``, ``codex``, and ``gemini_cli`` planners
via either the CLI or the Dashboard. It cannot be combined with
``--planner flash`` or ``--explore``. PyRUA mode does not load task
memory or generate an action recipe.

Run Without an Action Model
---------------------------

Use ``--policy-mode code --no-am`` to disable VLA and WAM services and tools.
For LIBERO this combination supports standard suites. Set up standard LIBERO
and Claude Code first, then set ``SAM3_CHECKPOINT_PATH`` for a local SAM3
server or pass ``--sam3-endpoint`` to reuse one.

.. code-block:: bash

   rpent --robot libero --libero-type standard \
     --suite libero_object --task 0 --seed 0 \
     --planner claude_code --model deepseek-v4.1-flash-1017 \
     --policy-mode code --no-am

The standard LIBERO controller uses SAM3 and RGB-D localization, then
native motion and gripper tools. Its action replies carry state. Call
``robo.view_env_state(step=-1)`` when you need images.

RoboDojo also supports ``--policy-mode code --no-am`` using native ARX-X5
controls. See :doc:`../simulators/robodojo` for its setup.

How Code Execution Works
------------------------

The planner submits code through ``python(code)`` and invokes robot tools using
``robo.<tool_name>(argument=value)``. Use ``print(...)`` to display
values; variables remain available in subsequent cells.

Python can import the standard library and NumPy and write to the session's
scratch directory. The sandbox separates the worker from the host workspace,
environment variables and network. Planner shell permissions are configured
separately.

Each cell has a default timeout of 120 seconds, including tool calls,
and allows up to 100 robot tool calls. These limits can be adjusted
using ``--code-timeout-s`` and ``--code-max-calls``, respectively.
When ``robo.finish(...)`` successfully returns a completion marker,
RPent terminates the current code cell and ends the planning session.
When available, use ``request_scene_reset`` and ``request_operator_verdict``
as separate planner tools so human input can arrive outside the cell timeout.


Inspect a Run
-------------

The following paths are relative to ``code_policy/`` in the run directory
selected by ``--output-dir``.

.. list-table::
   :header-rows: 1

   * - Path
     - Contents
   * - ``cell_*.py`` and ``cell_*.json``
     - Submitted code and its execution result, including output and exceptions.
   * - ``calls.jsonl``
     - Robot calls made from Python, with their results and durations.
   * - ``worker.stderr.log``
     - Sandbox startup and worker process diagnostics.
   * - ``scratch/``
     - Files written by Python cells during the session.


LIBERO-PRO Evaluation
---------------------

This evaluation used the Codex planner with GPT-6 Astra at high reasoning
effort. It covered the Spatial, Object, Goal, and Long LIBERO-PRO task
families, each with task and swap suites. Tasks 0–9 used one seed per suite.
The overall success rate was 80.00% (64/80 planned episodes).

The table summarizes results by suite. Token usage, elapsed time, and
planner turns are averaged over episodes with transcripts, including
both successful and unsuccessful episodes. The historical summary lacks
the transcript count, exact seed, action-model checkpoint, and the configured
values of ``--code-timeout-s``, ``--code-max-calls`` and ``--max-turns``.

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


Elapsed time uses ``elapsed_s`` from the run record, measured after the
environment runtime is ready. Planner turns use ``stats.turns_used``.
