
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
:doc:`../get_started/quickstart` and set up Codex using
:doc:`configure_planner`.

Add ``--policy-mode code`` to a regular LIBERO-PRO task command to
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
memory or generate an action recipe. See :doc:`cli` for the full option list.

Run Without an Action Model
---------------------------

Use ``--policy-mode code --no-am`` to disable VLA and WAM services and tools.
For LIBERO this combination supports standard suites. Set up standard LIBERO
with :doc:`../simulators/libero` and Claude Code with :doc:`configure_planner`.
Set ``SAM3_CHECKPOINT_PATH`` for a local SAM3 server or pass
``--sam3-endpoint`` to reuse one.

.. code-block:: bash

   rpent --robot libero --libero-type standard \
     --suite libero_object --task 0 --seed 0 \
     --planner claude_code --model deepseek-v4.1-flash-1017 \
     --policy-mode code --no-am

The planner uses SAM3 and RGB-D to locate objects. Check
``code_policy/calls.jsonl`` in the run directory for native motion and gripper
calls. In this mode, standard LIBERO action replies contain state without
images. The planner calls
``robo.view_env_state(step=-1)`` when it needs a fresh image.

RoboDojo also supports ``--policy-mode code --no-am`` using native ARX-X5
controls. See :doc:`../simulators/robodojo` for its setup.

How Code Execution Works
------------------------

The planner submits code through ``python(code)`` and invokes robot tools using
``robo.<tool_name>(argument=value)``. It can use ``print(...)`` to display
values and reuse variables in subsequent cells.

Python can import the standard library and NumPy and write to the session's
scratch directory. The sandbox separates the worker from the host workspace,
environment variables and network. For Codex planner shell permissions,
see ``--codex-sandbox`` in the :doc:`CLI reference <cli>`.

Each cell has a default timeout of 120 seconds, including tool calls,
and allows up to 100 robot tool calls. Adjust these limits with
``--code-timeout-s`` and ``--code-max-calls``, respectively. A call beyond
the limit raises a Python error. A timeout or cancellation discards the
Python namespace. Robot actions already executed remain in effect.

.. warning::

   Stopping the Python worker does not trigger the robot's hardware
   emergency stop.

When ``robo.finish(...)`` successfully returns a completion marker,
RPent terminates the current code cell and ends the planning session.
When available, the planner calls ``request_scene_reset`` and
``request_operator_verdict`` directly so the wait for human input falls
outside the cell timeout.


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

PyRUA Mode succeeded in 663 of 800 LIBERO-PRO episodes (82.88%).
The evaluation covered Spatial, Object, Goal, and
Long, with task and swap suites, tasks 0–9, and seeds 1–10 in each suite.
An episode counts as a success when LIBERO reports ``environment_success`` as
``true`` in the transcript. The success rate includes all
800 cases in the denominator. Failed and incomplete cases, including timeouts,
count as failures. See :doc:`Performance <../leaderboard/performance>`.

Mean time and mean total tokens per episode use the 786 complete final
records, including 663 successes and 123 failures. The 14 timeouts are
excluded from both means. See :doc:`Time & Token Costs <../leaderboard/time-token-costs>`.

The evaluation used the API planner with DeepSeek V4.1 Flash
(``deepseek-v4.1-flash-1017``), high reasoning effort, Pi0.5 for action
execution, and no task memory. Runs allowed up to 100 planner turns;
each Python cell allowed 120 seconds and 100 robot tool calls.

.. list-table:: PyRUA evaluation on LIBERO-PRO (800 cases)
   :header-rows: 1

   * - Suite
     - Success Rate (successes/cases)
     - Mean Total Tokens (input + output)
     - Mean Time (s)
     - Time & Token Samples
   * - Spatial Task
     - 95.00% (95/100)
     - 1,104,045.70
     - 567.25
     - 99
   * - Spatial Swap
     - 87.00% (87/100)
     - 1,550,259.47
     - 663.28
     - 94
   * - Object Task
     - 100.00% (100/100)
     - 435,436.86
     - 217.44
     - 100
   * - Object Swap
     - 98.00% (98/100)
     - 527,173.55
     - 263.90
     - 100
   * - Goal Task
     - 79.00% (79/100)
     - 1,775,835.65
     - 769.18
     - 93
   * - Goal Swap
     - 90.00% (90/100)
     - 1,951,382.73
     - 861.88
     - 100
   * - Long Task
     - 60.00% (60/100)
     - 3,345,833.40
     - 1,387.30
     - 100
   * - Long Swap
     - 54.00% (54/100)
     - 3,621,965.83
     - 1,515.42
     - 100
   * - Overall
     - 82.88% (663/800)
     - 1,791,802.63
     - 781.98
     - 786
