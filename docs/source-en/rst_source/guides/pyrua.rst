.. _pyrua-mode:

PyRUA Mode
==========

Use PyRUA Mode to let the planner group robot actions in Python and choose
the next action based on tool results. RPent saves the code and tool calls
so you can inspect how the task ran.

Run a Task
----------

Use Linux with user namespaces enabled and Bubblewrap (``bwrap``) installed.
On Ubuntu, run ``sudo apt install bubblewrap``. Complete steps 1 and 2 of
:doc:`../get_started/quickstart` to install LIBERO-PRO and download its assets
and checkpoints. Set up :ref:`Codex <planner-codex>` and check your connection
with ``rpent-check-llm --planner codex``.

Run LIBERO-PRO Object Swap task 2 with seed 0. The
``--policy-mode code`` option enables PyRUA Mode.

.. code-block:: bash

   rpent --robot libero --libero-type pro \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner codex \
     --policy-mode code --reasoning-effort low --output-dir ./runs/pyrua

After the run, open ``./runs/pyrua/``. The ``environment_success`` field
in ``transcript_*.json`` reports whether LIBERO marked the task successful.
See :ref:`pyrua-run-records` for the code and tool logs.

Use ``api``, ``claude_code``, ``codex``, or ``gemini_cli`` in the CLI or
:doc:`Dashboard <dashboard>`. Pass ``--policy-mode code`` when starting the
Dashboard. Set ``--policy-mode tools`` to use
``--planner flash``, ``--explore``, or task memory. See :doc:`cli` for options.

Run Without an Action Model
---------------------------

Add ``--no-am`` to use the robot's native controls with action model services
and tools disabled. For the example below, set up
:doc:`standard LIBERO <../simulators/libero>` in a separate Python environment
from LIBERO-PRO. Activate it, use the Codex configuration above, and set
``SAM3_CHECKPOINT_PATH`` for a local SAM3 service or pass
``--sam3-endpoint`` to reuse one. The example uses only the SAM3 checkpoint.

.. code-block:: bash

   rpent --robot libero --libero-type standard \
     --suite libero_object --task 0 --seed 0 \
     --planner codex \
     --policy-mode code --no-am --output-dir ./runs/pyrua-no-am

Check ``./runs/pyrua-no-am/code_policy/calls.jsonl`` for motion and gripper
calls. Standard LIBERO action replies omit images in this mode. Look for
``view_env_state`` in the call log to see when the planner requested images.

These robots and simulators also support both flags. Follow the linked guides
for environment, perception, and operator setup.

.. list-table::
   :header-rows: 1

   * - Robot or simulator
     - Control available with ``--no-am``
   * - :doc:`RoboDojo <../simulators/robodojo>`
     - Native ARX-X5 controls.
   * - :doc:`RoboCasa <../simulators/robocasa>` and :doc:`RoboTwin <../simulators/robotwin>`
     - Scripted motion and gripper tools.
   * - :doc:`RoboLab <../simulators/robolab>`
     - Native DROID actions.
   * - :doc:`EFE <../simulators/efe>`
     - Tools from the connected EFE controller. EFE already runs without an RPent action model.
   * - :doc:`Franka <../real_world_robots/franka>`, :doc:`Dual Franka <../real_world_robots/dual_franka>` and :doc:`YAM <../real_world_robots/yam>`
     - Native motion and gripper tools with the backend's operator controls.

Task 1 on both Franka and Dual Franka requires an action model. Use task 0
to check native motion and gripper control with ``--no-am``.

.. _pyrua-run-records:

Inspect a Run
-------------

Open ``code_policy/`` under the directory set by ``--output-dir``.
If PyRUA fails to start, check ``bwrap``, user namespaces, and any
diagnostics in ``worker.stderr.log``.

.. list-table::
   :header-rows: 1

   * - Path
     - Contents
   * - ``cell_*.py`` and ``cell_*.json``
     - Submitted code and its execution result, including output and exceptions.
   * - ``calls.jsonl``
     - Robot calls made from Python, with their results and durations.
   * - ``worker.stderr.log``
     - Startup and runtime diagnostics.
   * - ``scratch/``
     - Files written by Python cells during the session.

.. _how-code-execution-works:

Adjust Run Limits
-----------------

Each cell has 120 seconds, including tool calls, and up to 100 robot calls
by default. Change the limits with ``--code-timeout-s`` and
``--code-max-calls``. A call beyond the limit raises a Python error.
Time spent waiting for an operator's scene reset or verdict is outside
the cell timeout.

Python cells can use the standard library and NumPy and write to ``scratch/``.
The sandbox isolates cells from the host workspace and network. Codex shell
permissions use the separate ``--codex-sandbox`` setting.

Python variables carry over between cells. A timeout or cancellation clears
them, while completed robot actions remain in effect. Check the current robot
state before continuing.

.. warning::

   Use the robot's hardware emergency stop when motion must stop immediately.

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
excluded from both means. Elapsed time excludes service startup and includes
run cleanup. For retried episodes, only the final attempt counts. Input tokens
include cached input. See :doc:`Time & Token Costs <../leaderboard/time-token-costs>`.

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
