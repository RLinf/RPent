Interactive Usage
=================

Use the terminal to enter instructions, or the Dashboard to submit tasks and
watch planner output, cameras, and actions in a browser. The examples use
LIBERO after :doc:`../get_started/quickstart`; install other platforms through their
environment guides first.

.. _terminal-interaction:

Terminal Interaction (TUI)
--------------------------

Add ``--tui`` (``-t``) to a task command to enter instructions in the
terminal. Use a terminal with interactive input (TTY).

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner claude_code --model claude-opus-4-8 --tui

With ``claude_code`` or ``codex``, the first prompt contains the task
instruction. Edit it or submit it with Enter. During execution, type a message
to guide the agent. Claude Code interrupts the current turn before receiving
your message; Codex steers the current turn. Use ``/help`` to see commands and
``/quit``, ``/exit``, or ``/q`` to end the session.

With the ``api`` planner, the selected task starts automatically. After the
agent responds, the terminal accepts follow-up instructions with the existing
conversation history. Use ``/exit`` to close it. For example:

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner api --model anthropic:claude-opus-4-8 --tui

Configure credentials as described in :doc:`configure_planner`.
``--tui`` and ``--dashboard`` cannot be used together.
For physical robot setup and human confirmation, see the
:doc:`../real_world_robots/franka` and :doc:`../real_world_robots/dual_franka` guides.

.. _dashboard-usage:

Dashboard
---------

The Dashboard supports the ``api``, ``claude_code``, and ``codex`` planners.
Configure ``--planner`` and ``--model`` on the command line as for a normal
run; see :doc:`configure_planner`. ``--planner manual`` runs no planner and
lets the operator drive the robot from the Dashboard instead; see
:ref:`dashboard-teleoperation`.

Start a Session and Submit a Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add ``--dashboard`` to start a long-lived local Dashboard Session. It
selects an available port and prints the URL in the terminal:

.. code-block:: bash

   rpent --robot libero --dashboard \
     --planner claude_code --model claude-opus-4-8

Session configuration comes from the command line, and the URL opens directly
in the live monitor. After the shared services are ready, start a TaskRun with:

.. code-block:: text

   /rpent-task libero_object_swap 2 0

Each TaskRun gets a fresh environment while the VLA and SAM3 services are
reused by the Session. Submit a new ``/rpent-task`` to start or switch tasks;
during a run, normal messages steer the agent and Esc requests an interruption.
Press Ctrl+C in the terminal to stop the Session.

``--dashboard`` cannot be combined with ``--tui`` or
``--env-endpoint``. External ``--vla-endpoint`` and ``--sam3-endpoint``
services remain supported. Use ``--dashboard-language zh-cn`` for the
Chinese UI.

.. _dashboard-teleoperation:

Teleoperate Without a Planner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--planner manual`` starts the same Session with no language model: the
TaskRun hands the base to the operator instead of to a planner.

.. code-block:: bash

   rpent --robot lekiwi --dashboard --planner manual \
     --env-endpoint http://ROBOT_HOST:8120

The LeKiwi environment is external, so ``--env-endpoint`` is required: start
the env server on the robot's computer first, as described in
:doc:`../real_world_robots/lekiwi`.

Start a TaskRun with ``/rpent-task`` as usual. The Teleoperation panel then
drives the base: hold a direction button to move, release it to stop. Each
press sends the velocity to the env server, which steps the wheels at the
robot's own control rate until the command expires or a stop arrives, so
network jitter delays a change of direction but never pauses the base. A
command that stops being refreshed expires after one second, so a closed or
disconnected browser leaves the robot at rest. The sliders set the linear
speed in m/s and the turning rate in deg/s. ``End run`` finishes the TaskRun
and returns the Session to waiting for the next task.

The two checkboxes choose what the run keeps, and can be changed while it
runs. With ``Record video`` on, each continuous stretch of driving is written
as ``teleop_NNN.mp4`` in the run output; turning it off saves the stretch
recorded so far and stops collecting frames, while the live camera keeps
updating. With ``Record actions`` on, each stretch driven at one constant
velocity is appended to ``teleop_actions.jsonl``, with the velocity, its
duration, and the number of control steps it took, and is shown in the action
timeline.

``--planner manual`` requires ``--dashboard`` and cannot be combined with
``--explore``, ``--model``, or ``--base-url``.

Inspect Results and End the Session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a task ends, review the action timeline and video; use the environment guide for its success criterion. The run output reports the artifact directory, including ``transcript_*.json``. Closing the browser does not stop backend services; press Ctrl+C in the launch terminal to end the session.

The default bind address is ``127.0.0.1``. For remote access, use the SSH forwarding instructions in :doc:`advanced_deployment`.
