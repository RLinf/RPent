Interactive Usage
=================

Use the terminal to enter instructions, or the Dashboard to submit tasks and
watch planner output, cameras, and actions in a browser. The examples use
LIBERO after :doc:`../get_started/quickstart`; install other platforms through their
environment guides first.

.. _terminal-interaction:

Terminal Interaction (TUI)
--------------------------

Add ``--interactive`` (``-i``) to a task command to enter instructions in the
terminal. Use a terminal with interactive input (TTY).

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner claude_code --model claude-opus-4-8 --interactive

With ``claude_code``, ``codex``, or ``gemini_cli``, the first prompt contains the
task instruction. Edit it or submit it with Enter. Use ``/help`` to see commands
and ``/quit``, ``/exit``, or ``/q`` to end the session.

Claude Code, Codex, and Gemini queue additional messages in arrival order and
submit one message after each complete agent turn, not after each tool call. Use
``/interrupt`` to stop the current turn and send the next queued message.
``/interrupt new instruction`` puts that instruction ahead of the remaining
queue. Both continue in the same conversation. Tool execution is cancelled and
drained before the backend is interrupted; new input is sent only after the
backend confirms completion. If interruption fails, messages remain queued
until a successful interrupt retry. Completed physical actions are not undone.

With the ``api`` planner, the selected task starts automatically. After the
agent responds, the terminal accepts follow-up instructions with the existing
conversation history. Use ``/exit`` to close it. For example:

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner api --model anthropic:claude-opus-4-8 --interactive

Configure credentials as described in :doc:`configure_planner`.
``--interactive`` and ``--dashboard`` cannot be used together.
For physical robot setup and human confirmation, see the
:doc:`../real_world_robots/franka` and :doc:`../real_world_robots/dual_franka` guides.

.. _dashboard-usage:

Dashboard
---------

The Dashboard supports the ``api``, ``claude_code``, ``codex``, and ``gemini_cli`` planners.
Configure ``--planner`` and ``--model`` on the command line as for a normal
run; see :doc:`configure_planner`.

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

With Claude Code, Codex, and Gemini, ordinary messages queue until the current
agent turn completes. To continue sooner, submit the message and press Esc: the active tool
and backend turn are stopped before the next queued message is sent. Remaining
messages keep their queue order. A successful ``finish`` ends the task without
sending pending messages.

``--dashboard`` cannot be combined with ``--interactive`` or
``--env-endpoint``. External ``--vla-endpoint`` and ``--sam3-endpoint``
services remain supported. Use ``--dashboard-language zh-cn`` for the
Chinese UI.

Inspect Results and End the Session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a task ends, review the action timeline and video; use the environment guide for its success criterion. The run output reports the artifact directory, including ``transcript_*.json``. Closing the browser does not stop backend services; press Ctrl+C in the launch terminal to end the session.

The default bind address is ``127.0.0.1``. For remote access, use the SSH forwarding instructions in :doc:`advanced_deployment`.
