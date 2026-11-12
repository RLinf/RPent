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

With ``claude_code``, ``codex``, or ``gemini_cli``, the first prompt contains the
task instruction. Edit it or submit it with Enter. Use ``/help`` to see commands
and ``/quit``, ``/exit``, or ``/q`` to end the session.

For example, while Claude Code, Codex, or Gemini CLI is running a task, you
want it to check the cup's position. Choose an input below and press Enter:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example input
     - Behavior
   * - ``Check the cup's position``
     - Append the message to the queue and wait for the current agent turn
       to end. A turn can include several tool calls; finishing one tool call
       does not immediately send the message.
   * - ``/interrupt``
     - Interrupt the current turn, then send the first queued message. For
       example, this sends "Check the cup's position" if it is first in the queue.
   * - ``/interrupt Check the cup's position first``
     - Interrupt the current turn and send "Check the cup's position first"
       ahead of the existing queued messages, which remain queued.

Follow-ups use the same conversation, with one message sent after each turn.
On interruption, RPent cancels the active tool and waits for it to stop, then
interrupts the backend turn. It sends the message only after the backend has
stopped. If interruption fails, messages remain queued until a successful
interrupt retry. Completed physical actions are not undone.

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
during a run, submit follow-up messages or press Esc to request an interruption.
Press Ctrl+C in the terminal to stop the Session.

With Claude Code, Codex, and Gemini CLI, ordinary messages queue until the
current agent turn ends. For example, to ask the planner to check the cup's
position during a task:

1. Enter "Check the cup's position first" in the input box and press Enter.
   The message joins the queue.
2. To interrupt instead of waiting for the turn to end, press Esc while the
   input box has focus.
3. RPent cancels the active tool and waits for it to stop, then interrupts the
   backend turn. Once it has stopped, RPent automatically sends the first
   queued message and continues in the same conversation.

Esc does not reorder the queue; any earlier queued messages are sent first.
If interruption fails, no new message is sent until a successful interrupt
retry. A successful ``finish`` ends the task without sending pending messages.

``--dashboard`` cannot be combined with ``--tui`` or
``--env-endpoint``. External ``--am-endpoint`` and ``--sam3-endpoint``
services remain supported. Use ``--dashboard-language zh-cn`` for the
Chinese UI.

Inspect Results and End the Session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a task ends, review the action timeline and video; use the environment guide for its success criterion. The run output reports the artifact directory, including ``transcript_*.json``. Closing the browser does not stop backend services; press Ctrl+C in the launch terminal to end the session.

The default bind address is ``127.0.0.1``. For remote access, use the SSH forwarding instructions in :doc:`advanced_deployment`.
