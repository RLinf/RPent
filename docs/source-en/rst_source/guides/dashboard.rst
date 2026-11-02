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

With ``claude_code``, ``codex``, or ``deepseek_harness``, the first prompt
contains the task instruction. Edit it or submit it with Enter. Claude Code
interrupts the current turn before receiving follow-up messages; Codex steers
the current turn. Use ``/help`` to see commands and ``/quit``, ``/exit``, or
``/q`` to end the session.

For DeepSeek Harness, complete :ref:`login <planner-deepseek-harness>`, then run:

.. code-block:: bash

   rpent --robot libero --libero-type pro --suite libero_10_task --task 0 --seed 0 \
     --planner deepseek_harness --dsh-home "$DSH_HOME" \
     --model gpt-6.1-sol --reasoning-effort xhigh --tui

Follow-up messages queue while the model or a tool is busy; sending a message
does not interrupt an action. Enter ``/interrupt`` and wait for the pause to be
confirmed, then submit a message to continue. An accepted ``finish`` ends the
current task. Continuation is limited to the running process.

Ctrl-D on an empty input line signals end of input (EOF) and ends the run;
no Enter is needed. Clear any prefilled or typed text first. Enter on an empty
line is ignored. Ctrl+C ends the process rather than requesting a resumable pause.

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

The Dashboard supports the ``api``, ``claude_code``, ``codex``, and
``deepseek_harness`` planners.
Configure ``--planner`` and ``--model`` on the command line as for a normal
run; see :doc:`configure_planner`.

Start a Session and Submit a Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add ``--dashboard`` to start a long-lived local Dashboard Session. It
selects an available port and prints the URL in the terminal:

.. code-block:: bash

   rpent --robot libero --dashboard \
     --planner claude_code --model claude-opus-4-8

For DeepSeek Harness, use the same login home and model settings as the terminal:

.. code-block:: bash

   rpent --robot libero --dashboard --planner deepseek_harness \
     --dsh-home "$DSH_HOME" --model gpt-6.1-sol --reasoning-effort xhigh

Session configuration comes from the command line and is retained when switching
tasks. A new process needs these options again. The URL opens directly
in the live monitor. After the shared services are ready, start a TaskRun with:

.. code-block:: text

   /rpent-task libero_object_swap 2 0

Each TaskRun gets a fresh environment while the VLA and SAM3 services are
reused by the Session. Submit a new ``/rpent-task`` to start or switch tasks;
during a run, normal messages steer the agent and Esc requests an interruption.
Press Ctrl+C in the terminal to stop the Session.

With DeepSeek Harness, busy messages queue without interrupting execution. A
message is marked sent only after the agent consumes it; withdraw it before
consumption if needed. Esc requests a pause. Wait for the idle state before
using manual controls; manual actions are disabled while the agent is busy.
To continue, submit a new message or resubmit an unsent message. The agent
receives fresh observations after manual actions.

If execution, result delivery, or cleanup cannot be confirmed, the Session
closes and queued replacement tasks do not start. Saved artifacts do not enable
cross-process recovery.

``--dashboard`` cannot be combined with ``--tui`` or
``--env-endpoint``. External ``--vla-endpoint`` and ``--sam3-endpoint``
services remain supported. Use ``--dashboard-language zh-cn`` for the
Chinese UI.

Inspect Results and End the Session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a task ends, review the action timeline and video; use the environment guide for its success criterion. The run output reports the artifact directory, including ``transcript_*.json``. Closing the browser does not stop backend services; press Ctrl+C in the launch terminal to end the session.

The default bind address is ``127.0.0.1``. For remote access, use the SSH forwarding instructions in :doc:`advanced_deployment`.
