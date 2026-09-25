Dashboard
=========

Use the Dashboard to watch planner output, cameras, and actions, and submit tasks in a browser. This example uses LIBERO after :doc:`../quickstart`; install other platforms through their environment guides first.

Start a Session and Submit a Task
---------------------------------

Add ``--dashboard`` to start a long-lived local Dashboard Session. It
selects an available port and prints the URL in the terminal:

.. code-block:: bash

   rpent --robot libero --dashboard \
     --planner claude_code --model claude-opus-4-8

Session configuration comes from the command line, and the URL opens directly
in the live monitor. After the shared services are ready, start a TaskRun with:

.. code-block:: text

   /rpent-task libero_object_swap 2 0

The Dashboard supports the ``api``, ``claude_code``, and ``codex`` planners.
Configure ``--planner`` and ``--model`` on the command line as for a normal
run; see :doc:`configure_planner`.

Each TaskRun gets a fresh environment while the VLA and SAM3 services are
reused by the Session. Submit a new ``/rpent-task`` to start or switch tasks;
during a run, normal messages steer the agent and Esc requests an interruption.
Press Ctrl+C in the terminal to stop the Session.

``--dashboard`` cannot be combined with ``--interactive`` or
``--env-endpoint``. External ``--vla-endpoint`` and ``--sam3-endpoint``
services remain supported. Use ``--dashboard-language zh-cn`` for the
Chinese UI.

Inspect Results and End the Session
-----------------------------------

After a task ends, review the action timeline and video; use the environment guide for its success criterion. The run output reports the artifact directory, including ``transcript_*.json``. Closing the browser does not stop backend services; press Ctrl+C in the launch terminal to end the session.

The default bind address is ``127.0.0.1``. For remote access, use the SSH forwarding instructions in :doc:`advanced_deployment`.
