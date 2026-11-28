.. Simulator guide template: copy this file, replace every <...>, and delete
.. these comments. Rules: .agents/skills/docs/rules/simulator.md and
.. docs/STYLE_GUIDE.md.

<Simulator Name>
================

.. Optional overview figure (remove if no upstream figure is needed):
.. .. figure:: https://raw.githubusercontent.com/<org>/<repo>/<branch>/<path>
..    :alt: <Simulator Name> overview
..    :width: 90%
..    :align: center
..
..    <What the figure shows>. Source: `<Project> <<URL>>`_.

Run <task domain> tasks with RPent in `<Simulator Name> <<project URL>>`_. <One sentence on simulator backend, supported variants, and default action model.>

Overview
--------

Check the supported models, planners, tasks, and hardware requirements before installing.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      <Default VLA/WAM and any optional action models.>

   .. grid-item-card:: Planners

      ``api``, ``claude_code``, ``codex``, ``gemini_cli``

   .. grid-item-card:: Tasks

      <Task suites or categories.>

   .. grid-item-card:: Hardware

      Linux, NVIDIA GPU; Python 3.11; CUDA and EGL.

Tasks
~~~~~

<One sentence summarizing the supported suites or task families.>

.. list-table::
   :header-rows: 1

   * - Suite / Family
     - Scope
     - Notes
   * - ``<suite_name>``
     - <task count or scope>
     - <one-line description>

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The table distinguishes planner tools, action-model inputs, and the environment's native success criterion.

.. list-table::
   :header-rows: 1

   * - Item
     - Description
   * - Observation
     - <Camera views, depth/state inputs seen by the planner and action model.>
   * - Action
     - <Tools and primitives exposed to the planner, and how they step the environment.>
   * - Reward / success
     - <Exact native success criterion (for example final ``terminated``). A step limit or planner ``finish`` is not a success label.>
   * - Task prompt
     - <Where the task language instruction comes from.>

Installation and Resources
--------------------------

Install the ``robot-<name>`` component into an isolated virtual environment:

.. code-block:: bash

   ./rpent-install.sh install robot-<name> --venv .venv-<name>
   source .venv-<name>/bin/activate

Download the required simulator assets and action-model weights:

.. code-block:: bash

   <asset and checkpoint download commands>

Run a Task
----------

Configure your planner credentials (see :doc:`../guides/configure_planner`), then run one task:

.. code-block:: bash

   rpent --robot <robot_id> --planner <planner> <key flags>

Key options you may adjust:

- ``<flag>``: <what it controls and default>.

View Results
------------

Each run saves step observations, tool logs, and final status under ``--output-dir``. Check ``<result file and native success field>`` to verify task success.
