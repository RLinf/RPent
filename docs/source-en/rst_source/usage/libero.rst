LIBERO
======

`LIBERO <https://libero-project.github.io/>`_ is RPent's primary simulation
benchmark for MuJoCo/robosuite-based tabletop manipulation.
RPent focuses on four core base task families (``libero_object``,
``libero_goal``, ``libero_spatial``, ``libero_10``) and three variants
(``standard``, ``pro``, ``plus``).
The default VLA is **Pi0.5**, served over HTTP by
``robots/libero/vla_server.py``.

VLA configuration
-----------------

Before using Pi0.5, point ``PI05_CHECKPOINT_PATH`` to the local checkpoint
directory:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

Download the recommended SFT checkpoint from HuggingFace:
`RLinf-Pi05-LIBERO-130-fullshot-SFT
<https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT>`_.

Task selection
--------------

A LIBERO run uses the following task settings:

- ``--suite`` — selects the task suite to run. See
  :ref:`libero-pro-core-suites` for the complete core-suite list.
- ``--task`` — the task index within the suite.
- ``--seed`` — the environment seed.
- ``--libero-type`` — the LIBERO variant: ``standard`` | ``pro`` |
  ``plus``. If omitted, RPent falls back to ``LIBERO_TYPE`` in the
  environment (default ``pro``).

.. _libero-pro-core-suites:

Core LIBERO-PRO suites
~~~~~~~~~~~~~~~~~~~~~~

This table covers RPent's four core LIBERO-PRO task families and all of
their perturbation suites.

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Family
     - Base suite
     - Perturbation suites
   * - Object
     - ``libero_object``
     - ``libero_object_task``, ``libero_object_swap``,
       ``libero_object_lan``, ``libero_object_object``
   * - Goal
     - ``libero_goal``
     - ``libero_goal_task``, ``libero_goal_swap``,
       ``libero_goal_lan``, ``libero_goal_object``
   * - Spatial
     - ``libero_spatial``
     - ``libero_spatial_task``, ``libero_spatial_swap``,
       ``libero_spatial_lan``, ``libero_spatial_object``
   * - LIBERO-10
     - ``libero_10``
     - ``libero_10_task``, ``libero_10_swap``, ``libero_10_lan``,
       ``libero_10_object``

The suffixes identify LIBERO-PRO perturbations: ``_task`` is Task/P1,
``_swap`` is Position/P2, ``_lan`` is Semantic, and ``_object`` is Object.

Minimal command
---------------

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   export LIBERO_TYPE=pro
   export CUDA_VISIBLE_DEVICES=0

   rpent --env libero \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

What runs where
---------------

- **env_server** (``robots/libero/env_server.py``) — owns the LIBERO
  MuJoCo env and EGL rendering. Exposes ``reset``, ``step``,
  ``chunk_step``, ``render_camera``, ``get_camera_meta``,
  ``cached_image``, … over an RPC transport (HTTP by default; socket
  via ``--transport socket``).
- **vla_server** (``robots/libero/vla_server.py``) — owns the Pi0.5
  weights. Exposes ``predict`` over the same RPC transport (HTTP or
  socket).
- **Toolkit** (``robots/libero/toolkit.py``) — defines the tools the
  LLM can call: ``pi0_pick`` (fed to Pi0.5), ``move_to``,
  ``rotate_wrist``, ``back_project``, ``view_driver_state``,
  ``finish``, …

Tools the planner sees
----------------------

Key LIBERO tools include:

- ``pi0_pick(prompt, ...)`` — use Pi0.5 to execute a closed-loop grasp.
- ``pi0_doubled(prompt, ...)`` — use Pi0.5 for a non-pick contact action.
- ``move_to(xyz, ...)`` — move the end effector to a world-frame position.
- ``move_pose(xyz, target_pitch=..., target_yaw=..., ...)`` — move position
  and orientation together.
- ``rotate_wrist(target_yaw=... / delta_yaw=..., ...)`` — rotate wrist yaw
  to an absolute target or by a relative amount.
- ``rotate_pitch(target_pitch=... / delta_pitch=..., ...)`` — tilt the
  gripper to an absolute pitch or by a relative amount.
- ``set_gripper(gripper=..., steps=...)`` — hold the pose and drive the
  gripper for a fixed number of steps.
- ``release(...)`` — open the gripper.
- ``back_project(row, col, ...)`` — back-project an image pixel to world
  coordinates.
- ``segment(prompt=... / point=..., ...)`` — segment an existing image with
  a text or point prompt.
- ``view_driver_state(step=None)`` — read an existing state and image record.
- ``view_camera_meta(camera=..., step=None)`` — read existing camera metadata.
- ``finish(status, summary)`` — end the current run.

Physical action tools record new state and images after execution. Read-only
tools do not advance the environment.

Live dashboard
--------------

Add ``--dashboard`` to start a local monitor. By default, it selects an
available port and prints the URL in the terminal; pass
``--dashboard-port <port>`` to use a fixed port:

.. code-block:: bash

   rpent --env libero --dashboard \
     --suite libero_goal_task --task 1 --seed 0 \
     --planner claude_code --model claude-opus-4-8

The dashboard streams reasoning, agentview + wrist camera + Pi0.5
overlays, and an action timeline. Use
``--dashboard-language zh-cn`` for the Chinese UI.

Bringing your own VLA
---------------------

If you have a LIBERO-compatible VLA that is not Pi0.5, swap the model
client without touching the env by:

1. Writing a new ``vla_server.py`` that exposes the same ``predict``
   RPC contract (over http or socket).
2. Pointing at it with ``--vla-endpoint [protocol://]host:port``.
3. Optionally updating ``robots/libero/toolkit.py`` if the tool
   surface (e.g. ``pi0_pick`` → ``mymodel_pick``) needs to change.

See :doc:`../development/add_primitive` for the full walkthrough.
