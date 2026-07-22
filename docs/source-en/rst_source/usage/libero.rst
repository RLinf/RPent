LIBERO
======

`LIBERO <https://libero-project.github.io/>`_ is the default RPent
environment: a MuJoCo/robosuite-based tabletop manipulation benchmark
with four suites (``libero_object``, ``libero_goal``, ``libero_spatial``,
``libero_10``) and three variants (``standard``, ``pro``, ``plus``).
The default VLA is **Pi0.5**, served over HTTP by
``robots/libero/vla_server.py``.

VLA configuration
-----------------

Pi0.5 needs one thing: a checkpoint on disk. Point at it via
``PI05_CHECKPOINT_PATH``:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

Download the recommended SFT checkpoint from HuggingFace:
`rlinf-pi05-libero-130-fullshot-sft
<https://huggingface.co/datasets/RLinf/rlinf-pi05-libero-130-fullshot-sft>`_.

Task selection
--------------

Every LIBERO run picks:

- ``--suite`` — one of the four suites, each optionally suffixed with
  the variant flavor (see below). Examples:
  ``libero_object_task``, ``libero_object_swap``,
  ``libero_goal_lan``, ``libero_spatial_task``,
  ``libero_10_swap``.
- ``--task`` — the task index within the suite.
- ``--seed`` — the environment seed.
- ``--libero-type`` — the LIBERO variant: ``standard`` | ``pro`` |
  ``plus``. If omitted, RPent falls back to ``LIBERO_TYPE`` in the
  environment (default ``pro``).

Suite × variant matrix
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Suite
     - Variants
     - Purpose
   * - ``libero_object``
     - ``_task`` / ``_swap`` / ``_lan``
     - Object-centric tasks with optional target swap or language
       perturbations.
   * - ``libero_goal``
     - ``_task`` / ``_swap`` / ``_lan``
     - Goal-conditioned tasks with optional swap / language
       perturbations.
   * - ``libero_spatial``
     - ``_task`` / ``_lan``
     - Spatial-relations tasks.
   * - ``libero_10``
     - ``_task`` / ``_swap`` / ``_lan``
     - The long-horizon LIBERO-10 suite.

Minimal command
---------------

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   export LIBERO_TYPE=pro
   export CUDA_VISIBLE_DEVICES=0

   rpent --env libero \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner api --model anthropic:claude-opus-4-8 \
     --max-tokens 8192

What runs where
---------------

- **env_server** (``robots/libero/env_server.py``) — owns the LIBERO
  MuJoCo env and EGL rendering. Exposes ``reset``, ``step``,
  ``chunk_step``, ``render_agentview``, ``get_camera_meta``,
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

By default the LIBERO toolkit exposes:

- ``pi0_pick(target)`` — invoke Pi0.5 for a pick chunk targeting
  ``target`` (a natural-language object description).
- ``move_to(dx, dy, dz)`` — scripted Cartesian motion (deterministic;
  no VLA).
- ``rotate_wrist(delta_rad)`` — scripted wrist rotation.
- ``release()`` — open the gripper.
- ``back_project(pixel_x, pixel_y)`` — turn a click on the agentview
  image into a 3D point in world coordinates.
- ``view_driver_state()`` — force a fresh state dump (images, depths,
  camera meta, ``states.json``).
- ``finish(status)`` — end the episode with ``success`` / ``failure``
  / ``stuck``.

Every tool re-renders the world after it runs, so the next turn's
context reflects the post-action state.

Live dashboard
--------------

Add ``--dashboard`` to open a local monitor for a LIBERO run:

.. code-block:: bash

   rpent --env libero --dashboard \
     --suite libero_goal_task --task 1 --seed 0 --planner claude_code

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
