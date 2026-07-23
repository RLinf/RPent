Action Primitives
=================

Where the planner chooses *what* to do, the **action primitive**
chooses *how* it happens. A primitive turns a tool call
(``pi0_pick``, ``move_to``, ``open_drawer``, …) into an executable
action chunk for the environment.

RPent supports two families of primitives out of the box:

- **VLA policies** (Vision-Language-Action models). These run in the
  dedicated ``vla_server`` process, keep GPU weights isolated from
  the physics engine, and are called by the toolkit through a per-env
  model client. Examples: Pi0.5 (LIBERO), RLDX-1 (RoboCasa).
- **Scripted primitives**. Deterministic motions such as ``move_to``,
  ``rotate_wrist``, ``release``, or ``back_project``. They live on the
  agent side (no VLA weights needed) and are wired directly to
  ``env_server`` RPCs.

For the concrete per-environment configuration (which VLA runs
against which robot, checkpoint paths, tool surface), see the
environment pages: :doc:`libero`, :doc:`robocasa`, :doc:`franka`,
:doc:`so101`.

Which VLA runs where
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Environment / robot
     - Default VLA
     - Wire codec
     - Server
   * - LIBERO (sim)
     - Pi0.5
     - HTTP or socket RPC (``--transport``)
     - ``robots/libero/vla_server.py``
   * - RoboCasa (sim)
     - RLDX-1
     - pickle-framed socket RPC
     - ``robots/robocasa/vla_server.py`` *(planned)*
   * - Franka (real)
     - Pi0.5 or RLDX-1 (task-dependent)
     - HTTP or socket
     - ``robots/franka/vla_server.py`` *(planned)*
   * - SO-101 (real)
     - RLDX-1 (task-dependent)
     - socket RPC
     - ``robots/so101/vla_server.py`` *(planned)*

The wire codec is chosen per env to fit the observation shape. The
VLA server exposes the same ``predict`` / ``healthz`` methods over both
HTTP (JSON) and socket (pickle-framed) transports; pick whichever suits
the observation shape via ``--transport {http,socket}`` (defaults to
``http``). See :doc:`../development/add_robot` for the design rationale.

Reusing a running VLA server
----------------------------

Every VLA server is designed to be **shared across runs**. Point at an
already-running instance with ``--vla-endpoint`` instead of spawning a
new one each time:

.. code-block:: bash

   rpent --env libero --vla-endpoint http://localhost:8000 \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

``--vla-endpoint`` accepts ``[protocol://]host:port``. Protocol may be
``http`` (default) or ``socket``. The same applies to
``--env-endpoint`` for reusing an env_server.

Adding a brand-new primitive family
-----------------------------------

If the primitive you want is neither a VLA nor a scripted motion —
say a WAM (World Action Model), a diffusion planner, or a Model
Predictive Control primitive — see :doc:`../development/add_primitive`.
