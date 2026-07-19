RoboCasa
========

`RoboCasa <https://robocasa.ai>`_ is the kitchen-scale, long-horizon
manipulation environment. In RPent it is driven by the **RLDX-1** VLA
policy, served over a pickle-framed socket RPC (rather than HTTP as
LIBERO uses), because RLDX observations are history-stacked nested
numpy dicts that ride sockets natively.

One-time setup
--------------

RoboCasa needs its assets, robosuite / composite-controller configs,
and the RLDX-1 checkpoint. Run:

.. code-block:: bash

   bash scripts/setup_robocasa.sh

The script downloads the RoboCasa365 assets, patches robosuite, and
places the RLDX-1 checkpoint where the ``vla_server`` expects it. See
`docs/SETUP_ROBOCASA.zh.md
<https://github.com/RLinf/RPent/blob/main/docs/SETUP_ROBOCASA.zh.md>`_
for the full walkthrough.

Minimal command
---------------

RoboCasa has its own convenience entrypoint:

.. code-block:: bash

   bash scripts/run_robocasa.sh <task> <gpu> <seed>

For example:

.. code-block:: bash

   bash scripts/run_robocasa.sh PickPlaceCounterToCabinet 0 0

Under the hood this launches the same three processes as LIBERO —
agent, env_server, vla_server — with the RoboCasa client classes and
the RLDX-1 model client wired in.

Available task families
-----------------------

RoboCasa in RPent supports the standard kitchen benchmarks:

- ``PickPlace*`` — pick objects from a source, place them at a target
  (counter → cabinet, sink → counter, …).
- ``Open*`` / ``Close*`` — open and close cabinet doors, drawers, and
  appliances.
- ``TurnOn*`` / ``TurnOff*`` — operate stove burners, microwave
  buttons, kettle switches, and similar toggles.

The exact list depends on the RoboCasa release you have installed;
inspect ``robots/robocasa/`` (once it lands under
``robots/``) or the RoboCasa upstream for the current catalog.

Toolkit differences vs. LIBERO
------------------------------

The RoboCasa toolkit exposes the same *shape* of tools (a primitive
call, a state view, a ``finish``), but two things are RoboCasa-specific:

- **Env-side helpers.** ``check_grasp`` and ``assemble_action`` — the
  eval-time ``unmap_action`` + composite-controller split-index
  assembly — need the live robosuite env, so they live in
  ``env_server`` as RPCs. The agent-side skill (``RLDXSkill``) holds
  **both** clients: the env client for render/step/grasp/assemble,
  the model client for RLDX-1 inference. See
  :doc:`../development/add_robot` for the rationale.
- **Observation shape.** RLDX-1 sees 3 camera video tensors
  ``(1, T, H, W, 3)`` stacked over history ``T``, plus ``state.*``
  fields, an annotation, and session / reset_memory. This is why the
  RoboCasa vla_server chose socket RPC over HTTP.

Reusing a running vla_server
----------------------------

Load RLDX-1 once, run many tasks:

.. code-block:: bash

   # Terminal 1: start the vla_server
   python -m robots.robocasa.vla_server --host 0.0.0.0 --port 8100

   # Terminal 2..N: point runs at it
   VLA_ENDPOINT=socket://host:8100 bash scripts/run_robocasa.sh \
     PickPlaceCounterToCabinet 0 0

The env server is cheap (a few seconds); the VLA is not.

Live dashboard
--------------

RoboCasa runs support ``--dashboard`` in the same way as LIBERO. Add
``--dashboard`` to whichever underlying ``cli/main.py`` invocation
your ``run_robocasa.sh`` shells out to.
