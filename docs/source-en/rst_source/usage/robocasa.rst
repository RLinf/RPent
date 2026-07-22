RoboCasa
========

`RoboCasa <https://robocasa.ai>`_ is the kitchen-scale, long-horizon
manipulation environment. In RPent it is driven by the **RLDX-1** VLA
policy, served over a pickle-framed socket RPC (rather than HTTP as
LIBERO uses), because RLDX observations are history-stacked nested
numpy dicts that ride sockets natively.

Task families
-------------

RoboCasa covers the standard kitchen benchmarks:

- ``PickPlace*`` — pick objects from a source, place them at a target
  (counter → cabinet, sink → counter, …).
- ``Open*`` / ``Close*`` — open and close cabinet doors, drawers, and
  appliances.
- ``TurnOn*`` / ``TurnOff*`` — operate stove burners, microwave
  buttons, kettle switches, and similar toggles.

The exact catalog depends on the RoboCasa release; see the
`RoboCasa <https://robocasa.ai>`_ upstream for the current list.

Toolkit design vs. LIBERO
-------------------------

The RoboCasa toolkit exposes the same *shape* of tools as LIBERO (a
primitive call, a state view, a ``finish``), with two RoboCasa-specific
aspects:

- **Env-side helpers.** Grasp checks and action assembly need the live
  simulator env, so they live in ``env_server`` as RPCs. The agent-side
  skill holds **both** clients: the env client for render/step, the
  model client for RLDX-1 inference. See
  :doc:`../development/add_robot` for the rationale.
- **Observation shape.** RLDX-1 sees 3 camera video tensors
  ``(1, T, H, W, 3)`` stacked over history ``T``, plus ``state.*``
  fields, an annotation, and session / reset_memory.
