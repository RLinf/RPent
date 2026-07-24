Overview
========

**RPent (Recursive Physical Agent)** is an open framework for building embodied 
agents that continuously evolve through recursive interaction with the physical world. 
Rather than prescribing a single foundation model, RPent provides a recursive agent 
framework that harnesses heterogeneous intelligence, including perception, reasoning, 
memory, execution, and self-evolution, into a unified physical agent. 
Through continuous interaction, reflection, and adaptation, RPent enables physical agents 
to acquire new capabilities and evolve beyond their initial design.

The name Pent is inspired by the Pentagram, whose five points symbolize the integration 
of multimodal intelligence into a unified embodied agent. At its center, the infinity 
symbol (∞) represents the endless recursive cycle of perception, reasoning, 
execution, and self-evolution, through which intelligence continuously expands into the physical world.

.. image:: https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png
   :alt: RPent framework diagram
   :align: center
   :width: 90%

RPent is built upon three core design principles: **service-oriented, standardized, and composable**. 
RPent enables capabilities to be deployed as reusable services, 
connected through unified interfaces, and flexibly composed into diverse physical agents. 
Together, these principles allow RPent to move beyond traditional robot control frameworks 
and establish an agentic infrastructure for the physical world, where intelligence 
is not only deployed, but continuously built, expanded, and evolved.

Feature Matrix
--------------

.. list-table::
   :header-rows: 1
   :widths: 26 28 26 20

   * - Agentic Planner
     - Action Primitive
     - Simulator
     - Real World
   * - - Claude Code ✅
       - Codex ✅
       - Custom planner ✅
     - - **VLA**

         - Pi0.5 ✅
         - RLDX-1

       - **WAM**

         - DreamZero
     - - LIBERO-PRO ✅
       - RoboCasa
     - - Franka
       - SO-101

Next steps
----------

- New to RPent? Complete :doc:`installation`, then follow
  :doc:`quickstart` to run one LIBERO task end-to-end.
- Want to use a specific robot environment? See its usage guide, such as
  :doc:`usage/libero`.
- Want to switch planners? See :doc:`usage/configure_planner`.
- Extending RPent for your own scenarios? See
  :doc:`development/architecture`.
