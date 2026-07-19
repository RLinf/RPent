Overview
========

RPent — short for **Recursive Physical Agent** — is an open framework for
building embodied agents that continuously evolve through recursive
interaction with the physical world. Rather than prescribing a single
foundation model, RPent provides a **recursive agent framework** that
harnesses heterogeneous intelligence — perception, reasoning, memory,
execution, and self-evolution — into a unified physical agent. Through
continuous interaction, reflection, and adaptation, RPent lets physical
agents acquire new capabilities and evolve beyond their initial design.

.. image:: https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png
   :alt: RPent framework diagram
   :align: center
   :width: 90%

Where the name comes from
-------------------------

The name **Pent** is inspired by the **Pentagram**, whose five points
symbolize the integration of multimodal intelligence into a unified
embodied agent:

- **Perception** — seeing, hearing, touching the world.
- **Reasoning** — planning and deciding what to do next.
- **Memory** — retaining what worked and what did not.
- **Execution** — turning intent into motion.
- **Self-evolution** — learning from every interaction.

At the center of the pentagram sits the **infinity symbol (∞)**, which
represents the endless recursive cycle of *perception → reasoning →
execution → self-evolution*, through which intelligence continuously
expands into the physical world. The pentagram plus its infinite core is
the RPent logo, and it is also the shape of the framework: five
first-class subsystems, wired together in a closed loop.

The **R** in *RPent* pins the framework's core commitment: intelligence
is not a one-shot deployment but a *recursive* process. Every episode
feeds back into the memory that shapes the next.

Design principles
-----------------

RPent is built on three principles:

- **Service-oriented.** Each capability — a VLA policy, a planner brain,
  a memory store, a simulator — is deployed as an independent service
  with its own process boundary and its own lifecycle. Heavyweight GPU
  models never share an interpreter with a physics engine.
- **Standardized.** Services talk over unified interfaces (RPC, HTTP,
  MCP) so a new brain or a new arm plugs in without rewriting the runner.
- **Composable.** Any planner can drive any set of primitives against
  any environment. Swap the brain with one flag; swap the arm by
  dropping a package under ``robots/``.

Together these principles let RPent move beyond a traditional robot
control framework and become **Agentic Infrastructure for the Physical
World**, where intelligence is not just deployed but continuously built,
expanded, and evolved.

Supported environments and brains
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Agentic planner
     - Action primitives
     - Simulator
     - Real world
   * - ``api`` — pydantic-ai ✅
     - **VLA manipulation** ✅
     - **LIBERO** ✅
     - **Franka** (in progress)
   * - Anthropic (Claude) ✅
     - Pi0.5 (LIBERO, HTTP) ✅
     - libero_object / _goal / _spatial / _10
     - **SO-101** (in progress)
   * - OpenAI (responses) ✅
     - RLDX-1 (RoboCasa, socket-RPC) ✅
     - **RoboCasa** ✅
     -
   * - OpenAI-compatible (chat) ✅
     -
     - PickPlace, Open/Close, TurnOn/Off …
     -
   * - ``claude_code`` — Claude Agent SDK ✅
     -
     -
     -
   * - ``codex`` — OpenAI Codex SDK ✅
     -
     -
     -

Where to go next
----------------

- New to RPent? Start with :doc:`installation` and then
  :doc:`quickstart` to get a LIBERO task running end-to-end.
- Want to drive a specific robot or swap the planner? Head to
  :doc:`usage/configure_planner`.
- Extending RPent for your own scenarios? See
  :doc:`development/architecture`.
