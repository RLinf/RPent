.. _home:

Welcome to RPent
================

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rpent_logo.png
   :alt: RPent
   :class: rpent-home-logo

RPent (Recursive Physical Agent) is an open-source framework for building embodied agents that continually evolve through recursive interaction with the physical world. It supports different foundation models and integrates perception, reasoning, memory, execution, and self-evolution in a unified agent framework. Through ongoing interaction, agents reflect on and adjust their behavior, accumulate experience, and develop new capabilities beyond their initial design.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Quick Start
      :link: rst_source/get_started/quickstart
      :link-type: doc

      Install RPent, run a LIBERO-PRO task, and inspect the result.

   .. grid-item-card:: Introduction to RPent
      :link: rst_source/get_started/overview
      :link-type: doc

      Learn how planning, perception, actions, and memory work together.

   .. grid-item-card:: Memory and Exploration
      :link: rst_source/guides/memory
      :link-type: doc

      Use published experience or build memory through exploration.

   .. grid-item-card:: Leaderboard
      :link: rst_source/leaderboard/index
      :link-type: doc

      Compare reported success rates, runtime, and token costs.

   .. grid-item-card:: Real-World Demos
      :link: rst_source/real_world_demos/franka
      :link-type: doc

      Watch dual-arm Franka tasks and follow the deployment guide.

   .. grid-item-card:: Extend RPent
      :link: rst_source/development/architecture
      :link-type: doc

      Understand the execution flow and add robots, tools, or planners.

Choose an environment: :doc:`LIBERO <rst_source/simulators/libero>`, :doc:`RoboCasa365 <rst_source/simulators/robocasa>`, or :doc:`RoboTwin <rst_source/simulators/robotwin>`. For robot deployment, see :doc:`Single-Arm Franka <rst_source/real_world_robots/franka>` and :doc:`Dual-Arm Franka <rst_source/real_world_robots/dual_franka>`; :doc:`YAM <rst_source/real_world_demos/yam>` has a task demo.

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Get Started

   Introduction to RPent <rst_source/get_started/overview>
   Quick Start <rst_source/get_started/quickstart>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Leaderboard

   Performance <rst_source/leaderboard/performance>
   Time & Token Costs <rst_source/leaderboard/time-token-costs>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Real-World Demos

   Dual-Arm Franka <rst_source/real_world_demos/franka>
   YAM <rst_source/real_world_demos/yam>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Guides

   Memory and Exploration <rst_source/guides/memory>
   Action Primitives and Tools <rst_source/guides/configure_primitives>
   Planner Configuration <rst_source/guides/configure_planner>
   Command-Line Reference <rst_source/guides/cli>
   Interactive Usage <rst_source/guides/dashboard>
   Flash Mode <rst_source/guides/flash>
   Trajectory Collection and Data Flywheel <rst_source/guides/flywheel>
   Remote Services and Parallel Runs <rst_source/guides/advanced_deployment>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Simulators

   LIBERO <rst_source/simulators/libero>
   RoboCasa365 <rst_source/simulators/robocasa>
   RoboTwin <rst_source/simulators/robotwin>
   RoboDojo <rst_source/simulators/robodojo>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Real-World Robots

   Single-Arm Franka <rst_source/real_world_robots/franka>
   Dual-Arm Franka <rst_source/real_world_robots/dual_franka>
   YAM <rst_source/real_world_robots/yam>
   SO-101 <rst_source/real_world_robots/so101>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Concepts & Development

   Architecture and Execution <rst_source/development/architecture>
   Core Interfaces <rst_source/development/interfaces>
   Memory Design <rst_source/development/memory>
   Add a Robot or Simulator <rst_source/development/add_robot>
   Add an Action Primitive <rst_source/development/add_primitive>
   Add a Planner <rst_source/development/add_planner>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Resources

   Harness VLA <rst_source/resources/harnessvla>
   Contributing <rst_source/resources/contributing>
   Release Notes <rst_source/resources/release_notes>
