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
      :link: rst_source/quickstart
      :link-type: doc

      Install RPent, run a LIBERO-PRO task, and inspect the result.

   .. grid-item-card:: Introduction to RPent
      :link: rst_source/overview
      :link-type: doc

      Learn how planning, perception, actions, and memory work together.

   .. grid-item-card:: Memory and Exploration
      :link: rst_source/usage/memory
      :link-type: doc

      Use published experience or build memory through exploration.

   .. grid-item-card:: Leaderboard
      :link: rst_source/leaderboard
      :link-type: doc

      Compare reported success rates, runtime, and token costs.

   .. grid-item-card:: Real-World Demos
      :link: rst_source/usage/real_world_demos_franka
      :link-type: doc

      Watch dual-arm Franka tasks and follow the deployment guide.

   .. grid-item-card:: Extend RPent
      :link: rst_source/development/architecture
      :link-type: doc

      Understand the execution flow and add robots, tools, or planners.

Choose an environment: :doc:`LIBERO <rst_source/usage/libero>`, :doc:`RoboCasa365 <rst_source/usage/robocasa>`, or :doc:`RoboTwin <rst_source/usage/robotwin>`. For robot deployment, see :doc:`Single-Arm Franka <rst_source/usage/franka>` and :doc:`Dual-Arm Franka <rst_source/usage/dual_franka>`; :doc:`YAM <rst_source/usage/real_world_demos_yam>` has a task demo.

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Get Started

   Introduction to RPent <rst_source/overview>
   Quick Start <rst_source/quickstart>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:

   Leaderboard <rst_source/leaderboard>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Real-World Demos

   Dual-Arm Franka <rst_source/usage/real_world_demos_franka>
   YAM <rst_source/usage/real_world_demos_yam>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Guides

   Memory and Exploration <rst_source/usage/memory>
   Action Primitives and Tools <rst_source/usage/configure_primitives>
   Planners and Model Services <rst_source/usage/configure_planner>
   CLI and Configuration <rst_source/usage/cli>
   Dashboard <rst_source/usage/dashboard>
   Flash Mode <rst_source/usage/flash>
   Trajectory Collection and Data Flywheel <rst_source/usage/flywheel>
   Remote Services and Parallel Runs <rst_source/usage/advanced_deployment>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Simulators

   LIBERO <rst_source/usage/libero>
   RoboCasa365 <rst_source/usage/robocasa>
   RoboTwin <rst_source/usage/robotwin>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Real-World Robots

   Single-Arm Franka <rst_source/usage/franka>
   Dual-Arm Franka <rst_source/usage/dual_franka>
   YAM <rst_source/usage/yam>
   SO-101 <rst_source/usage/so101>

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

   Harness VLA <rst_source/awesome_works/harnessvla>
   Contributing <rst_source/resources/contributing>
   Release Notes <rst_source/resources/release_notes>
