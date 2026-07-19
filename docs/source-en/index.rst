.. _home:

Welcome to RPent
================

.. raw:: html

   <div class="rpent-hero">
     <img class="rpent-hero-architecture"
          src="https://github.com/RLinf/misc/raw/main/pic/rpent_logo.png"
          alt="RPent logo" />
     <p class="rpent-hero-subtitle">
       RPent (Recursive Physical Agent) is an open framework for building
       embodied agents that continuously evolve through recursive interaction
       with the physical world. Rather than prescribing a single foundation
       model, RPent provides a recursive agent framework that harnesses
       heterogeneous intelligence, including perception, reasoning, memory,
       execution, and self-evolution, into a unified physical agent. Through
       continuous interaction, reflection, and adaptation, RPent enables
       physical agents to acquire new capabilities and evolve beyond their
       initial design.
     </p>
   </div>

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Overview
      :link: rst_source/overview
      :link-type: doc
      :text-align: center

      What RPent is, what the pentagram + ∞ logo means, and the
      high-level architecture at a glance.

   .. grid-item-card:: Installation
      :link: rst_source/installation
      :link-type: doc
      :text-align: center

      Install RPent with a simple command, which creates the virtualenv, install
      the LIBERO PRO/PLUS etc.

   .. grid-item-card:: Quick start
      :link: rst_source/quickstart
      :link-type: doc
      :text-align: center

      Choose a planner, configure the agent, and run one LIBERO task
      end-to-end.

   .. grid-item-card:: Usage tutorial
      :link: rst_source/usage/configure_planner
      :link-type: doc
      :text-align: center

      Drive the LIBERO / RoboCasa simulators or a Franka / SO-101 arm,
      switch planners, and pick action primitives.

   .. grid-item-card:: Development tutorial
      :link: rst_source/development/architecture
      :link-type: doc
      :text-align: center

      RPent's implementation-level architecture, plus how to add a new
      robot, a new action primitive, or extend memory.

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:

   Overview <rst_source/overview>
   Installation <rst_source/installation>
   Quick start <rst_source/quickstart>

.. toctree::
   :maxdepth: 1
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Usage tutorial

   Agentic planner <rst_source/usage/configure_planner>
   Action primitives <rst_source/usage/configure_primitives>
   LIBERO <rst_source/usage/libero>
   RoboCasa <rst_source/usage/robocasa>
   Franka <rst_source/usage/franka>
   SO-101 <rst_source/usage/so101>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Development tutorial

   System internals <rst_source/development/architecture>
   Add a new robot <rst_source/development/add_robot>
   Add an action primitive <rst_source/development/add_primitive>
   Memory management <rst_source/development/memory>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :titlesonly:
   :hidden:
   :caption: Awesome works

   HarnessVLA <rst_source/awesome_works/harnessvla>
