.. _home:

Welcome to RPent
================

.. raw:: html

   <div class="rpent-hero">
     <h1 class="rpent-hero-title">Welcome to RPent</h1>
     <p class="rpent-hero-subtitle">
       Agentic Infrastructure for the Physical World —
       an open framework for building embodied agents that continuously
       evolve through recursive interaction with their environment.
     </p>
     <img class="rpent-hero-architecture" src="architecture.svg" alt="RPent architecture" />
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

      Clone RLinf + RPent side by side, create the virtualenv, install
      the LIBERO PRO/PLUS bits.

   .. grid-item-card:: Quick start
      :link: rst_source/quickstart
      :link-type: doc
      :text-align: center

      Set your keys, point at a checkpoint, and run one LIBERO task
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

   Configure planner <rst_source/usage/configure_planner>
   Configure action primitives <rst_source/usage/configure_primitives>
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

   RPent architecture <rst_source/development/architecture>
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
