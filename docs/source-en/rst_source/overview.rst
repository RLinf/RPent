Introduction to RPent
=====================

RPent (Recursive Physical Agent) is a framework for embodied agents. A language model selects tools from task instructions and visual observations, then controls the robot through action models or scripted motions. Results return to the planner to inform its next action.

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rpent_framework.png
   :alt: RPent planning, perception, memory, and execution architecture
   :width: 100%

How a Task Runs
---------------

1. The planner reads the task, observations, and available task experience.
2. It locates targets with visual tools and selects a VLA or scripted action.
3. The environment executes the action and returns state and camera views.
4. The planner continues, adjusts its strategy, or finishes.

Exploration supports repeated attempts and local memory generation. Evaluation reads existing memory and uses the environment’s success criterion. See :doc:`development/architecture` for implementation details.

Leaderboard
-----------

Compare success rates on LIBERO, LIBERO-PRO, RoboCasa365 Target50, and RoboTwin
C2R. Rankings apply to the methods and evaluation coverage shown; see
:doc:`leaderboard` for detailed results, configurations, and sources.

.. image:: https://cdn.jsdelivr.net/gh/RLinf/misc@a6657fc43a6b3874a20ee1695a480090a4737c35/rpent/benchmarks/leaderboard-en-light.png
   :alt: RPent Leaderboard
   :class: only-light
   :width: 100%
   :target: leaderboard.html

.. image:: https://cdn.jsdelivr.net/gh/RLinf/misc@a6657fc43a6b3874a20ee1695a480090a4737c35/rpent/benchmarks/leaderboard-en-dark.png
   :alt: RPent Leaderboard
   :class: only-dark
   :width: 100%
   :target: leaderboard.html

Choose a Platform
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Platform
     - Documentation
   * - :doc:`LIBERO <usage/libero>`
     - Pi0.5, SAM3, and LIBERO / LIBERO-PRO runs and reproduction.
   * - :doc:`RoboCasa365 <usage/robocasa>`
     - RLDX-1, kitchen tasks, and Target50 reproduction.
   * - :doc:`RoboTwin <usage/robotwin>`
     - LingBot-VLA, dual-arm simulation tasks, and C2R reproduction.
   * - :doc:`Single-Arm Franka <usage/franka>`
     - Hardware preparation, calibration, motion checks, and operation.
   * - :doc:`Dual-Arm Franka <usage/dual_franka>`
     - Two-node deployment, operation, and exploration with an operator.
   * - :doc:`YAM <usage/yam>`
     - Task demo available; installation and usage documentation is coming soon.
   * - :doc:`SO-101 <usage/so101>`
     - Coming soon.

Choose ``api``, ``claude_code``, or ``codex`` for online planning. LIBERO also supports :doc:`Flash Mode <usage/flash>` for executing stored plans. See :doc:`usage/configure_planner` for model-service configuration.

Start with :doc:`quickstart`. See :doc:`leaderboard` for reported results and resource costs, and :doc:`awesome_works/harnessvla` for the research background.

DreamZero, Cosmos Policy, and RoboDojo are also listed in the project roadmap; their usage documentation is pending.
