Action Primitives and Tools
===========================

The planner operates through tool calls. Action primitives perform motions such as grasping, moving an end effector, or opening a gripper. Perception and state tools locate targets, read images, and inspect results.

Choose Action Tools
-------------------

RPent provides two main types of action tools:

- **VLA actions** use a vision-language-action model to generate action sequences, such as LIBERO’s ``pi0_pick``. The model usually runs in a separate service.
- **Scripted actions** execute parameterized motions, such as ``move_to``, ``rotate_wrist``, and ``release``. Names and arguments depend on the environment.

``back_project``, ``segment``, and ``view_env_state`` are perception or state-reading tools; they do not themselves represent robot motion. ``finish`` ends the planner loop. Task success is determined by the environment or the real-robot operator.

Models by Platform
------------------

.. list-table::
   :header-rows: 1

   * - Environment / robot
     - Action model and configuration
   * - :doc:`libero`
     - Pi0.5
   * - :doc:`robocasa`
     - RLDX-1
   * - :doc:`robotwin`
     - LingBot-VLA
   * - :doc:`franka`
     - Pi0.5 / openpi
   * - :doc:`dual_franka`
     - Pi0.5 / openpi

Each platform page provides checkpoint paths, available tools, and task requirements. YAM deployment instructions and SO-101 content are pending; see :doc:`yam` and :doc:`so101`.

Services and Extensions
-----------------------

VLA services expose action prediction through ``predict`` and health checks through ``healthz``. RPent RPC services support HTTP and socket transports; see :doc:`advanced_deployment` for deployment.

To add a tool, define its arguments, execution, and result handling as described in :doc:`../development/add_primitive`. Planners call these tools through the shared toolkit.
