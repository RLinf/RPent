Harness VLA
===========

*Steering Frozen VLAs into Reliable Manipulation Primitives via
Memory-Guided Agents*

**Resources:** `Paper <https://arxiv.org/abs/2607.08448>`_ | `Project Page
<https://harnessvla.github.io/>`_ | `Code <https://github.com/RLinf/RPent>`_

Overview
--------

Modern vision-language-action (VLA) models perform strongly on standard robot
benchmarks, yet can degrade sharply when instructions, target bindings, or
spatial layouts change. π\ :sub:`RLinf` achieves 95.3% success on standard
LIBERO but drops to 50.0% on LIBERO-PRO under perturbations. Many such failures
do not arise because the VLA cannot grasp or place an object. Instead, the
model applies a locally plausible action to the wrong target, from an
unfavorable state, or at the wrong stage of a long-horizon task.

Harness VLA shifts the question from how to train a larger VLA to how an
existing VLA should be organized and invoked. It turns the frozen VLA into a
retryable primitive for contact-rich manipulation, while an Agentic Planner
combines it with a small, fixed library of Analytic Primitives. The planner
re-grounds the task, creates suitable local conditions for the VLA, checks the
physical outcome, and reorganizes execution after a failure. The VLA weights
remain frozen throughout.

Harness VLA is RPent's first publication. In paper v4, the Claude Code planner
reaches **82.4%** success on LIBERO-PRO and **58.4%** on RoboTwin C2R; the Codex
planner reaches **57.1%** on RoboCasa365. The VLA remains frozen during deployment.
See :doc:`../benchmarks` for model configurations, benchmark splits, and sources.

.. figure:: https://github.com/RLinf/misc/raw/main/pic/harnessvla_scheme.png
   :alt: Overview of the Harness VLA framework
   :align: center
   :width: 100%

   Overview of the Harness VLA framework

Framework
---------

Harness VLA organizes the planner, Action Primitives, and two forms of memory
within a unified framework:

* **Agentic Planner.** A coding agent interprets the task and current RGB-D
  observations, rebinds target objects and regions, checks execution feedback,
  and selects, sequences, or retries the available primitives.
* **Action Primitives.** RPent encapsulates different robot capabilities as
  Action Primitives that the planner can invoke. Harness VLA primarily combines
  the following two types:

  * **Analytic Primitives.** A fixed set of Analytic Primitives handles
    non-contact operations such as staging, spatial transport, pose adjustment,
    gripper control, navigation, and release, complementing the contact-rich
    operations handled by the VLA.
  * **VLA Primitive.** Harness VLA turns the frozen VLA into an Action Primitive
    that the Agentic Planner can invoke flexibly. It handles contact-rich
    operations such as irregular-object grasping, constrained placement, button
    pressing, and interaction with articulated mechanisms such as drawers and
    doors. After each execution, the Agentic Planner checks the outcome from the
    latest RGB-D observations and, when needed, adjusts the robot state for a
    more targeted attempt.
* **Memory.** Task-Specific Memory distills validated execution strategies into
  parameterized compositions of Action Primitives, allowing the Agentic Planner
  to rebind targets and spatial parameters from current observations. Global
  Memory captures reusable success patterns, failure modes, and recovery
  strategies across tasks to guide subsequent planning.

During exploration, the Agentic Planner starts from a single seed task instance
to discover an effective division of labor between Analytic Primitives and the
VLA, then distills successful execution strategies and recovery experience into
Task-Specific Memory and Global Memory. At deployment time, the Agentic Planner
combines these memories with live observations to dynamically rebind targets
and spatial parameters, allowing validated strategies to adapt to changes in
target bindings and spatial layouts.

Results
-------

The :doc:`benchmark results page <../benchmarks>` collects the complete Codex
and Claude Code split tables from `paper v4
<https://arxiv.org/html/2607.08448v4#S3.SS3>`_, alongside separately identified
RPent evaluations and baseline references. It covers standard LIBERO,
LIBERO-PRO Task/Swap, RoboCasa365 Atomic/Seen/Unseen, RoboTwin C2R, and the
zero-shot Goal ablation, with the model, reasoning configuration, sample size,
and metric definition for each setting.

The historical RoboCasa365 value of 55.4% belongs to `paper v3
<https://arxiv.org/html/2607.08448v3#S3.T4>`_; the v4 Codex value is 57.1%.
The separate RPent Target50 reproduction reports 57.00%.

Quick Start
-----------

For paper-result reproduction, use the environment-specific branch listed
below. See each tutorial for the exact setup and commands.

* **LIBERO:** :doc:`Tutorial <../usage/libero>` —
  `reproduce/libero <https://github.com/RLinf/RPent/tree/reproduce/libero>`_
* **RoboCasa:** :doc:`Tutorial <../usage/robocasa>` — use ``main``
* **RoboTwin:** :doc:`Tutorial <../usage/robotwin>` —
  `reproduce/robotwin <https://github.com/RLinf/RPent/tree/reproduce/robotwin>`_

Citation
--------

.. code-block:: bibtex

   @article{zhang2026harnessvla,
     title={Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents},
     author={Zhang, Yixian and Zhang, Huanming and Gao, Feng and Li, Xiao and
             Liu, Zhihao and Zhu, Chunyang and Qiu, Jiaxing and Yan, Yuchen and
             Liu, Jiyuan and Tang, Wenhao and Fang, Zhengru and Nie, Yi and
             Wei, Changxu and Wang, Yu and Ding, Wenbo and Yu, Chao},
     journal={arXiv preprint arXiv:2607.08448},
     year={2026},
     url={https://arxiv.org/abs/2607.08448}
   }
