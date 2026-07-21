Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents
================================================================================================

**Resources:** `Paper <https://arxiv.org/abs/2607.08448>`_ | `Project Page
<https://harnessvla.github.io/>`_ | `Code <https://github.com/RLinf/RPent>`_

Overview
--------

Modern vision-language-action (VLA) models perform strongly on standard robot
benchmarks, yet can degrade sharply when instructions, target bindings, or
spatial layouts change. π\ :sub:`RLinf` achieves 95.3% success on standard
LIBERO but drops to 50.0% on LIBERO-Pro under perturbations. Many such failures
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

Harness VLA is RPent's first publication. Without updating the VLA or expanding
the primitive library during deployment, it reaches **82.4%** success on
LIBERO-Pro, **55.4%** on RoboCasa365, and **58.4%** on the RoboTwin 2.0
clean-to-randomized setting.

.. figure:: /_static/images/harnessvla_scheme.png
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

Harness VLA is evaluated across standard and perturbed tabletop manipulation,
household-kitchen long-horizon tasks, and clean-to-randomized bimanual
manipulation. Representative success rates are summarized below.

.. list-table:: Representative Harness VLA results
   :header-rows: 1
   :widths: 24 34 42

   * - Benchmark
     - Evaluation setting
     - Reported success rate
   * - LIBERO
     - Standard task suites
     - Harness VLA: **96.0%**; π\ :sub:`RLinf`: 95.3%
   * - LIBERO-Pro
     - Perturbed tabletop manipulation
     - Harness VLA: **82.4%**; π\ :sub:`RLinf`: 50.0%; RATS: 43.8%; Cap-X: 18.2%
   * - RoboCasa365
     - Household-kitchen manipulation
     - Harness VLA: **55.4%**; RLDX-1: 30.0%
   * - RoboTwin 2.0 C2R
     - Clean-to-randomized bimanual manipulation
     - Harness VLA: **58.4%**; LingBot-VLA: 50.4%

Harness VLA reaches 96.0% success on standard LIBERO, comparable to the 95.3%
of π\ :sub:`RLinf`. On the more challenging LIBERO-Pro benchmark, Harness VLA
reaches 82.4%, outperforming π\ :sub:`RLinf` at 50.0%, RATS at 43.8%, and Cap-X
at 18.2%. On RoboCasa365, Harness VLA raises the task-weighted overall success
rate from 30.0% with RLDX-1 to 55.4%. On RoboTwin 2.0 C2R, Harness VLA reaches
58.4%, outperforming LingBot-VLA at 50.4%. These gains come from three
complementary mechanisms: semantic re-grounding by the planner, sparse and
targeted VLA retries after restaging, and Analytic Primitives that isolate
non-contact execution.

Quick Start
-----------

* **Tutorial:** :doc:`LIBERO <../usage/libero>`

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
