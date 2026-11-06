.. _release-v0-1:

RPent v0.1 Release 🎉
========================================

We are releasing v0.1, the first public release of RPent.

RPent (Recursive Physical Agent) is an open-source infrastructure for embodied agents in the
physical world. It connects general-purpose models, VLA models, memory, and robot interfaces to
carry out physical tasks through planning, execution feedback, and experience reuse.

v0.1 supports agent harnesses such as Codex and Claude Code for task planning. It integrates the
LIBERO family, RoboCasa365, RoboTwin, and RoboDojo simulators, as well as single-arm Franka,
dual-arm Franka, and YAM robots, and provides Memory, Flash Mode, an interactive CLI, and a Web
Dashboard. RPent has been evaluated in simulation and on real robots, with leading task success
rates on benchmarks such as LIBERO-PRO among the methods listed on the leaderboard.

Highlights
--------------------

Real-World Demos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Robot task demonstrations**: Dual Franka adapts tableware placement to task goals and performs
  steel-ball pouring, plate wiping, and storage tasks. YAM sorts drinks by brand labels, checks
  grasp status, and moves occluding objects to find a spoon. :doc:`Dual Franka demos
  </rst_source/real_world_demos/franka>` | :doc:`YAM demos </rst_source/real_world_demos/yam>`

Simulation Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Multi-environment evaluation and a public leaderboard**: RPent with GPT-6 Astra achieves success
  rates of 92.63% on LIBERO-PRO (low reasoning) and 59.20% on RoboCasa365 Target50 (high reasoning).
  RPent with GPT-5.5 achieves 62.4% on RoboTwin C2R. The leaderboard includes evaluation configurations, success rates, runtime,
  and token usage. :doc:`Leaderboard </rst_source/leaderboard/index>`

System Design
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Modular architecture**: An open, modular design separates the system into user, intelligence,
  interface, and environment layers.

- **Unified interfaces**: VLA models and analytical primitives are exposed as tools through MCP or
  native tool calling. RPC connects model, robot, and simulator services. :doc:`Core interfaces
  </rst_source/development/interfaces>`

Memory
~~~~~~~~~~~~

- **Accumulating and reusing experience**: Exploration accumulates experience. Evaluation uses
  read-only memory organized into general knowledge, task strategies, and execution records.
  :doc:`Memory </rst_source/development/memory>`

- **Faster execution with task cards**: Flash Mode turns successful workflows into task cards and
  updates action targets using current visual observations, reducing step-by-step planning overhead.
  Across 800 LIBERO-PRO episodes, it achieves a **\ 72.63%** success rate with an average runtime of
  **\ 60.19 seconds**, approximately **\ 6.8× faster** than Codex + GPT-6 Astra (low reasoning).
  It uses the officially released GPT-5.5 exploration memory; the reported scores are the best
  evaluation results from seeds 0–9.
  :doc:`Flash Mode </rst_source/guides/flash>` | :doc:`Evaluation data
  </rst_source/leaderboard/time-token-costs>`

Research
~~~~~~~~~~~~~~~~

- **Harness VLA**: RPent's first research paper describes how task memory combines VLA models with
  analytical primitives to handle task changes, environmental perturbations, and multi-step
  manipulation without updating VLA weights. `Paper <https://arxiv.org/abs/2607.08448>`__ | `Project
  page <https://harnessvla.github.io/>`__

Features
----------------

Real-World Robots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Single-Arm and Dual-Arm Franka**: Arm and gripper control, multi-camera observations, and
  hand-eye calibration. Single-arm Franka can connect to an external VLA grasping service. Dual
  Franka provides grasping, handover, and placement skills, with two-node deployment and
  operator-assisted exploration. Both support replaying human-reviewed Flash task cards, with
  operators confirming the outcomes. :doc:`Single-arm </rst_source/real_world_robots/franka>` |
  :doc:`Dual-arm </rst_source/real_world_robots/dual_franka>`

- **YAM**: Bimanual end-effector motion, wrist rotation, gripper control, multi-camera observations,
  and Pi0.5 action models. Exploration, human feedback, and experience storage are available through
  the terminal or Dashboard. :doc:`Guide </rst_source/real_world_robots/yam>`

Simulators
~~~~~~~~~~~~~~~~~~~~

- **LIBERO and LIBERO-PRO**: Spatial, Object, Goal, and LIBERO-10 task suites, with instruction and
  scene-layout perturbations, exploration, memory-based evaluation, and Flash Mode. :doc:`Guide
  </rst_source/simulators/libero>`

- **RoboCasa365**: Navigation, mobile-base motion, and arm manipulation with PandaOmron, including
  Target50 evaluation configurations. :doc:`Guide </rst_source/simulators/robocasa>`

- **RoboTwin**: Bimanual simulation with ``demo_clean`` and ``demo_randomized`` scenes, multi-camera
  observations, and evaluation configurations for 50 tasks. :doc:`Guide
  </rst_source/simulators/robotwin>`

- **RoboDojo**: An Isaac Sim / IsaacLab simulator with the ARX-X5 dual-arm platform. Installation is
  currently supported. :doc:`Guide </rst_source/simulators/robodojo>`

Models
~~~~~~~~~~~~

- **Planners and models**: Codex, Claude Code, and RPent's built-in planner can connect to models
  such as Claude, GPT, and Qwen, with planner-specific reasoning effort, turn limits, and timeouts.
  In LIBERO, RoboCasa365, and RoboTwin, planners can also output action control values for the
  environment to execute directly. :doc:`Configuration guide </rst_source/guides/configure_planner>`

- **VLA models**: Pi0.5 for LIBERO, RoboDojo, Dual Franka, and YAM; RLDX-1 for RoboCasa365; and
  LingBot-VLA for RoboTwin. :doc:`Pi0.5 </rst_source/simulators/libero>` | :doc:`RLDX-1
  </rst_source/simulators/robocasa>` | :doc:`LingBot-VLA </rst_source/simulators/robotwin>`

- **Perception models**: SAM3 for object segmentation and Molmo2-8B for visual grounding, combined
  with depth and camera information to locate targets. :doc:`Visual grounding guide
  </rst_source/guides/flash>`

System
~~~~~~~~~~~~

- **Extension interfaces**: Integrate new robots, models, and tools while reusing service
  communication, task execution, and memory management. :doc:`Developer guide
  </rst_source/development/add_robot>`

- **Interactive CLI and Dashboard**: Task interaction and saved run records. The Dashboard displays
  planning, tool calls, camera views, and action timelines, with English and Chinese interfaces.
  :doc:`CLI </rst_source/guides/cli>` | :doc:`Dashboard </rst_source/guides/dashboard>`

- **Service deployment**: Model and environment services can run independently on local or remote
  machines. Consecutive tasks can reuse running model services. :doc:`Deployment guide
  </rst_source/guides/advanced_deployment>`

- **Memory**: Public datasets of task experience and task cards, with local memory validation,
  merging, and index updates, allow agents to reuse existing experience. :doc:`User guide
  </rst_source/guides/memory>` | `Dataset <https://huggingface.co/datasets/RLinf/RPent-memory>`__

Contributors
------------------------

Thank you to all contributors, listed in no particular order.

`@qurakchin <https://github.com/qurakchin>`__ `@QuanluZhang <https://github.com/QuanluZhang>`__
`@zoeyuchao <https://github.com/zoeyuchao>`__ `@wilburx813 <https://github.com/wilburx813>`__
`@Elessar123 <https://github.com/Elessar123>`__ `@Sonorous281 <https://github.com/Sonorous281>`__
`@ye1457 <https://github.com/ye1457>`__ `@silence-breaker <https://github.com/silence-breaker>`__
`@huanming99 <https://github.com/huanming99>`__ `@jx-qiu
<https://github.com/jx-qiu>`__ `@wkzCode <https://github.com/wkzCode>`__ `@6UOOON9
<https://github.com/6UOOON9>`__ `@zhexuanxu <https://github.com/zhexuanxu>`__ `@ucfyao
<https://github.com/ucfyao>`__ `@catr1xLiu <https://github.com/catr1xLiu>`__ `@liushb9
<https://github.com/liushb9>`__ `@andylin-hao
<https://github.com/andylin-hao>`__ `@chen00double00guang-cmyk
<https://github.com/chen00double00guang-cmyk>`__ `@Code-Hit-ai <https://github.com/Code-Hit-ai>`__
`@Brunch-Life <https://github.com/Brunch-Life>`__ `@JamesHuang0319
<https://github.com/JamesHuang0319>`__ `@NewYork424 <https://github.com/NewYork424>`__ `@Travor278
<https://github.com/Travor278>`__ `@mintonight <https://github.com/mintonight>`__ `@Gaojr-2024
<https://github.com/Gaojr-2024>`__ `@Hiforever-h <https://github.com/Hiforever-h>`__ `@scilwb
<https://github.com/scilwb>`__ `@ZhuWeiwen315 <https://github.com/ZhuWeiwen315>`__ `@skyswordx
<https://github.com/skyswordx>`__ `@littleZ05 <https://github.com/littleZ05>`__
`@mvanhorn <https://github.com/mvanhorn>`__ `@akushonkamen <https://github.com/akushonkamen>`__ `@Yang-star-code <https://github.com/Yang-star-code>`__

RPent Benchmark Results
----------------------------------------------

See the :doc:`evaluation documentation </rst_source/leaderboard/performance>` for public benchmark
results and configurations. Release validation results are included at the end of this page.

Getting Started
------------------------------

- **Run your first task**: Start with LIBERO-PRO, install the environment, download Pi0.5 and SAM3
  weights, and configure a planner. :doc:`Installation and quick start
  </rst_source/get_started/quickstart>`

- **Explore and reuse experience**: Generate Memory through exploration or reusable Flash execution
  plans from successful trajectories. :doc:`LIBERO exploration </rst_source/simulators/libero>` |
  :doc:`Flash Mode </rst_source/guides/flash>`

- **Deploy on real robots**: Follow the platform guides to install robot control software, configure
  cameras, complete hand-eye calibration, and obtain and configure the required model weights.
  :doc:`Franka </rst_source/real_world_robots/franka>` | :doc:`Dual Franka
  </rst_source/real_world_robots/dual_franka>` | :doc:`YAM </rst_source/real_world_robots/yam>`

RPent v0.1 Test Results
----------------------------------------------

RPent v0.1 was validated with 180 release-version episodes, 180 sampled paper-reproduction episodes,
and 15 memory-reuse episodes across LIBERO-PRO, RoboCasa365, and RoboTwin, plus three real-world
tasks on Dual Franka and YAM.

One execution of a task with a specified seed counts as one episode.
Simulation tasks use each environment's native success criterion. Failures and timeouts count toward
the total episode count. Success rates are the proportion of successful episodes. Runtime and token
usage for release-version and paper-reproduction tests are per-episode averages; aggregates are
weighted by episode count. These results cover the selected release-validation tasks. See the
Leaderboard for full benchmark results.

1. Release-Version Simulation Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests use RPent ``release/v0.1`` and HF Memory ``release/v0.1``, with read-only memory during
evaluation.

1.1 LIBERO-PRO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Codex + GPT-6 Astra (low) with Pi0.5. Eight Task and Swap suites cover Spatial, Object, Goal, and
LIBERO-10. Each suite includes two selected tasks, each run with seeds 1 through 5, for 80 episodes.

.. list-table::
   :header-rows: 1
   :class: rpent-release-results


   * - Suite and Task
     - Succeeded
     - Episodes
     - Success Rate
     - Time (s)
     - Input Tokens
     - Output Tokens
   * - libero_spatial_task · task 0
     - 5
     - 5
     - 100%
     - 288.54
     - 1,469,403.20
     - 4,355.00
   * - libero_spatial_task · task 4
     - 4
     - 5
     - 80%
     - 348.88
     - 1,587,509.80
     - 4,888.00
   * - libero_spatial_swap · task 0
     - 5
     - 5
     - 100%
     - 212.86
     - 741,139.60
     - 3,241.20
   * - libero_spatial_swap · task 4
     - 3
     - 5
     - 60%
     - 339.52
     - 1,370,942.40
     - 4,389.80
   * - libero_object_task · task 0
     - 5
     - 5
     - 100%
     - 180.36
     - 726,203.80
     - 2,863.00
   * - libero_object_task · task 4
     - 5
     - 5
     - 100%
     - 192.32
     - 929,688.80
     - 3,153.00
   * - libero_object_swap · task 0
     - 5
     - 5
     - 100%
     - 251.54
     - 1,311,042.20
     - 3,873.80
   * - libero_object_swap · task 4
     - 5
     - 5
     - 100%
     - 247.34
     - 1,057,305.80
     - 3,253.80
   * - libero_goal_task · task 0
     - 3
     - 5
     - 60%
     - 410.84
     - 1,657,833.40
     - 4,453.60
   * - libero_goal_task · task 7
     - 5
     - 5
     - 100%
     - 143.88
     - 434,249.00
     - 2,133.20
   * - libero_goal_swap · task 0
     - 5
     - 5
     - 100%
     - 523.70
     - 1,462,251.80
     - 3,985.20
   * - libero_goal_swap · task 7
     - 5
     - 5
     - 100%
     - 213.40
     - 316,889.40
     - 1,931.20
   * - libero_10_task · task 3
     - 5
     - 5
     - 100%
     - 507.38
     - 2,201,277.20
     - 5,133.20
   * - libero_10_task · task 9
     - 3
     - 5
     - 60%
     - 469.48
     - 2,895,074.00
     - 6,348.40
   * - libero_10_swap · task 3
     - 5
     - 5
     - 100%
     - 518.64
     - 3,341,025.40
     - 6,639.00
   * - libero_10_swap · task 9
     - 1
     - 5
     - 20%
     - 715.74
     - 5,376,288.20
     - 9,809.60
   * - **Total**
     - **69**
     - **80**
     - **86.25%**
     - **347.78**
     - **1,679,882.75**
     - **4,403.19**

1.2 RoboCasa365
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Codex + GPT-6 Astra (low) with RLDX-1. Three Atomic tasks use seeds 1 through 10, and four Composite
tasks use seeds 1 through 5, for 50 episodes.

.. list-table::
   :header-rows: 1
   :class: rpent-release-results


   * - Suite and Task
     - Succeeded
     - Episodes
     - Success Rate
     - Time (s)
     - Input Tokens
     - Output Tokens
   * - Atomic · OpenDrawer
     - 5
     - 10
     - 50%
     - 261.97
     - 504,529.00
     - 2,066.10
   * - Atomic · NavigateKitchen
     - 10
     - 10
     - 100%
     - 86.55
     - 154,368.10
     - 854.40
   * - Atomic · PickPlaceCounterToCabinet
     - 8
     - 10
     - 80%
     - 186.62
     - 318,589.00
     - 1,537.20
   * - Composite-Seen · PrepareCoffee
     - 0
     - 5
     - 0%
     - 697.10
     - 1,318,701.80
     - 4,115.00
   * - Composite-Seen · LoadDishwasher
     - 1
     - 5
     - 20%
     - 542.68
     - 898,868.40
     - 2,864.80
   * - Composite-Unseen · RecycleBottlesByType
     - 0
     - 5
     - 0%
     - 785.30
     - 1,926,543.40
     - 4,675.80
   * - Composite-Unseen · GatherTableware
     - 0
     - 5
     - 0%
     - 404.88
     - 695,211.80
     - 2,549.80
   * - **Total**
     - **24**
     - **50**
     - **48.00%**
     - **350.02**
     - **679,429.76**
     - **2,312.08**

1.3 RoboTwin
^^^^^^^^^^^^^^^^^^^^^^^^

Codex + GPT-5.5 (xhigh) with LingBot-VLA. Ten tasks run in ``demo_randomized``, with five validated
scene seeds per task, for 50 episodes.

.. list-table::
   :header-rows: 1
   :class: rpent-release-results


   * - Suite and Task
     - Succeeded
     - Episodes
     - Success Rate
     - Time (s)
     - Input Tokens
     - Output Tokens
   * - adjust_bottle
     - 4
     - 5
     - 80%
     - 624.46
     - 1,158,068.40
     - 6,144.20
   * - beat_block_hammer
     - 4
     - 5
     - 80%
     - 891.18
     - 3,216,841.00
     - 14,468.80
   * - handover_block
     - 2
     - 5
     - 40%
     - 2,369.52
     - 7,202,960.80
     - 29,643.80
   * - place_empty_cup
     - 5
     - 5
     - 100%
     - 431.60
     - 775,905.80
     - 4,554.60
   * - place_shoe
     - 5
     - 5
     - 100%
     - 672.50
     - 1,501,595.00
     - 6,732.80
   * - stack_bowls_three
     - 3
     - 5
     - 60%
     - 1,654.54
     - 3,416,140.80
     - 12,468.00
   * - blocks_ranking_rgb
     - 4
     - 5
     - 80%
     - 1,581.44
     - 4,454,111.00
     - 16,199.20
   * - lift_pot
     - 4
     - 5
     - 80%
     - 609.38
     - 1,751,153.40
     - 6,595.00
   * - dump_bin_bigbin
     - 3
     - 5
     - 60%
     - 1,161.22
     - 3,479,640.40
     - 13,112.80
   * - open_laptop
     - 4
     - 5
     - 80%
     - 1,693.66
     - 3,892,900.20
     - 15,953.20
   * - **Total**
     - **38**
     - **50**
     - **76.00%**
     - **1,168.95**
     - **3,084,931.68**
     - **12,587.24**

2. Installation, Exploration, and Memory Reuse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LIBERO-PRO, RoboCasa365, and RoboTwin were checked through installation, exploration, memory
creation, and memory reuse after restarting. Five seeds per environment give 15 reuse episodes.
Success rates cover only the reuse stage; exploration runs are recorded separately.

LIBERO-PRO and RoboCasa365 explore with seed 0 and test with seeds 1 through 5. RoboTwin explores in
``demo_clean`` and tests in ``demo_randomized`` with seeds 100000, 100001, 100003, 100004, and
100005. Runtime and token usage in this table refer to the exploration stage.

.. list-table::
   :header-rows: 1
   :class: rpent-release-results rpent-memory-results


   * - Environment and Task
     - Workflow
     - Succeeded
     - Episodes
     - Success Rate
     - Time (s)
     - Input Tokens
     - Output Tokens
   * - LIBERO-PRO · libero_object_swap · task 0
     - Passed
     - 5
     - 5
     - 100%
     - 194.16
     - 1,074,213.00
     - 3,224.60
   * - RoboCasa365 · OpenDrawer
     - Passed
     - 5
     - 5
     - 100%
     - 2,293.80
     - 1,812,551.00
     - 38,699.00
   * - RoboTwin · beat_block_hammer
     - Passed
     - 3
     - 5
     - 60%
     - 4,267.00
     - 5,088,881.00
     - 28,691.00

RoboDojo completed installation and a single execution check of ``put_bottles_into_dustbin``. This
validates installation and the execution workflow; it is not a task-success-rate evaluation.

3. Sampled Paper Reproduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using Harness VLA code and Memory, the reproduction tests cover 80 LIBERO-PRO, 50 RoboCasa365, and
50 RoboTwin episodes, totaling 180. Tasks and seed ranges match Section 1. Configurations and
results are listed below.

3.1 LIBERO-PRO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Planning model: Claude Opus 4.7 (xhigh). VLA: Pi0.5. Sixteen tasks, 80 episodes. :doc:`Environment
setup and run guide </rst_source/simulators/libero>`

.. list-table::
   :header-rows: 1
   :class: rpent-release-results


   * - Suite and Task
     - Succeeded
     - Episodes
     - Success Rate
     - Time (s)
     - Input Tokens
     - Output Tokens
   * - libero_spatial_task · task 0
     - 4
     - 5
     - 80%
     - 804.10
     - 3,189,007.00
     - 31,817.00
   * - libero_spatial_task · task 4
     - 5
     - 5
     - 100%
     - 464.60
     - 1,707,359.00
     - 16,834.00
   * - libero_spatial_swap · task 0
     - 4
     - 5
     - 80%
     - 497.50
     - 1,877,124.00
     - 14,657.00
   * - libero_spatial_swap · task 4
     - 4
     - 5
     - 80%
     - 534.30
     - 1,629,280.00
     - 18,174.00
   * - libero_object_task · task 0
     - 3
     - 5
     - 60%
     - 721.10
     - 2,623,604.00
     - 18,532.00
   * - libero_object_task · task 4
     - 4
     - 5
     - 80%
     - 456.40
     - 1,706,284.00
     - 10,606.00
   * - libero_object_swap · task 0
     - 4
     - 5
     - 80%
     - 485.40
     - 1,565,337.00
     - 12,344.00
   * - libero_object_swap · task 4
     - 5
     - 5
     - 100%
     - 282.40
     - 913,677.00
     - 10,398.00
   * - libero_goal_task · task 0
     - 1
     - 5
     - 20%
     - 314.30
     - 730,283.00
     - 12,251.00
   * - libero_goal_task · task 7
     - 5
     - 5
     - 100%
     - 119.00
     - 362,758.00
     - 4,226.00
   * - libero_goal_swap · task 0
     - 0
     - 5
     - 0%
     - 1,066.90
     - 3,140,629.00
     - 44,427.00
   * - libero_goal_swap · task 7
     - 5
     - 5
     - 100%
     - 86.50
     - 318,485.00
     - 2,915.00
   * - libero_10_task · task 3
     - 1
     - 5
     - 20%
     - 1,207.30
     - 3,659,185.00
     - 31,154.00
   * - libero_10_task · task 9
     - 0
     - 5
     - 0%
     - 841.10
     - 1,797,568.00
     - 26,700.00
   * - libero_10_swap · task 3
     - 0
     - 5
     - 0%
     - 904.80
     - 2,799,185.00
     - 25,717.00
   * - libero_10_swap · task 9
     - 0
     - 5
     - 0%
     - 376.10
     - 896,539.00
     - 14,635.00
   * - **Total**
     - **45**
     - **80**
     - **56.25%**
     - **572.60**
     - **1,807,269.00**
     - **18,462.00**

3.2 RoboCasa365
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Planner: Codex + GPT-5.5 (xhigh). VLA: RLDX-1. Seven tasks, 50 episodes. :ref:`Environment setup and
run guide <reproduce-target50>`

.. list-table::
   :header-rows: 1
   :class: rpent-release-results


   * - Suite and Task
     - Succeeded
     - Episodes
     - Success Rate
     - Time (s)
     - Input Tokens
     - Output Tokens
   * - OpenDrawer
     - 8
     - 10
     - 80%
     - 339.40
     - 1,360,316.00
     - 10,583.00
   * - NavigateKitchen
     - 10
     - 10
     - 100%
     - 288.00
     - 1,286,723.00
     - 13,702.00
   * - PickPlaceCounterToCabinet
     - 8
     - 10
     - 80%
     - 317.20
     - 1,626,763.00
     - 10,042.00
   * - PrepareCoffee
     - 0
     - 5
     - 0%
     - 2,435.90
     - 9,453,302.00
     - 49,694.00
   * - LoadDishwasher
     - 2
     - 5
     - 40%
     - 1,744.40
     - 5,114,983.00
     - 32,871.00
   * - RecycleBottlesByType
     - 0
     - 5
     - 0%
     - 2,332.60
     - 8,435,321.00
     - 47,109.00
   * - GatherTableware
     - 0
     - 5
     - 0%
     - 1,760.80
     - 7,701,994.00
     - 51,364.00
   * - **Total**
     - **28**
     - **50**
     - **56.00%**
     - **1,016.29**
     - **3,925,320.40**
     - **24,969.20**

3.3 RoboTwin
^^^^^^^^^^^^^^^^^^^^^^^^

Planner: Codex + GPT-5.5 (xhigh). VLA: LingBot-VLA. Ten tasks, 50 episodes, in ``demo_randomized``.
:doc:`Environment setup and run guide </rst_source/simulators/robotwin>`

.. list-table::
   :header-rows: 1
   :class: rpent-release-results


   * - Suite and Task
     - Succeeded
     - Episodes
     - Success Rate
     - Time (s)
     - Input Tokens
     - Output Tokens
   * - adjust_bottle
     - 5
     - 5
     - 100%
     - 272.40
     - 303,276.00
     - 1,861.00
   * - beat_block_hammer
     - 2
     - 5
     - 40%
     - 753.10
     - 4,075,393.00
     - 21,435.00
   * - handover_block
     - 0
     - 5
     - 0%
     - 2,317.00
     - 11,461,447.00
     - 45,969.00
   * - place_empty_cup
     - 5
     - 5
     - 100%
     - 343.30
     - 773,897.00
     - 5,453.00
   * - place_shoe
     - 4
     - 5
     - 80%
     - 434.90
     - 1,399,450.00
     - 8,048.00
   * - stack_bowls_three
     - 2
     - 5
     - 40%
     - 738.50
     - 2,084,738.00
     - 12,753.00
   * - blocks_ranking_rgb
     - 3
     - 5
     - 60%
     - 1,484.70
     - 8,932,639.00
     - 37,453.00
   * - lift_pot
     - 2
     - 5
     - 40%
     - 690.40
     - 3,016,443.00
     - 14,398.00
   * - dump_bin_bigbin
     - 3
     - 5
     - 60%
     - 651.90
     - 2,620,781.00
     - 11,887.00
   * - open_laptop
     - 5
     - 5
     - 100%
     - 349.70
     - 845,135.00
     - 5,620.00
   * - **Total**
     - **31**
     - **50**
     - **62.00%**
     - **803.60**
     - **3,551,320.00**
     - **16,488.00**

Code uses the ``reproduce/libero``, ``reproduce/robocasa``, and ``reproduce/robotwin`` branches.
Memory uses the Hugging Face dataset's ``reproduce/memory`` branch.

4. Real-World Demo Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dual Franka and YAM completed three tasks: tableware sorting, pouring, and drink-label recognition
and sorting. The tableware-sorting task uses one bowl and two plates. Runtime and token usage below
refer to a single successful execution of each task.

.. list-table::
   :header-rows: 1
   :class: rpent-release-results rpent-demo-results


   * - Platform and Task
     - Planner Configuration
     - Result
     - Time (s)
     - Input Tokens
     - Output Tokens
   * - Dual Franka · demo01: Tableware Sorting
     - Codex + GPT-6 Astra (xhigh)
     - Completed
     - 600.00
     - 2,996,944
     - 7,503
   * - Dual Franka · demo03: From Placement to Pouring
     - GPT-6 Astra (low)
     - Completed
     - 1,984.00
     - 14,163,904
     - 24,159
   * - YAM · demo02: Drink Label Recognition and Sorting
     - GPT-6 Sol (medium)
     - Completed
     - 1,886.85
     - 12,026,547
     - 38,909

See the :doc:`Leaderboard </rst_source/leaderboard/index>` for full benchmark results and
configurations.
