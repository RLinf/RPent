LIBERO
============

.. figure:: https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/images/fig1.png
   :alt: LIBERO 环境概览
   :width: 90%
   :align: center

   LIBERO 基准概览。图片来源：`LIBERO 项目 <https://libero-project.github.io/>`_。

使用 RPent 在 `LIBERO <https://libero-project.github.io/>`_ 中运行桌面操作任务，并复现 LIBERO-PRO 实验。仿真器基于 MuJoCo/robosuite，RPent 支持 ``standard``、``pro`` 和 ``plus`` 三种版本，默认动作模型为 Pi0.5。

.. _libero-overview:

概览
------------

先确认所需模型、任务与运行环境，再按后续步骤安装并运行。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      Pi0.5

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``；另见 :doc:`flash`。

   .. grid-item-card:: 任务

      物体、目标、空间、LIBERO-10

   .. grid-item-card:: 硬件

      Linux、NVIDIA GPU；Python 3.11；CUDA 与 EGL。

.. _libero-pro:

.. _libero-pro-core-suites:

任务
~~~~~~~~~~~~

下表完整列出 RPent 的四个 LIBERO-PRO 核心任务族及其全部扰动套件。

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - 任务族
     - 基础套件
     - 扰动套件
   * - 物体
     - ``libero_object``
     - ``libero_object_task``、``libero_object_swap``、
       ``libero_object_lan``、``libero_object_object``
   * - 目标
     - ``libero_goal``
     - ``libero_goal_task``、``libero_goal_swap``、
       ``libero_goal_lan``、``libero_goal_object``
   * - 空间
     - ``libero_spatial``
     - ``libero_spatial_task``、``libero_spatial_swap``、
       ``libero_spatial_lan``、``libero_spatial_object``
   * - LIBERO-10
     - ``libero_10``
     - ``libero_10_task``、``libero_10_swap``、``libero_10_lan``、
       ``libero_10_object``

.. _libero-observation-action:

观测与动作
~~~~~~~~~~~~

下表区分规划器使用的工具、模型输入及环境的成功判定。

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - 相机 RGB／深度图像及末端、夹爪状态。Pi0.5 使用场景图像、腕部图像和机器人状态。
   * - 动作
     - 规划器调用 Pi0.5 工具，或 ``move_to``、``set_gripper`` 等动作原语，由工具执行环境动作。
   * - 奖励与成功判定
     - 任务成功以最终状态的顶层 ``terminated`` 为准。达到步数上限或规划器调用 ``finish`` 均不代表成功。
   * - 任务指令
     - 任务指令来自所选套件、任务及当前环境。

安装与资源准备
--------------

首次使用 LIBERO-PRO，请先完成 :doc:`../quickstart` 的安装、资源下载和模型配置。本页用于选择更多任务与复现实验。

其他 LIBERO 版本请在独立环境中安装对应 extra，再下载资源；每个 Python 环境只安装一种 LIBERO 版本：

.. list-table::
   :header-rows: 1

   * - 版本
     - 安装命令
     - 资源下载
   * - ``standard``
     - ``uv pip install -e ".[libero]"``
     - ``libero-download-assets --skip-existing``
   * - ``pro``
     - ``uv pip install -e ".[libero-pro]"``
     - ``liberopro-download-assets --skip-existing``
   * - ``plus``
     - ``uv pip install -e ".[libero-plus]"``
     - ``liberoplus-download-assets --skip-existing``

运行时通过 ``--libero-type`` 选择与安装包一致的版本。

VLA 配置
--------

下载推荐的 SFT checkpoint `RLinf-Pi05-LIBERO-130-fullshot-SFT <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT>`_，再将 ``PI05_CHECKPOINT_PATH`` 指向本地 checkpoint 目录：

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
     --local-dir /path/to/rlinf-pi05-libero-130-fullshot-sft

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

SAM3 配置
---------

每次 LIBERO 运行都默认启用 SAM 3.0 分割。从 `Hugging Face: facebook/sam3 <https://huggingface.co/facebook/sam3>`_ 或 `ModelScope: facebook/sam3 <https://modelscope.cn/models/facebook/sam3>`_ 下载 ``sam3.pt``，再通过 ``SAM3_CHECKPOINT_PATH`` 指定本地 checkpoint：

.. code-block:: bash

   # Hugging Face（需要先在模型页面申请访问权限）
   hf auth login
   hf download facebook/sam3 sam3.pt --local-dir /path/to/sam3

   # ModelScope（与上面的 Hugging Face 命令二选一）
   modelscope download --model facebook/sam3 sam3.pt --local_dir /path/to/sam3

   export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt

任务选择
--------

运行 LIBERO 任务时，可通过以下参数选择任务：

.. list-table::
   :header-rows: 1

   * - 参数
     - 说明
   * - ``--suite``
     - 选择任务套件，见 :ref:`LIBERO-PRO 核心套件 <libero-pro-core-suites>`。
   * - ``--task``
     - 套件内的任务索引。
   * - ``--seed``
     - 环境随机种子，默认为 ``0``。
   * - ``--libero-type``
     - 已安装的 LIBERO 版本：``standard`` | ``pro`` | ``plus``。

运行一个任务
------------

完成上面的模型配置，并按 :doc:`configure_planner` 配置规划器。以下命令运行 ``libero_object_swap`` 的任务 2，seed 为 0。

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

   rpent --robot libero \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

如需切换 planner，请参阅 :doc:`configure_planner`。

查看结果
------------

终端会显示服务器启动、规划器对话和工具调用。结束后查看输出目录中的 ``episode.mp4``、``transcript_*.json`` 和 ``run.log``；默认目录与逐步观测文件见 :ref:`run-output-files`。

通过 ``view_env_state(step=-1)`` 查看最终状态，顶层 ``terminated`` 表示环境是否判定成功。规划器调用 ``finish`` 只表示结束规划。实时观察方法见 :doc:`dashboard`。

.. _libero-exploration:

任务记忆与探索模式
------------------

使用 ``--explore`` 生成本地记忆，或通过 ``--memory-profile local`` 读取已有记忆。完整的 LIBERO 命令与操作步骤见 :doc:`memory`。LIBERO 以环境成功判定决定是否发布成功任务记录。

实验复现
--------

RPent 在 LIBERO-PRO Task/Swap 上的统一模型对比及对应配置见 :doc:`../leaderboard`。

:doc:`GPT-6 Astra 套件汇总 <../leaderboard>`
记录全部八个完整套件及 800 个已核验回合：741 成功、59 失败，Overall 92.63%，配置为 Codex / GPT-6 Astra / low / reasoning。

以下保留历史复现记录，实验使用 `reproduce/libero <https://github.com/RLinf/RPent/tree/reproduce/libero>`_ 分支和 ``gpt-5.5`` 模型：

- ``libero_10_task``：70%（70/100）
- ``libero_10_swap``：55%（55/100）

先使用上述复现分支，并完成本页资源配置。下面以任务 0、seed 0 为例；完整成绩需要覆盖该任务集中的全部任务和约定的 seed。示例服务端口需按 :doc:`advanced_deployment` 预先启动。

复现命令如下：

.. code-block:: bash

   rpent --robot libero \
     --suite libero_10_task --task 0 --seed 0 \
     --planner codex \
     --model gpt-5.5 --reasoning-effort xhigh \
     --max-turns 100 \
     --planner-timeout-s 5000 \
     --max-episode-steps 10000 \
     --libero-type pro \
     --vla-endpoint http://127.0.0.1:8220 \
     --sam3-endpoint http://127.0.0.1:8114

进程分工
--------

LIBERO 将仿真和模型推理分别放在独立服务器中运行，工具集通过对应客户端执行规划器的请求。

- **env_server** （``robots/libero/env_server.py``）—— 负责运行 LIBERO 的 MuJoCo 环境并通过 EGL 渲染。它通过 RPC 传输（默认使用 HTTP；添加 ``--transport socket`` 后使用 pickle-framed socket）对外暴露 ``reset``、``step``、``chunk_step``、``render_camera``、 ``get_camera_meta`` 等接口。
- **vla_server** （``rpent/robots/components/pi05_vla_server.py``）—— 持有 Pi0.5 权重，通过同一套 RPC 传输（HTTP 或 socket）暴露 ``predict``。
- **sam3_server** （``rpent/robots/components/sam3_server.py``）—— 加载 SAM 3.0，根据文字描述或目标物体上的一个像素点，分割出目标区域。如果模型给出多个候选结果，只保留评分最高的一个，并通过同一套 RPC 传输（HTTP 或 socket）返回压缩 PNG 格式的分割掩码（mask）。
- **toolkit（工具集）** （``robots/libero/toolkit.py``）—— 定义 LLM 能调用的工具：``pi0_pick`` （交给 Pi0.5）、``move_to``、``rotate_wrist``、 ``back_project``、``view_env_state``、``finish``…

规划器可调用的工具
--------------------

LIBERO 工具分为物理动作工具和只读工具。

**物理动作工具：**

- ``pi0_pick(prompt, ...)`` —— 调用 Pi0.5 执行闭环抓取。
- ``pi0_doubled(prompt, ...)`` —— 调用 Pi0.5 执行非抓取类接触动作。
- ``move_to(xyz, ...)`` —— 将末端执行器移动到世界坐标系中的目标位置。
- ``move_pose(xyz, target_pitch=..., target_yaw=..., ...)`` —— 同时调整末端位置和姿态。
- ``rotate_wrist(target_yaw=... / delta_yaw=..., ...)`` —— 按绝对或相对 yaw 旋转腕部。
- ``rotate_pitch(target_pitch=... / delta_pitch=..., ...)`` —— 按绝对或相对 pitch 倾斜夹爪。
- ``set_gripper(gripper=..., steps=...)`` —— 保持末端姿态，并在指定步数内控制夹爪。
- ``release(...)`` —— 打开夹爪。

物理动作工具执行后会推进环境，并记录新的状态和图像。

**只读工具：**

- ``back_project(row, col, ...)`` —— 将图像像素反投影到世界坐标。
- ``segment(prompt=... / point=..., ...)`` —— 通过 SAM3 对已有图像进行文本或点提示分割。
- ``view_env_state(step=-1)`` —— 读取已记录的状态和内嵌观测图像；第 0 步为初始状态，``-1`` 表示最新状态。
- ``view_camera_meta(camera=..., step=-1)`` —— 读取指定步骤的相机元数据； ``-1`` 表示最新状态。
- ``finish(status, summary)`` —— 结束当前运行。

这些工具不会推进环境。

Dashboard
---------

实时监控、提交任务和停止会话的方法见 :doc:`dashboard`。

接入自定义 VLA
----------------

如果你有一个与 LIBERO 兼容、但并非 Pi0.5 的 VLA，可以在不修改机器人实现的情况下替换 model client：

1. 写一个新的 ``vla_server.py``，暴露相同的 ``predict`` RPC 契约（HTTP 或 socket 均可）。
2. 用 ``--vla-endpoint [protocol://]host:port`` 指向它。
3. 如果可用工具需要调整（比如将 ``pi0_pick`` 改成 ``mymodel_pick``），相应更新 ``robots/libero/toolkit.py``。

完整流程见 :doc:`../development/add_primitive`。
