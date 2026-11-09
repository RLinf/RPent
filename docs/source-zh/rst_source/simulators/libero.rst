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

      Pi0.5；Cosmos Policy、Cosmos3 和 Fast-WAM 支持 standard/Pro 评测。

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``；另见 :doc:`../guides/flash`。

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

首次使用 LIBERO-PRO，请先完成 :doc:`../get_started/quickstart` 的安装、资源下载和模型配置。本页用于选择更多任务与复现实验。

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

以上各组依赖均包含 Pi0.5（``rpent-openpi``）、SAM3 和 RLinf 运行时（``rpent-rlinf``）。模型权重需按下文单独下载。

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

SAM 3.0 是 GeometryAgent 的内部依赖。从 `Hugging Face: facebook/sam3 <https://huggingface.co/facebook/sam3>`_ 或 `ModelScope: facebook/sam3 <https://modelscope.cn/models/facebook/sam3>`_ 下载 ``sam3.pt``，再通过 ``SAM3_CHECKPOINT_PATH`` 指定本地 checkpoint：

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
     - 任务索引从 0 开始。环境会拒绝越界索引和没有初始状态的任务，不会将无效选择替换成另一项任务。
   * - ``--seed``
     - 环境随机种子，默认为 ``0``。
   * - ``--libero-type``
     - 已安装的 LIBERO 版本：``standard`` | ``pro`` | ``plus``。

运行一个任务
------------

完成上面的模型配置，并按 :doc:`../guides/configure_planner` 配置规划器。以下命令运行 ``libero_object_swap`` 的任务 2，seed 为 0。

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

   rpent --robot libero \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

如需切换 planner，请参阅 :doc:`../guides/configure_planner`。

WAM 评测
~~~~~~~~

按 :doc:`../development/add_action_model` 准备 worker，保留上面的仿真器和 SAM3 配置。WAM 不需要 Pi0.5 权重。Cosmos 在端口 8116 启动后：

.. code-block:: bash

   rpent --robot libero --suite libero_spatial --task 0 --seed 0 \
     --libero-type standard --am-backend cosmos-policy \
     --am-endpoint http://127.0.0.1:8116 --memory-profile local \
     --planner api --model <planner-model>

Fast-WAM 使用 ``--am-backend fast-wam`` 及其 worker 地址。LIBERO-Pro 需在 Pro 环境中选择 Pro suite 并指定 ``--libero-type pro``。WAM 不支持 LIBERO-Plus、Flash 和探索。WAM 必须启用本地记忆管理器和文件工具，默认目录按后端区分，语料允许为空；可通过 ``--memory-dir`` 指定兼容的本地语料，目前不支持 HF 记忆。``wam_act(prompt=None, max_chunks=1)`` 替换 Pi0.5 工具，每次执行 1 至 4 个 chunk，每个 chunk 前重新获取观测。每个 chunk 的动作数由 worker 的 ``chunk_size`` 元数据声明。prompt 仅覆盖本次调用，chunk 预算不指定在 chunk 内的哪一步停止。

查看结果
------------

终端会显示服务器启动、规划器对话和工具调用。结束后查看输出目录中的 ``episode.mp4``、``transcript_*.json`` 和 ``run.log``；默认目录与逐步观测文件见 :ref:`run-output-files`。

通过 ``view_env_state(step=-1)`` 查看最终状态，顶层 ``terminated`` 表示环境是否判定成功。规划器调用 ``finish`` 只表示结束规划。实时观察方法见 :doc:`../guides/dashboard`。

.. _libero-exploration:

任务记忆与探索模式
------------------

本节的 HF 记忆与探索流程适用于 VLA；WAM 使用上文说明的本地记忆。

评测只运行一次尝试，读取记忆但不更新。默认的 ``hf`` 模式使用 ``--memory-version auto`` 按模型选择记忆版本。显式版本、离线准备和发布来源见 :doc:`../guides/memory`。

使用 ``--explore`` 生成本地记忆，或通过 ``--memory-profile local`` 读取已有记忆。完整的 LIBERO 命令与操作步骤见 :doc:`../guides/memory`。LIBERO 以环境成功判定决定是否发布成功任务记录。

实验复现
--------

RPent 在 LIBERO-PRO Task/Swap 上的统一模型对比及对应配置见 :doc:`../leaderboard/index`。

:doc:`GPT-6 Astra 套件汇总 <../leaderboard/index>`
记录全部八个完整套件及 800 个已核验回合：741 成功、59 失败，Overall 92.63%，配置为 Codex / GPT-6 Astra / low / reasoning。

Long 结果使用 `reproduce/libero <https://github.com/RLinf/RPent/tree/reproduce/libero>`_ 分支、``gpt-5.5`` 模型和 ``xhigh`` 推理强度：

- ``libero_10_task``：70%（70/100）
- ``libero_10_swap``：55%（55/100）

结合 Spatial/Object/Goal 六项论文成绩（81%、69%、94%、91%、75%、66%）和这两项 Long 复现结果，八项评测的 **Overall 为 75.13%**。:doc:`排行榜 <../leaderboard/performance>` 使用这一汇总口径。

先使用上述复现分支，并完成本页资源配置。下面以任务 0、seed 0 为例；完整成绩需要覆盖该任务集中的全部任务和约定的 seed。示例服务端口需按 :doc:`../guides/advanced_deployment` 预先启动。

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
     --am-endpoint http://127.0.0.1:8220 \
     --sam3-endpoint http://127.0.0.1:8114

进程分工
--------

LIBERO 将仿真和模型推理分别放在独立服务器中运行，工具集通过对应客户端执行规划器的请求。

- **env_server** （``robots/libero/env_server.py``）—— 负责运行 LIBERO 的 MuJoCo 环境并通过 EGL 渲染。它通过 RPC 传输（默认使用 HTTP；添加 ``--transport socket`` 后使用 pickle-framed socket）对外暴露 ``reset``、``step``、``chunk_step``、``render_camera``、 ``get_camera_meta`` 等接口。
- **vla_server** （``rpent/robots/components/pi05_vla_server.py``）—— 持有 Pi0.5 权重，通过同一套 RPC 传输（HTTP 或 socket）暴露 ``predict``。
- **sam3_server** （``rpent/robots/components/sam3_server.py``）—— 加载 SAM 3.0，根据文字描述或目标物体上的一个像素点，分割出目标区域。如果模型给出多个候选结果，只保留评分最高的一个，并通过同一套 RPC 传输（HTTP 或 socket）返回压缩 PNG 格式的分割掩码（mask）。
- **toolkit（工具集）** （``robots/libero/toolkit.py``）—— 定义 LLM 能调用的工具：``pi0_pick`` （交给 Pi0.5）、``move_to``、``rotate_wrist``、``geometry``、``view_env_state``、``finish``…
- **GeometryAgent** （``toolkits/geometry/geometry_agent.py``）—— 通过 ``geometry`` 接收目标和部位描述，由视觉定位模型选择物体上的可见像素，再通过内部的分割与坐标工具计算结果。主规划器没有直接调用 ``segment`` 或 ``back_project`` 的工具入口。

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

- ``geometry(query, camera="agentview", step=-1, measurement="surface_point")`` —— 将目标或部位的定位交给 GeometryAgent。``surface_point`` 返回一个可见材质点；``visible_region`` 返回可见分割点的逐轴中位数及边界。添加 ``required_geometry=["rim_center", "body_center", "bottom_center", "axis", "radius", "height"]`` 可请求按部位清理和拟合。主规划器只传目标描述和 step/camera，无需提供标定参数。
- ``view_env_state(step=-1)`` —— 读取已记录的状态和内嵌观测图像；第 0 步为初始状态，``-1`` 表示最新状态。
- ``view_camera_meta(camera=..., step=-1)`` —— 读取指定步骤的相机元数据； ``-1`` 表示最新状态。
- ``finish(status, summary)`` —— 结束当前运行。

这些工具不会推进环境。

GeometryAgent 每次请求固定一个已记录的观测帧。每个部位调用一次视觉模型、最多两次 SAM；杯子请求最多涉及三个部位，加上平面请求时最多四个。证据保存为 ``geometry_XXX.json``，同时保留掩码、候选点云、清理后点云和拟合内点索引，数组不进入工具返回文本。

杯口和底部边缘使用稳健三维圆拟合，杯身比较圆柱与圆台模型，可见平面使用 RANSAC 和 SVD。清理包括体素采样、目标连通分量和邻域离群点过滤；杯身、杯口可剔除从掩码外观测到的水平背景平面，底部和平面请求会保留目标自身平面。杯身拟合将把手和内壁点视作异常样本；支持不足时拒绝结果。

明确的参考点请求可使用 ``inner_rim_center`` （内口上沿圆心）、``plate_center`` （中央承托面圆心）、``lid_center`` （封口圆盖中心）、``rectangle_center`` （主要矩形顶面中心）及 ``body_segment_center`` （排除瓶颈、瓶嘴的主瓶身段中心）。内口使用已辨认的内缘轮廓，不用洞中心像素的深度；盘面和罐盖的高度来自所选表面。矩形需要四条物理边和角点支撑，前景遮挡、平面上的标签裁剪或图像截断不足以确定完整顶面中心。酒瓶的两个主段端部需要半径过渡或真实端缘支撑，也支持横放；可见点云的轴向范围不能代替主瓶身范围。

拟合结果位于 ``geometry[field]``，包含定义、假设、证据引用和质量信息。``radius`` 指杯口半径，``body_radii`` 仅描述可见截面的半径。杯身中心定义为已观察到的共轴杯口圆心和杯底圆心的中点；完整高度需要两个端部边界，可见轴向跨度不能代替完整高度。仅看到杯身时，轴线正负方向采用约定。低残差不足以证明可靠，覆盖、退化和部位一致性检查仍可能拒绝拟合。算法阈值和残差不是经校准的定位误差，量化世界坐标图的精度限制仍然保留。

内口沿候选保留最大的连通分量，不要求该分量靠近视觉材质点。视觉点仍用于确认 SAM 区域，并参与背景平面分离及圆拟合一致性检查；离群点过滤和圆拟合验收条件继续生效。

``partial`` 表示部分请求字段可用，其余保持未知。无法获得几何结果时，子代理请求新视角，由主规划器执行移动。这些模型尚不支持多视角融合、完整长方体尺寸和内腔几何恢复。

API 和 Codex 规划器默认复用主规划器的模型和服务地址进行视觉定位。可通过 ``--geometry-model provider:model`` 或 ``--geometry-model codex:model`` 指定其他模型；单独使用 ``codex`` 表示读取 Codex 配置中的默认模型。``--geometry-base-url`` 可覆盖继承的 ``--base-url``。Claude Code 规划器需要显式指定 ``--geometry-model``。视觉模型只接收选定的原始 RGB 图像和目标描述，不接收机器人动作工具或物体真值位姿。

固定参考点清单实验
~~~~~~~~~~~~~~~~~~

几何评估程序固定 16 项 Task/Swap 任务、seed 1～5，共 80 张起始帧、160 个 A/B
参考点。请在 LIBERO-Pro 环境中运行，并完成上文的仿真资源、EGL/CUDA 和 SAM3
配置。实验不需要动作模型权重、任务记忆或主规划器；采集只启动 ENV，推理只启动
SAM3 和显式指定的视觉模型。

.. code-block:: bash

   python -m toolkits.geometry.evaluation.experiment validate-manifest
   python -m toolkits.geometry.evaluation.experiment collect \
     --output experiments/geometry/observations
   python -m toolkits.geometry.evaluation.experiment infer \
     --inputs experiments/geometry/observations \
     --geometry-model codex \
     --output experiments/geometry/predictions

每组只使用 step 0 / agentview，目标保持固定。程序不执行操作动作或补视角动作，
只复制允许的传感器观测，独立真值不进入子代理输入。输出目录必须为空或尚不存在。

`几何工具说明 <https://github.com/RLinf/RPent/blob/main/toolkits/geometry/README.md>`_
定义 ``geometry`` 接口、拟合量、质量门槛和证据文件。
`几何评估说明 <https://github.com/RLinf/RPent/blob/main/toolkits/geometry/evaluation/README.md>`_
提供完整任务表、模型配置、少量样本试跑、``--permissive`` 候选估计、独立参考点标注和误差统计。
宽松候选保留 ``strict_accepted``、``estimate_method`` 和 ``evaluation_only``；
返回坐标不代表通过严格拟合检查。定位精度和任务操作成功率需要分别测量。

Dashboard
---------

实时监控、提交任务和停止会话的方法见 :doc:`../guides/dashboard`。

接入自定义 VLA
----------------

如果你有一个与 LIBERO 兼容、但并非 Pi0.5 的 VLA，可以在不修改机器人实现的情况下替换 model client：

1. 写一个新的 ``vla_server.py``，暴露相同的 ``predict`` RPC 契约（HTTP 或 socket 均可）。
2. 用 ``--am-endpoint [protocol://]host:port`` 指向它。
3. 如果可用工具需要调整（比如将 ``pi0_pick`` 改成 ``mymodel_pick``），相应更新 ``robots/libero/toolkit.py``。

完整流程见 :doc:`../development/add_primitive`。
