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

定位 Demo
---------

公共模块 ``rpent.tools.localization.localization`` 把图片、深度和相机参数
整理成 ``LocalizationFrame``。``frame.rgb`` 中的一个像素，在 ``depth_m``、
``world_xyz`` 和 ``valid_mask`` 中都有相同的数组位置。深度和世界坐标的
单位是米。第 0 行在图片最上方；OpenCV 相机坐标轴为 x 向右、y 向下、z 向前。

LIBERO 是这套采集接口的首个接入环境。环境服务器在一次独占的 RPC 调用中，
取得所有请求视角的图像、深度和相机参数，期间不会重置或推进仿真。
客户端用 NumPy 计算世界坐标表。RoboCasa 和 RoboTwin 保留原有接入实现。

先按本页说明安装 LIBERO、下载 SAM3 权重，再按
:doc:`../guides/advanced_deployment` 启动环境和 SAM3 服务。
demo 客户端和环境服务器都需要使用本版本的采集接口。
服务就绪后，使用已完成任务初始化的环境。如果环境服务器刚刚启动，
先显式调用一次重置，再运行 demo：

.. code-block:: python

   from contextlib import closing
   from rpent.utils.rpc import make_rpc_client

   with closing(make_rpc_client("http://ENV_HOST:ENV_PORT")) as rpc:
       rpc.call("env.reset", timeout_s=120.0)

然后在源码目录运行下面的命令，替换为实际服务地址：

.. code-block:: bash

   python -m robots.libero.localization_demo \
     --env-endpoint http://ENV_HOST:ENV_PORT \
     --sam3-endpoint http://SAM3_HOST:SAM3_PORT \
     --prompt "black bowl" \
     --output-dir /tmp/libero-localization-demo

每次运行使用一个新的输出目录。demo 拍摄一张 1024x1024 的 ``agentview``
图片，用已有的 ``Sam3Tools`` 分割目标，再计算目标区域内有效像素的世界坐标
中位数。使用 ``--camera robot0_eye_in_hand`` 可切换到腕部相机，
``--height 256 --width 256`` 可降低采集分辨率。
demo 连接已有服务，不需要规划器和动作模型服务；服务由操作者负责启动和关闭。

在输出目录查看 ``camera.png/00.png``、``overlay.png/00.png`` 和
``result.json/00.json``。结果中的 ``world_xyz`` 单位是米，并附有有效像素数。
找不到目标、有效深度不足或掩码与图片尺寸不一致时，会返回 ``world_error``。
目录还会保存米制深度、世界坐标、几何有效掩码、物体掩码、相机元数据及
EnvState 文件清单。只有检测到的掩码与图片尺寸一致时，才会保存叠加图。

如果只需要采集，不调用 SAM3，可以连接已有服务。这样不会触发环境客户端
默认执行的重置操作：

.. code-block:: python

   from contextlib import closing
   from rpent.tools.localization.localization import RpcFrameSource, capture
   from rpent.utils.rpc import make_rpc_client

   with closing(make_rpc_client("http://ENV_HOST:ENV_PORT")) as rpc:
       frame = capture(RpcFrameSource(rpc), "agentview", height=256, width=256)
   point = frame.world_xyz[row, col]  # 仅在 valid_mask[row, col] 为真时有效
   object_points = frame.points(object_mask)  # 掩码必须与 frame.rgb 完全对齐

``row``、``col`` 和 ``object_mask`` 必须来自这次采集的 RGB 图片。
SAM3 的点提示使用 ``[row, col]``；``project_to_image`` 把世界坐标投影回图片时，
返回 ``(u, v) = (col, row)``。如果图片经过缩放或翻转，需要先把像素或掩码
转换回原图坐标，才能查询对应的世界坐标。

已记录数据的兼容性
~~~~~~~~~~~~~~~~~~

LIBERO 保留原有图片、深度、世界坐标表和元数据的文件名。
记录状态时，会为两个相机分别采集观测分辨率和 1024x1024 的标准帧。
高分辨率批量采集失败时，会重新采集低分辨率的两个视角。
``segment`` 优先选择完整的高分辨率图片与世界坐标表，缺少时使用低分辨率组合。
批量采集会重新渲染两个低分辨率视角，相比直接复用原始观测会增加渲染开销，
但没有新增依赖。

已有 LIBERO 数据读取脚本需要留意以下变化：

- 无效深度和世界坐标使用 ``NaN``。计算统计值前，使用 ``valid_mask`` 或
  ``np.isfinite(world).all(axis=-1)`` 筛选；新数据不要用坐标是否非零来判断有效性。
- 高分辨率世界坐标从 ``float16`` 改为 ``float32``，压缩前数组的内存占用翻倍。
  去掉固定为 ``float16`` 的假设，处理新文件时保留 ``float32`` 精度。
- 新增 ``agentview_valid_mask.npz``、``wrist_valid_mask.npz``，以及对应的
  ``agentview_valid_mask_high.npz``、``wrist_valid_mask_high.npz``。
  它们表示哪些像素有可用的三维信息，与 SAM3 的物体分割掩码含义不同。
- 元数据顶层的 ``intrinsic_K`` 现在对应保存的顶部为第 0 行的 RGB 图片，
  整数坐标表示像素中心；``T_world_from_cv`` 是标准相机到世界坐标的变换。
  原始标定参数放在 ``native`` 下。读取 ``height``、``width`` 和
  ``pixel_convention``，不要假定固定尺寸，也不要再次翻转或修正主点。
  已有 ``get_camera_meta``、``render_camera`` RPC 仍使用原生格式。

下面演示读取新记录的低分辨率数据，NPZ 中的键仍然是 ``array``：

.. code-block:: python

   import numpy as np

   with np.load(run_dir / "agentview_world.npz" / "00.npz") as archive:
       world = archive["array"]
   with np.load(run_dir / "agentview_valid_mask.npz" / "00.npz") as archive:
       valid = archive["array"]
   points = world[object_mask & valid]
   centre = np.median(points, axis=0) if len(points) else None

高分辨率数据需使用相互对应的 ``_high`` 世界坐标表、图片和有效掩码。
旧记录没有有效掩码文件；读取旧记录时，还需保留原先的非零坐标检查，
排除旧版本用全零表示的无效点。``agentview_policy.png`` 是单独的模型输入，
不能用其中的像素坐标查询上述世界坐标表。

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
