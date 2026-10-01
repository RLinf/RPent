LIBERO
======

`LIBERO <https://libero-project.github.io/>`_ 是 RPent 主要使用的仿真基准，\
包含一系列基于 MuJoCo/robosuite 的桌面操作任务。RPent 主要使用四个核心基础任务\
族（``libero_object``、``libero_goal``、``libero_spatial``、
``libero_10``）和三个变体（``standard``、``pro``、``plus``）。默认 VLA
是 **Pi0.5**，由 ``rpent/robots/components/pi05_vla_server.py`` 通过 HTTP 提\
供服务。

VLA 配置
--------

下载推荐的 SFT checkpoint
`RLinf-Pi05-LIBERO-130-fullshot-SFT
<https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT>`_，\
再将 ``PI05_CHECKPOINT_PATH`` 指向本地 checkpoint 目录：

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
     --local-dir /path/to/rlinf-pi05-libero-130-fullshot-sft

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

SAM3 配置
---------

除 ``--planner onejev`` 外，每次 LIBERO 运行都默认启用 SAM 3.0 分割。从
`Hugging Face: facebook/sam3 <https://huggingface.co/facebook/sam3>`_ 或
`ModelScope: facebook/sam3 <https://modelscope.cn/models/facebook/sam3>`_
下载 ``sam3.pt``，再通过 ``SAM3_CHECKPOINT_PATH`` 指定本地 checkpoint：

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

- ``--suite`` —— 选择要运行的任务套件。完整核心套件列表见
  :ref:`libero-pro-core-suites`。
- ``--task`` —— 套件内的任务索引。
- ``--seed`` —— 环境种子。
- ``--libero-type`` —— LIBERO 变体：``standard`` | ``pro`` |
  ``plus``。

.. _libero-pro-core-suites:

LIBERO-PRO 核心套件一览
~~~~~~~~~~~~~~~~~~~~~~~

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

最小命令
--------

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

   rpent --robot libero \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

如需切换 planner，请参阅 :doc:`configure_planner`。

.. _libero-exploration:

探索模式与本地 Memory 评测
--------------------------

RPent 支持两种 LIBERO 运行模式：

- **Exploration** 使用可重置的多次尝试和相互独立的 planner session
  探索成功策略，并将其提炼为本地 global/task-family/task-specific 三层
  memory corpus。它是
  memory 生成流程，不用于统计 benchmark success rate。
- **Evaluation** 是默认的单次评测模式，不会 reset episode，也不会更新
  memory。使用本地 memory 的 evaluation 会读取 exploration 生成并通过校验的
  audit、recipe 和经验。HarnessVLA 的 success rate 在 evaluation mode 下复现。

默认仍为单次评测模式。省略 ``--memory-profile`` 时使用 Hugging Face memory，
``--memory-version auto`` 按模型选择版本。手动覆盖、离线下载和发布来源详见
:ref:`Memory 管理 <memory-management>`。两种 profile 都执行相同的单次评测流程；\
区别仅在于评测 memory 的来源及所使用的 memory prompt。本地 memory 已准备好后\
（例如先执行下文的 exploration 流程），即可使用 ``local``。\
该选项不会开启 exploration，也不会从 Hugging Face 下载
memory；它只会针对 ``--memory-dir`` 执行普通的单次评测，并避免同步覆盖本地\
目录：

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner codex --memory-profile local \
     --memory-dir /path/to/libero-memory

探索模式沿用同一个 Python/CLI 入口。它支持可重置的多次尝试和独立
planner session，并在正常结束后校验、合并 memory，只有 LIBERO 确认成功时才发布
task audit/recipe。探索可以从空的 ``--memory-dir`` 开始，并始终使用
local profile；真正开启该流程的是 ``--explore``：

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 0 \
     --planner api --model anthropic:claude-opus-4-8 \
     --explore --explore-sessions 3 --explore-attempts-per-session 5 \
     --memory-dir /path/to/libero-memory

每个 planner session 使用一个新建的 toolkit，其状态轨迹和观测工件保存在
``<output-dir>/sessions/session_NNN/``，供最终 memory distillation 使用；同一
session 内通过 reset 发起的多次 attempt 仍复用该 toolkit。

在 exploration 命令中增加 ``--dashboard``，即可跨 planner session 查看完整推\
理过程、相机画面和连续动作时间线。

使用 ``--no-auto-merge-memory`` 可保留 inbox，稍后人工审核。也可直接使用
memory 维护命令：

.. code-block:: bash

   rpent-memory --memory-dir /path/to/libero-memory validate
   rpent-memory --memory-dir /path/to/libero-memory build-index
   rpent-memory --memory-dir /path/to/libero-memory merge \
     --cell 10_task_t0_s0 --output-dir logs/explore_10_task_t0_s0

运行时生成的 memory 数据不应提交到仓库。

进程分工
--------

- **env_server** （``robots/libero/env_server.py``）—— 负责运行 LIBERO
  的 MuJoCo 环境并通过 EGL 渲染。它通过 RPC 传输（默认使用 HTTP；添加
  ``--transport socket`` 后使用 pickle-framed socket）对外暴露
  ``reset``、``step``、``chunk_step``、``render_camera``、
  ``get_camera_meta`` 等接口。
- **vla_server** （``rpent/robots/components/pi05_vla_server.py``）—— 持有 Pi0.5
  权重，通过同一套 RPC 传输（HTTP 或 socket）暴露 ``predict``。
- **sam3_server** （``rpent/robots/components/sam3_server.py``）—— 持有 SAM
  3.0，通过同一套 RPC 传输（HTTP 或 socket）支持文本或单个正点分割，\
  仅返回排名第一的压缩 PNG mask。
- **toolkit（工具集）** （``robots/libero/toolkit.py``）—— 定义 LLM
  能调用的工具：``pi0_pick`` （交给 Pi0.5）、``move_to``、``rotate_wrist``、
  ``back_project``、``view_env_state``、``finish``…

Planner 能调用的工具
--------------------

LIBERO 工具分为物理动作工具和只读工具。

**物理动作工具：**

- ``pi0_pick(prompt, ...)`` —— 调用 Pi0.5 执行闭环抓取。
- ``pi0_doubled(prompt, ...)`` —— 调用 Pi0.5 执行非抓取类接触动作。
- ``move_to(xyz, ...)`` —— 将末端执行器移动到世界坐标系中的目标位置。
- ``move_pose(xyz, target_pitch=..., target_yaw=..., ...)`` —— 同时调整末端位置\
  和姿态。
- ``rotate_wrist(target_yaw=... / delta_yaw=..., ...)`` —— 按绝对或相对
  yaw 旋转腕部。
- ``rotate_pitch(target_pitch=... / delta_pitch=..., ...)`` —— 按绝对或相对
  pitch 倾斜夹爪。
- ``set_gripper(gripper=..., steps=...)`` —— 保持末端姿态，并在指定步数内控\
  制夹爪。
- ``release(...)`` —— 打开夹爪。

物理动作工具执行后会推进环境，并记录新的状态和图像。

**只读工具：**

- ``back_project(row, col, ...)`` —— 将图像像素反投影到世界坐标。
- ``segment(prompt=... / point=..., ...)`` —— 通过 SAM3 对已有图像进行文本或点\
  提示分割。
- ``view_env_state(step=-1)`` —— 读取已记录的状态和内嵌观测图像；第 0 步为初\
  始状态，``-1`` 表示最新状态。
- ``view_camera_meta(camera=..., step=-1)`` —— 读取指定步骤的相机元数据；
  ``-1`` 表示最新状态。
- ``finish(status, summary)`` —— 结束当前运行。

这些工具不会推进环境。

Dashboard
---------

加上 ``--dashboard`` 可启动长生命周期的本地 Dashboard Session。系统会自动选择一\
个空闲端口，并在终端输出访问 URL：

.. code-block:: bash

   rpent --robot libero --dashboard \
     --planner claude_code --model claude-opus-4-8

Session 配置全部来自命令行，打开地址后会直接进入实时监控。共享服务就绪后，\
输入以下命令启动 TaskRun：

.. code-block:: text

   /rpent-task libero_object_swap 2 0

Dashboard 支持 ``api``、``claude_code`` 和 ``codex`` planner。在命令行传递
``--planner`` 与 ``--model``，配置方式和普通运行一致，详见
:doc:`configure_planner`。

每个 TaskRun 使用独立环境，VLA 和 SAM3 服务由 Session 复用。可通过新的
``/rpent-task`` 启动或切换任务；运行中也可以发送消息引导智能体，并按 Esc
请求中断。在终端按 Ctrl+C 可结束 Session。

``--dashboard`` 不能与 ``--interactive`` 或 ``--env-endpoint`` 同时使用；外部
``--vla-endpoint`` 和 ``--sam3-endpoint`` 服务仍然可用。使用
``--dashboard-language zh-cn`` 可切换中文 UI。

接入自定义 VLA
----------------

如果你有一个与 LIBERO 兼容、但并非 Pi0.5 的 VLA，可以在不修改机器人实现的情况下\
替换 model client：

1. 写一个新的 ``vla_server.py``，暴露相同的 ``predict`` RPC 契约（HTTP 或 socket
   均可）。
2. 用 ``--vla-endpoint [protocol://]host:port`` 指向它。
3. 如果可用工具需要调整（比如将 ``pi0_pick`` 改成 ``mymodel_pick``），\
   相应更新 ``robots/libero/toolkit.py``。

完整流程见 :doc:`../development/add_primitive`。

结果复现
--------

RPent 在 LIBERO-PRO Task/Swap 上的统一模型对比及对应配置见
:doc:`../leaderboard`。

:doc:`GPT-6 Astra 套件汇总 <../leaderboard>`
记录全部八个完整套件及 800 个已核验回合：741 成功、59 失败，Overall 92.63%，\
配置为 Codex / GPT-6 Astra / low / reasoning。

Long 复现结果使用
`reproduce/libero <https://github.com/RLinf/RPent/tree/reproduce/libero>`_
分支、``gpt-5.5`` 模型及 ``xhigh`` reasoning effort：

- ``libero_10_task``：70%（70/100）
- ``libero_10_swap``：55%（55/100）

其余六项 Spatial/Object/Goal 成绩保持 81%、69%、94%、91%、75%、66%。\
将这六项论文结果与上述两项 Long 复现结果合并后，八套件 **Overall 为 75.13%**，\
与 :doc:`排行榜 <../leaderboard/performance>` 使用的更新汇总一致。

复现命令如下：

.. code-block:: bash

   rpent --robot libero \
     --suite libero_10_task --task "task" --seed "seed" \
     --planner codex \
     --model gpt-5.5 \
     --max-turns 100 \
     --planner-timeout-s 5000 \
     --max-episode-steps 10000 \
     --libero-type pro \
     --vla-endpoint http://127.0.0.1:8220 \
     --sam3-endpoint http://127.0.0.1:8114

OneJev 决策后端
-----------------

``--planner onejev`` 调用外部 OneJev-9B System One 服务，对 LIBERO 适配器已确定参数的动作评分。RPent 通过现有 Toolkit 执行最高分候选。Pi0.5 负责抓取和抽屉接触操作，scripted primitive 负责搬运、下降和松爪。

在独立于 RPent 的工作目录安装 OneJev。以下路径均相对于该目录：

.. code-block:: bash

   git clone https://github.com/OmniJev/OneJev.git
   uv venv .venv-onejev --python 3.11
   uv pip install --python .venv-onejev/bin/python --torch-backend auto \
     ./OneJev \
     'transformers>=5.17.0,<6' pillow torchvision
   hf download OmniJev/OneJev-9B --local-dir ./OneJev-9B

先在这个目录启动 OneJev 服务，并保持终端运行：

.. code-block:: bash

   ./.venv-onejev/bin/qev serve \
     --model ./OneJev-9B --name OneJev-9B \
     --host 127.0.0.1 --port 8008 \
     --device cuda:0 --dtype bfloat16 --multimodal \
     --no-cuda-graphs --no-gpu-preprocess

服务就绪后，在另一个终端激活 RPent 环境，进入 RPent 仓库根目录，运行：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH="data/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT/"
   export LIBERO_TYPE=pro
   rpent --robot libero --suite libero_object_swap --task 2 --seed 0 \
     --planner onejev --model OneJev-9B --base-url http://127.0.0.1:8008

该模式只启动 Env 和 Pi0.5，使用本次运行的产物，不加载 memory 语料，也不启动 SAM3 或 Molmo。OneJev 的依赖保留在独立环境，RPent Client 仅使用 HTTP。Env 服务以 ``--public-observations-only`` 启动，只返回 RGB-D、机器人状态、相机标定和公开 episode 终止标志，不返回对象私有状态、分割、接触真值或
reward/info 内容。借用 ``--env-endpoint`` 时，服务必须具有相同的公开观测模式及
suite/task/seed 设置；此模式不暴露完整的 ``env.raw_obs`` 路由。

``robots/libero/onejev/`` 中的适配器支持打开或关闭指定抽屉，也支持将一个或多个对象放进同一个 basket、bowl、plate 或 tray。它把 ``between ... and ...`` 等空间描述保留为单个对象，拒绝未支持的动词和独立的多阶段任务。抽屉任务使用有次数上限的 Pi0.5 接触动作；放置任务从 RGB-D 生成目标区域候选，不能据此确认物体身份。OneJev 根据带标记的 RGB 和任务选择动作。放置任务缺少几何候选时，会在抓取前停止；任务成功仅以 LIBERO 原生终止信号为准。

可使用 ``--onejev-config /path/to/config.json``。其中 ``target_roi`` 是操作者从 RGB 指定的区域，采用归一化的 ``[row_min, col_min, row_max, col_max]`` 坐标。其他配置包括工作空间、抓取与抽屉接触预算、携带余量及 EEF/物体偏移。这些都是明确的控制先验，不来自模拟器对象几何。配置示例：

.. code-block:: json

   {
     "target_roi": [0.2, 0.2, 0.8, 0.8],
     "pick_max_chunks": 24,
     "eef_object_offset": 0.09
   }

示例 ROI 仅用于说明格式，并非某个任务的已标定区域；应先查看保存的公开 RGB 再指定。
每次运行的 decision trace 拆为三个正常 JSON 数组文件，每条记录使用 4 空格缩进：

- ``onejev_state.json``：每次调用实际提供给 OneJev 的 state。
- ``onejev_question.json``：实际 question、instructions 和参数已完整确定的选项。
- ``onejev_decision.json``：服务原始输出、HTTP 状态、延迟、选中的候选 ID、``selected_tool_name``、执行结果及错误。

同一次运行通过 ``turn`` 字段对应记录。请求在 HTTP 调用前保存，响应在校验和动作前保存，
异常响应也会保留。PNG 图像仍存放在各步产物中，trace 不重复存放 base64。
这三个文件可直接用 ``json.load`` 读取，也可用 ``grep`` 查找独立行中的字段；
历史 ``onejev_decisions.jsonl`` 日志不会被改写。放置步骤生成的
``onejev_geometry.json`` 和 ``onejev_regions.png`` 记录测量来源与区域标记；
抽屉步骤使用未标记的公开 RGB 图像。``onejev_outcome.json`` 记录终止状态和用量。成功以环境的 native completion 为准，进程正常退出和抓取启发式不代表任务成功。

首版支持非交互 CLI evaluation，拒绝 exploration、Dashboard、交互任务修改及
``--no-images``。planner 超时通过现有 Toolkit 在安全边界请求取消；已经发出的模拟器或 VLA RPC 可能在取消生效前完成。
