LIBERO
======

`LIBERO <https://libero-project.github.io/>`_ 是 RPent 主要使用的仿真基准，
包含一系列基于 MuJoCo/robosuite 的桌面操作任务。RPent 主要使用四个核心基础
任务族（``libero_object``、``libero_goal``、``libero_spatial``、
``libero_10``）和三个变体（``standard``、``pro``、``plus``）。默认 VLA
是 **Pi0.5**，由 ``rpent/robots/components/pi05_vla_server.py`` 通过 HTTP 提供服务。

VLA 配置
--------

下载推荐的 SFT checkpoint
`RLinf-Pi05-LIBERO-130-fullshot-SFT
<https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT>`_，
再将 ``PI05_CHECKPOINT_PATH`` 指向本地 checkpoint 目录：

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
     --local-dir /path/to/rlinf-pi05-libero-130-fullshot-sft

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

SAM3 配置
---------

每次 LIBERO 运行都默认启用 SAM 3.0 分割。从
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

可选 World Action Model
-----------------------

LIBERO 可以在不改变默认 Pi0.5 路径的前提下连接一个可选 WAM。模型及其 CUDA
依赖运行在独立的上游环境中，RPent 仅通过轻量 bridge 调用
``action_model.capabilities`` 和 ``action_model.predict`` RPC。不传 WAM 参数时，
现有 runtime、prompt 和工具集合保持不变。

当前可执行接入以官方
`Cosmos Policy Predict2 2B LIBERO checkpoint
<https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B>`_ 为目标。
在官方 ``cosmos-policy`` LIBERO 环境中，将 RPent checkout 加入
``PYTHONPATH``，然后启动 bridge：

.. code-block:: bash

   cd /home/gao/worldmodel/harnessvla/cosmos-policy
   export HF_HOME=/home/gao/worldmodel/harnessvla/checkpoints/huggingface
   export HF_HUB_CACHE=/home/gao/worldmodel/harnessvla/checkpoints/huggingface-http
   export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 COSMOS_INTERNAL=1
   export CUDA_HOME=$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc
   export PYTHONPATH=/home/gao/worldmodel/harnessvla/rpent
   .venv/bin/python \
     /home/gao/worldmodel/harnessvla/rpent/scripts/wam/cosmos_policy_rpc_bridge.py \
       --checkpoint /home/gao/worldmodel/harnessvla/checkpoints/cosmos-policy \
       --host 127.0.0.1 --port 8120

然后在普通 RPent 命令中增加 endpoint：

.. code-block:: bash

   rpent --robot libero \
     --suite libero_10 --task 0 --seed 0 \
     --wam-backend cosmos --wam-endpoint http://127.0.0.1:8120 \
     --planner codex

bridge 继续使用上游提供的图像/proprio 预处理、数据集统计、动作反归一化、未来
状态解码和值函数解码。只有服务明确声明完整的 ``libero_7d`` schema 后，RPent
才会注册 ``wam_act``。该工具只执行有界动作块；需要多个 chunk 时，每轮都会从
真实环境的新观测重新预测。预测的未来图像不会进入 Agent 工具文本。请求使用
LIBERO 原始相机帧，以及 checkpoint 原生的 9 字段 proprio 顺序（两个夹爪位置、
EEF 位置、EEF 四元数）；bridge 再执行上游要求的垂直翻转。任何超出 LIBERO
``[-1, 1]`` 控制范围的位移/旋转动作都会被拒绝。仅二值夹爪维会裁剪到该范围，
以容忍 Cosmos 去噪输出在 ``-1`` / ``+1`` 边界的微小越界。``wam_act`` 自动使用
环境返回的原始任务语言和官方预计算 T5 embedding 缓存；它不接受 Agent 改写的
instruction，缓存未包含该任务时也不会在线加载 T5-11B。请使用下载缓存已覆盖
其准确任务语言的标准 LIBERO 任务；自定义 LIBERO-Pro 指令只有在事先加入官方
T5-11B embedding 后才受支持。

WAM 操作流程
~~~~~~~~~~~~~

bridge 所在终端必须在整个 Dashboard/RPent 会话期间保持运行。另开一个终端
检查 bridge：

.. code-block:: bash

   curl -sS http://127.0.0.1:8120/call \
     -H 'content-type: application/json' \
     -d '{"method":"healthz","args":[],"kwargs":{},"session_id":null}'

使用本地 Pi0.5/SAM3 checkpoint 启动 RPent：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/home/gao/worldmodel/harnessvla/rpent/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT
   export SAM3_CHECKPOINT_PATH=/home/gao/worldmodel/harnessvla/rpent/checkpoints/sam3/sam3.pt
   export LIBERO_TYPE=standard
   rpent --robot libero --dashboard --dashboard-language zh-cn \
     --planner codex --model gpt-6-astra --reasoning-effort low \
     --wam-backend cosmos --wam-endpoint http://127.0.0.1:8120 \
     --memory-profile local --memory-dir /path/to/RPent/memory/libero \
     --cuda-device 0

WAM 显示 ``ready`` 后，提交 ``/rpent-task <suite> <task> <seed>``。由于
``wam_act`` 是可选工具，Planner 可能选择 Pi0.5；可以明确发送：
``现在立即调用 wam_act；不要使用 Pi0.5；max_chunks=1，max_actions_per_chunk=8。``
任务启动后也可以绕过 Planner，直接调用 Dashboard primitive 接口：

.. code-block:: bash

   curl -sS -X POST http://127.0.0.1:59307/api/session/primitive \
     -H 'content-type: application/json' \
     -d '{"name":"wam_act","arguments":{"max_chunks":1,"max_actions_per_chunk":8}}'

结束时先关闭 Dashboard/RPent，再在 bridge 终端按 ``Ctrl+C``。

如果希望由 RPent 管理本地 Cosmos bridge，可将 ``--wam-endpoint`` 替换为：

.. code-block:: bash

   --wam-backend cosmos \
   --wam-checkpoint /home/gao/worldmodel/harnessvla/checkpoints/cosmos-policy

RPent 会启动隔离的 Cosmos 环境、等待 capabilities，并在 Dashboard 清理时停止
bridge。隔离 Python 默认使用 RPent checkout 同级的
``../cosmos-policy/.venv/bin/python``；需要时可通过 ``COSMOS_POLICY_PYTHON`` 覆盖。
``--wam-endpoint`` 和 ``--wam-checkpoint`` 不能同时使用。

DreamZero-DROID 使用官方 DreamZero WebSocket 服务。先在独立环境中启动原生
服务，再启动 RPent proxy bridge：

.. code-block:: bash

   # 在官方 DreamZero checkout/环境中：
   torchrun --standalone --nproc_per_node=2 socket_test_optimized_AR.py \
     --port 8000 --enable-dit-cache --model-path /path/to/DreamZero-DROID

   PYTHONPATH=/path/to/RPent \
     python /path/to/RPent/scripts/wam/dreamzero_rpc_bridge.py \
       --dreamzero-host 127.0.0.1 --dreamzero-port 8000 \
       --checkpoint /path/to/DreamZero-DROID \
       --host 127.0.0.1 --port 8121

DreamZero-DROID 需要两个外部相机、一个腕部相机和原生 14 字段 proprio，返回
7 个关节位置加 1 个夹爪命令。这并不是 LIBERO 的 7 维 OSC 动作 schema，因此
RPent 会主动拒绝使用该 checkpoint 在 LIBERO 中执行
``--wam-backend dreamzero``。该 bridge 当前只用于验证 DreamZero 原生连接和推理；
不得通过截断、补零或重新解释动作来绕过限制。只有另行验证的 LIBERO checkpoint
或 embodiment adapter 才能启用执行。原生预测请求必须提供非空的
``metadata.episode_id``，以便 DreamZero 服务在 episode 切换时重置时序状态。

Cosmos 需要将 ``--wam-backend cosmos`` 与以下二者之一配对：外部 bridge 使用
``--wam-endpoint``，RPent 托管生命周期使用 ``--wam-checkpoint``；两者不能同时
使用。DreamZero 仍由外部服务管理，并继续被 LIBERO 拒绝。Capabilities 和一次
真实 prediction 已验证；bounded ``wam_act`` 的 artifact、清理证据和任务成功应
分别记录。

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
  探索成功策略，并将其提炼为本地 global/task-family/task-specific 三层 memory corpus。它是
  memory 生成流程，不用于统计 benchmark success rate。
- **Evaluation** 是默认的单次评测模式，不会 reset episode，也不会更新
  memory。使用本地 memory 的 evaluation 会读取 exploration 生成并通过校验的
  audit、recipe 和经验。HarnessVLA 的 success rate 在 evaluation mode 下复现。

默认仍为原有单次评测模式。省略 ``--memory-profile`` 时，会继续同步并使用
Hugging Face memory 和原有 prompt。两种 profile 都执行相同的单次评测流程；
区别仅在于评测 memory 的来源及所使用的 memory prompt。本地 memory 已准备好后
（例如先执行下文的 exploration 流程），即可使用 ``local``。该选项不会开启 exploration，也不会从 Hugging Face 下载
memory；它只会针对 ``--memory-dir`` 执行普通的单次评测，并避免同步覆盖本地
目录：

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner codex --memory-profile local \
     --memory-dir /path/to/libero-memory

探索模式沿用同一个 Python/CLI 入口。它支持可重置的多次尝试和独立
planner session，并在正常结束后校验、合并 memory，只有 LIBERO 确认成功时
才发布 task audit/recipe。探索可以从空的 ``--memory-dir`` 开始，并始终使用
local profile；真正开启该流程的是 ``--explore``：

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 0 \
     --planner api --model anthropic:claude-opus-4-8 \
     --explore --explore-sessions 3 --explore-attempts-per-session 5 \
     --memory-dir /path/to/libero-memory

每个 planner session 使用一个新建的 toolkit，其状态轨迹和观测工件保存在
``<output-dir>/sessions/session_NNN/``，供最终 memory distillation 使用；同一
session 内通过 reset 发起的多次 attempt 仍复用该 toolkit。

在 exploration 命令中增加 ``--dashboard``，即可跨 planner session 查看完整
推理过程、相机画面和连续动作时间线。

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
- **sam3_server** （``rpent/robots/components/sam3_server.py``）—— 持有 SAM 3.0，
  通过同一套 RPC 传输（HTTP 或 socket）支持文本或单个正点分割，仅返回
  排名第一的压缩 PNG mask。
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
- ``move_pose(xyz, target_pitch=..., target_yaw=..., ...)`` —— 同时调整
  末端位置和姿态。
- ``rotate_wrist(target_yaw=... / delta_yaw=..., ...)`` —— 按绝对或相对
  yaw 旋转腕部。
- ``rotate_pitch(target_pitch=... / delta_pitch=..., ...)`` —— 按绝对或
  相对 pitch 倾斜夹爪。
- ``set_gripper(gripper=..., steps=...)`` —— 保持末端姿态，并在指定步数内
  控制夹爪。
- ``release(...)`` —— 打开夹爪。

物理动作工具执行后会推进环境，并记录新的状态和图像。

**只读工具：**

- ``back_project(row, col, ...)`` —— 将图像像素反投影到世界坐标。
- ``segment(prompt=... / point=..., ...)`` —— 通过 SAM3 对已有图像进行文本或
  点提示分割。
- ``view_env_state(step=-1)`` —— 读取已记录的状态和内嵌观测图像；第 0 步为
  初始状态，``-1`` 表示最新状态。
- ``view_camera_meta(camera=..., step=-1)`` —— 读取指定步骤的相机元数据；
  ``-1`` 表示最新状态。
- ``finish(status, summary)`` —— 结束当前运行。

这些工具不会推进环境。

Dashboard
---------

加上 ``--dashboard`` 可启动长生命周期的本地 Dashboard Session。系统会自动
选择一个空闲端口，并在终端输出访问 URL：

.. code-block:: bash

   rpent --robot libero --dashboard \
     --planner claude_code --model claude-opus-4-8

Session 配置全部来自命令行，打开地址后会直接进入实时监控。共享服务就绪后，
输入以下命令启动 TaskRun：

.. code-block:: text

   /rpent-task libero_object_swap 2 0

Dashboard 支持 ``api``、``claude_code`` 和 ``codex`` planner。
在命令行传递 ``--planner`` 与 ``--model``，配置方式和普通运行一致，详见
:doc:`configure_planner`。

每个 TaskRun 使用独立环境，VLA 和 SAM3 服务由 Session 复用。可通过新的
``/rpent-task`` 启动或切换任务；运行中也可以发送消息引导智能体，并按 Esc
请求中断。在终端按 Ctrl+C 可结束 Session。

``--dashboard`` 不能与 ``--interactive`` 或 ``--env-endpoint`` 同时使用；外部
``--vla-endpoint`` 和 ``--sam3-endpoint`` 服务仍然可用。使用
``--dashboard-language zh-cn`` 可切换中文 UI。

接入自定义 VLA
----------------

如果你有一个与 LIBERO 兼容、但并非 Pi0.5 的 VLA，可以在不修改机器人实现的
情况下替换 model client：

1. 写一个新的 ``vla_server.py``，暴露相同的 ``predict`` RPC 契约
   （HTTP 或 socket 均可）。
2. 用 ``--vla-endpoint [protocol://]host:port`` 指向它。
3. 如果可用工具需要调整（比如将 ``pi0_pick`` 改成 ``mymodel_pick``），
   相应更新 ``robots/libero/toolkit.py``。

完整流程见 :doc:`../development/add_primitive`。

结果复现
--------

RPent 在 LIBERO-PRO Task/Swap 上的统一模型对比及对应配置见 :doc:`../leaderboard`。

:doc:`GPT-6 Astra 套件汇总 <../leaderboard>`
记录全部八个完整套件及 800 个已核验回合：741 成功、59 失败，Overall 92.63%，
配置为 Codex / GPT-6 Astra / low / reasoning。

以下保留历史复现记录，实验使用
`reproduce/libero <https://github.com/RLinf/RPent/tree/reproduce/libero>`_
分支和 ``gpt-5.5`` 模型：

- ``libero_10_task``：70%（70/100）
- ``libero_10_swap``：55%（55/100）

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
