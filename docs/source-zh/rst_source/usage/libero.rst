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

Cosmos Policy（实验性）
----------------------------------------

``--wam-backend cosmos-policy`` 通过独立启动的 RPC 服务使用 NVIDIA
`Cosmos Policy <https://github.com/NVlabs/cosmos-policy>`_ 的 LIBERO checkpoint。
任务集选择复用公共 LIBERO 目录，包括标准任务集及 Pro 的 ``_task``、``_swap``、
``_lan`` 和 ``_object`` 变体。首轮 120 回合评测仅覆盖四个核心基础任务族及其
``_task``、``_swap`` 变体；目录可选不代表其他任务集已有性能评测结果。
此适配器暂不支持探索模式、Flash Mode、plus 变体以及基于世界模型的
best-of-N 规划。

按官方 `安装指南 <https://github.com/NVlabs/cosmos-policy/blob/main/SETUP.md>`_
准备 Cosmos Policy 环境。适配器依据上游版本
``18a2accadf4e7a3531e56754102af5a24d2316da`` 实现。请使用独立环境：Cosmos
固定的 Torch 和 CUDA 扩展版本与 RPent 的 OpenPI 依赖不同。
假设 RPent 位于 ``/path/to/RPent``，在 Cosmos Policy 仓库目录下，使用官方
CUDA 12.8 / Python 3.10 环境启动服务：

.. code-block:: bash

   cd /path/to/cosmos-policy
   uv run --extra cu128 --group libero --python 3.10 \
     --with-editable /path/to/RPent \
     python -m rpent.robots.components.cosmos_policy_server \
     --cuda-device 0 --host 127.0.0.1 --port 8116

默认下载 ``nvidia/Cosmos-Policy-LIBERO-Predict2-2B``、对应的数据集统计量和
T5 指令嵌入。使用本地文件时，请同时设置 ``--checkpoint``、``--dataset-stats``
和 ``--text-embeddings``。从 Cosmos 仓库目录运行，以便解析其配置和 tokenizer
的相对路径。启动前，请在 Hugging Face 获得
``nvidia/Cosmos-Predict2-2B-Video2World`` 的访问权限，并在服务环境中登录；
即使使用本地策略 checkpoint，也需要该模型的视频 tokenizer。上述上游版本
在导入实验配置时，还会下载基础 Video2World 和 ALOHA 策略权重，需为这些
额外文件预留磁盘空间和网络访问条件。

在 RPent 环境安装 ``.[libero]``，执行
``libero-download-assets --skip-existing`` 下载标准 LIBERO 资源，并按下文配置
SAM3。Cosmos 运行不需要 Pi0.5 checkpoint：

.. code-block:: bash

   rpent --robot libero --suite libero_spatial --task 0 --seed 0 \
     --wam-backend cosmos-policy --wam-endpoint http://127.0.0.1:8116 \
     --memory-profile local \
     --cuda-device 1 --planner api --model anthropic:claude-opus-4-8

服务的 GPU 与 RPent 的 ``--cuda-device`` 分别配置，需为策略、仿真和 SAM3
预留足够显存；显存充足时也可共用一张 GPU。RPent 退出后，外部服务继续运行。
如果服务位于容器或另一台机器，请绑定可访问的网络接口，并在
``--wam-endpoint`` 中填写实际可访问的地址。

``cosmos_act(max_chunks=1)`` 替代 ``pi0_pick`` 和 ``pi0_doubled``，自动使用
环境的完整任务描述。每次预测执行 16 个原生 LIBERO 动作，各动作块之间重新
读取观测。原始相机图像仅做一次上下翻转，本体状态保持上游的
``[gripper_qpos, eef_pos, eef_quat_xyzw]`` 排列。图像预处理、归一化和动作
反归一化由 NVIDIA 实现负责，执行后的状态和图像由现有 LIBERO toolkit 记录。
未来视频及价值预测在此适配器中关闭。

可通过 ``cosmos_act(prompt="pick up the black bowl", max_chunks=1)`` 执行规划器
选择的具体子任务。指令仅对本次调用生效；省略 ``prompt`` 或传入 ``null`` 时，
恢复使用环境的原始任务描述。该参数不改变环境任务及成功判据，子任务是否完成需
观察执行后的图像，返回的 ``success`` 仍表示完整任务是否成功。子任务执行效果
取决于 checkpoint，需另行评测。

``max_chunks`` 限制完整动作块的数量，不是单个动作的数量。取值为 1–4，默认 1；
每块包含 16 个动作，因此一次调用最多执行 16–64 个动作。已开始的动作块会完整
执行，工具在块与块之间检查成功或截断标志，检测到任一标志后不再开始下一块，
不会精确停止在块内的某个动作。例如，回合预算还剩 5 个动作时，已开始的块仍会
执行完整的 16 个动作，因此最多可能超出预算 15 个动作。

先使用完整任务指令，在观察到进展时保持指令不变；遇到明确失败时再使用子任务。
与 ``pi0_pick`` 不同，此工具不会在检测到抓取完成时停止。``terminated`` 和
``success`` 记录执行过程中是否出现过原生任务成功，不保证同一块后续动作执行后，
最终姿态仍满足目标。``truncated`` 记录是否达到回合动作预算。这些标志会在已执行
动作中累计，因此两者可能同时为真。工具结果不能判断成功是否发生在预算耗尽之前；
严格动作预算下的成功率测量应使用下文的 benchmark runner。
未记录到原生成功时，``finish(status="success", ...)`` 会被拒绝。
工具报告回合结束后，后续动作工具调用会被拒绝，返回值会提示规划器按该结果结束。

对于向量缓存中没有的指令，官方服务会按需加载 ``google-t5/t5-11b``，并缓存
计算得到的文本向量。离线运行前需在服务的 Hugging Face 缓存中准备其 tokenizer
和权重，并为文本编码预留额外显存及首次请求时间。``--text-embeddings`` 应指向
可写的本地副本，因为上游会在编码新指令后更新该文件。

Cosmos 运行根据当前观测进行决策，暂不接入 Memory。仍需指定全局选项
``--memory-profile local``，用于跳过 CLI 和 Dashboard 的 HF 自动同步，
不表示启用本地 Cosmos 经验库。``--memory-dir`` 和 ``--memory-profile hf``
均会被拒绝。Cosmos toolkit 不提供 ``read_text_file``、``write_text_file``
和 ``list_dir``；观测及运行产物仍由 RPent 自动记录。

模型客户端和服务端位于 ``rpent/robots/components/``，当前观测格式和
checkpoint 仍仅支持 LIBERO；环境接线及 ``cosmos_act`` 保留在
``robots/libero/``。服务启动命令为
``python -m rpent.robots.components.cosmos_policy_server``，原有的
``robots/libero/cosmos_policy_server.py`` 入口已移除。
CLI 和 Dashboard 区分 ``--vla-backend pi05`` 与 ``--wam-backend cosmos-policy``，
Dashboard 中对应显示 VLA 或 WAM。两者复用公共策略预测和运行时生命周期代码。
模型选择位于 ``robots/libero/policy.py``；任务集名称及环境自动选择逻辑位于
``robots/libero/suites.py``。
公共策略机制、机器人工具和 benchmark 代码的职责划分见 :ref:`action-model-layers`。

更新已有 Cosmos 部署时，将 ``--vla-backend``、``--vla-endpoint`` 分别改为
``--wam-backend``、``--wam-endpoint``，并用与客户端相同版本的代码重启服务：
Cosmos 现在使用 ``wam.predict``。Pi0.5 仍使用 ``vla.predict`` 及原有参数。

运行 Pro 需安装 ``.[libero-pro]``，并使用
``liberopro-download-assets --skip-existing`` 准备资源。选择完整任务集名称，
例如 ``--suite libero_spatial_task`` 或 ``--suite libero_goal_swap``。
未指定 ``--libero-type`` 和 ``LIBERO_TYPE`` 时，公共 LIBERO 逻辑为扰动任务集
选择 ``pro``，为基础任务集选择 ``standard``；命令行参数优先于环境变量。
扰动任务集必须使用 ``pro``；基础任务集
允许显式选择已安装的其他变体（Cosmos 支持 standard/pro）。
若标准版配置指向不同的包，请为 Pro 使用独立的 ``LIBERO_CONFIG_PATH``。
运行前确认每个任务的初始状态非空，指令及成功条件均来自 Pro BDDL。
部分源码发行版本包含空初始状态文件，需从官方 HF 数据集
``zhouxueyang/LIBERO-Pro`` 获取对应文件，不能用标准任务的初始状态替代。

安装 ``.[test,libero]`` 后，可使用离线规划器验证真实服务及有界执行链路：

.. code-block:: bash

   RPENT_COSMOS_ENDPOINT=http://127.0.0.1:8116 CUDA_VISIBLE_DEVICES=1 \
     pytest tests/e2e_tests/libero/test_cosmos_policy.py -v

测试需要真实 LIBERO 资源；完整链路还需要 SAM3，子任务指令用例需要上述 T5
编码器权重。链路通过说明动作执行和产物记录正常，不代表任务成功。
设置 ``RPENT_COSMOS_SUITE=libero_spatial_task`` 或支持的 swap 任务集，
并配置 Pro 资源路径，即可在 Pro 上执行相同检查。

如需单独测量策略性能，准备运行中的服务和标准 LIBERO 资源后，在 RPent
仓库目录执行：

.. code-block:: bash

   RPENT_COSMOS_ENDPOINT=http://127.0.0.1:8116 CUDA_VISIBLE_DEVICES=1 \
     python -m tests.e2e_tests.libero.benchmark_cosmos_policy \
     --output-dir /path/to/new-cosmos-results

脚本使用固定的真实观测预热 5 次，再测量 100 次串行 RPC；随后评测 Spatial
全部 10 个任务，每任务使用初始状态 0、1、2，每回合最多执行 220 个策略动作。
``results.json`` 保存原始耗时、延迟分位数和每个回合的结果，包括异常。
RPC 耗时包含传输与推理，不含仿真步进；每次预测生成 16 个动作。
成功与否由仿真器的原生终止信号判断，评测不使用 LLM 规划器或 SAM3。
这 30 个回合属于小规模集成评测，并非论文基准复现。RPent 使用 RLinf 的
重置逻辑和当前安装的 LIBERO/robosuite 版本；报告结果时，应一并记录这些
版本以及服务的 checkpoint、去噪步数和随机种子。

评测脚本还支持 ``--suite``、``--tasks`` 和 ``--horizon``，复用公共目录，
但排除 ``libero_90``：此脚本面向每套十个任务的评测。默认动作预算为
Spatial 220、Object 280、Goal 300、Long 520，Pro 对应变体沿用相同预算。
例如 ``--suite libero_spatial_task --seeds 0 --warmup 0 --samples 0``
会测试该任务集全部十个任务的初始状态 0，并跳过独立延迟测量。
脚本保存首尾相机画面、每次 RPC 耗时、不含启动时间的控制循环耗时，以及
包含启动时间的回合总耗时。纯策略评测没有 LLM 输出，``total_output_tokens`` 为零。
与 ``cosmos_act`` 不同，此 runner 将预测动作逐个执行，在原生成功或达到动作上限时
立即停止，包括在 16 个动作的预测块内部停止。其成功率和动作数采用这一更严格的
评测协议。

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
  探索成功策略，并将其提炼为本地 global/suite/task_only 三层 memory corpus。它是
  memory 生成流程，不用于统计 benchmark success rate。
- **Evaluation** 是默认的单次评测模式，不会 reset episode，也不会更新
  memory。使用本地 memory 的 evaluation 会读取 exploration 生成并通过校验的
  audit、recipe 和经验。HarnessVLA 的 success rate 在 evaluation mode 下复现。

Pi0.5 默认仍为原有单次评测模式。省略 ``--memory-profile`` 时，会继续同步并使用
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
