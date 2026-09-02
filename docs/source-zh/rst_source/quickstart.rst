快速开始
========

开始前，请先按照 :doc:`installation` 安装 RPent，并下载
LIBERO-PRO 仿真资源。下面以 LIBERO-PRO 和 ``claude_code`` planner
为例，演示如何完成一次运行。

1. 配置 API key 与 checkpoint
------------------------------

导出 Anthropic API key，然后下载并配置 VLA 和 SAM 3.0 checkpoint：

.. code-block:: bash

   # Anthropic 密钥；使用 Anthropic 官方 API 时无需设置 base URL。
   export ANTHROPIC_BASE_URL=https://xxx
   export ANTHROPIC_API_KEY=sk-xxx

   # VLA checkpoint —— 从下面地址下载
   # https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT
   pip install "huggingface_hub>=0.34,<1.0"

   hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
     --exclude optimizer.pt \
     --local-dir ./checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT

   export PI05_CHECKPOINT_PATH=$PWD/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT

   # SAM 3.0 checkpoint —— 从以下地址下载
   # https://modelscope.cn/models/facebook/sam3
   pip install -U modelscope

   modelscope download facebook/sam3 \
     --local-dir ./checkpoints/sam3

   export SAM3_CHECKPOINT_PATH=$PWD/checkpoints/sam3/sam3.pt

2. 跑一个 LIBERO 任务
---------------------

使用 ``claude_code`` planner 跑单个 LIBERO PRO 任务
（``libero_object_swap``，任务 ``2``，种子 ``0``）：

.. code-block:: bash

   rpent --robot libero --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

若要切换到其他 planner（如 ``codex`` 或 ``api``），请参阅
:doc:`Agentic Planner <usage/configure_planner>`。

3. 通过 Dashboard 查看运行过程
------------------------------

添加 ``--dashboard`` 后，RPent 会启动本地 Dashboard，并在终端输出访问地址：

.. code-block:: bash

   rpent --robot libero --dashboard --dashboard-language zh-cn \
     --planner claude_code --model claude-opus-4-8

打开该地址并确认配置；服务就绪后，在页面输入
``/rpent-task libero_object_swap 2 0`` 启动任务。Dashboard 会实时显示智能体的
推理过程、相机画面和动作时间线；任务结束后可以继续提交下一任务。使用
``--dashboard-language zh-cn`` 可切换到中文界面。

关键 CLI 选项
-------------

下表列出主要的命令行选项。其他通用选项可运行 ``rpent --help`` 查看；
有关 LIBERO 机器人的更多配置，请参阅
:doc:`LIBERO 使用指南 <usage/libero>`。

**主参数**

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - 参数
     - 默认值
     - 说明
   * - ``--robot``
     - —（必填）
     - 机器人后端。当前支持 ``libero``。
   * - ``--suite``
     - —（必填）
     - 任务套件，如 ``libero_object_task``、``libero_spatial_swap``
   * - ``--task``
     - —（必填）
     - 套件内的任务编号
   * - ``--seed``
     - ``0``
     - 随机种子
   * - ``--libero-type``
     - ``LIBERO_TYPE`` 或 ``pro``
     - LIBERO 类型：``standard`` | ``pro`` | ``plus``

**Planner**

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - 参数
     - 默认值
     - 说明
   * - ``--planner``
     - ``api``
     - ``api`` | ``claude_code`` | ``codex``
   * - ``--model``
     - —
     - 模型 ID；``api`` 需带 provider 前缀（``anthropic:…``、
       ``openai:…``、``openai-chat:…``）
   * - ``--max-turns``
     - ``100``
     - 智能体最大轮数
   * - ``--max-tokens``
     - ``8192``
     - 单次 LLM 回复最大 token
   * - ``--reasoning-effort``
     - ``none``
     - ``api``、``claude_code`` 与 ``codex`` 的推理强度：``none`` |
       ``low`` | ``medium`` | ``high`` | ``xhigh``。在我们的 LIBERO Pro
       Long 评测中，关闭 reasoning 将平均运行时间从约 13.2 分钟缩短至
       7.9 分钟（约 40%）。较高强度可能提升任务成功率；实际支持的档位
       取决于所选模型。
   * - ``--no-images``
     - 关
     - 纯文本模式：不向模型发送图片字节（用于不支持图片输入的模型）

**环境**

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - 参数
     - 默认值
     - 说明
   * - ``--max-episode-steps``
     - ``10000``
     - 环境最大步数
   * - ``--cuda-device``
     - 继承当前环境
     - env_server、vla_server 和 sam3_server 可见的 GPU 设备
   * - ``--env-endpoint``
     - —（自动启动）
     - 已在运行的 env_server 的 ``[protocol://]host:port``
       （``protocol=http|socket``，默认 ``http``）。留空时自动启动本地
       实例。
   * - ``--vla-endpoint``
     - —（自动启动）
     - 已在运行的 vla_server 的 ``[protocol://]host:port``
       （同上）。留空时自动启动本地实例。
   * - ``--sam3-endpoint``
     - —（自动启动）
     - 已在运行的 sam3_server 的 ``[protocol://]host:port``
       （同上）。留空时自动启动本地实例。

**Dashboard**

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - 参数
     - 默认值
     - 说明
   * - ``--dashboard``
     - 关
     - 启动本地 Dashboard
   * - ``--dashboard-language``
     - ``en``
     - Dashboard 界面语言：``en`` | ``zh-cn``

运行结果
--------

一次成功的运行会：

1. 终端会先显示 ``env_server``、``vla_server`` 和 ``sam3_server`` 的启动信息。
2. 智能体的逐轮输出和工具调用会显示在终端中；运行结束时还会显示耗时、token 用量和运行记录的路径。
3. 启用 Dashboard 后，智能体的输出、相机视图、动作时间线和片段回放也会实时显示在 Dashboard 中。
4. 默认输出目录为 ``logs/<timestamp>_<suite>_t<task>_s<seed>/``，其中包含 ``transcript_*.json``\ （运行记录）、``states.json``\ （``EnvState`` 清单）、``*_recipe.jsonl``\ （动作序列）和 ``episode.mp4``\ （回合录像）。每种逐步工件使用一个与逻辑工件同名的目录，目录内按步骤保存零填充文件，例如 ``agentview_depth.npz/00.npz`` 和 ``agentview_depth.npz/01.npz``；运行级工件仍保存在输出目录根部。

通过 Dashboard 或 ``view_env_state(step=-1)`` 查看最终状态；其顶层
``terminated`` 即为基准任务结果。``states.json`` 是 ``EnvState`` 的内部
存储，调用方不应直接解析。也可以打开 ``episode.mp4`` 复核运行过程。
出问题时，参考 :doc:`installation` 页底部提到的四份日志文件。
