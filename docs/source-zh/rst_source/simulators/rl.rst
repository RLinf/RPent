Adaptive 机器人工作流
================================================================

使用 ``--robot rl``，让 VLM 通过末端控制和视觉工具操作机器人。LIBERO、RoboCasa、
RoboTwin 共享 episode planner，LIBERO 还支持实验性的 zap_turn、zap_episode、zap_subtask 模式。
成功与否由原生仿真器判定，无需 VLA 权重或任务解法记忆。

.. list-table:: 支持的组合
   :header-rows: 1

   * - 仿真器
     - Episode
     - ZAP
   * - ``libero`` （含 LIBERO-PRO）
     - 支持
     - 实验性
   * - ``robocasa``
     - 支持
     - 不支持
   * - ``robotwin``
     - 支持
     - 不支持

安装与资源
----------------------------------------------------------------

:download:`安装指南 <../../../../tools/rl/INSTALL_zh.md>` 从新的 checkout 开始，
覆盖系统库、隔离 Python 环境、模型、SAM3、仿真资产与排错。所有路径由自己的
checkout 和数据目录推导，不依赖集群路径。复制 ``tools/rl/config.example.sh``
为 ``.rl-env.sh`` ，再按实际资源修改。

公共环境运行 planner、SAM3、LIBERO；RoboCasa、RoboTwin 各用隔离仿真环境，
模型服务单独安装。Linux 上需要 NVIDIA EGL；RoboTwin 还需要 Vulkan 及编译
cuRobo 的 CUDA toolkit。先准备资产、检查导入，再运行任务。

运行一个任务
----------------------------------------------------------------

:download:`首次运行指南 <../../../../tools/rl/README_zh.md>` 提供完整的服务启动和
四条任务命令：LIBERO episode、LIBERO ZAP、RoboCasa episode、RoboTwin
episode。先从一个 episode 开始。

配置并启动模型和共享 SAM3 后，在 checkout 根目录运行：

.. code-block:: bash

   source .venv/bin/activate
   source .rl-env.sh
   source tools/rl/env.sh
   bash tools/rl/run.sh --simulator libero --workflow episode \
     --suite libero_object --task 0 --seed 1 --cuda-device 1

``--cuda-device`` 选择仿真 GPU，与模型/SAM3 的 GPU 独立。RoboCasa 使用
``--task-name`` 和 ``--split`` ；RoboTwin 使用 ``--task-name`` 、
``--task-config`` 及公布的有效 seed。首次运行指南给出完整示例和任务清单位置。

工作流与工具
----------------------------------------------------------------

五种模式使用 API planner，共用机器人、坐标和夹爪说明。

.. list-table:: LIBERO 上下文策略
   :header-rows: 1

   * - 工作流
     - 感知与计划
     - 对话保留范围
   * - ``episode``
     - ``view_env`` 、 ``segment`` 、 ``back_project``
     - 整条任务
   * - ``zap_turn``
     - 自动感知、文字 todo
     - 每次请求独立
   * - ``zap_episode``
     - 自动感知、没有 todo
     - 整条任务
   * - ``zap_subtask``
     - 自动感知、结构化 todo
     - 每次有效 ``update_todo`` 后清空

动作工具为 ``move_to`` 、 ``set_gripper`` 、 ``finish`` 。自动感知使用当前帧
SAM3 框和深度测得的世界几何。每次回复仅一个工具，子任务初始化则同时要求
query 和 todo 工具。重置不会重置全局预算。原生任务文本直接来自仿真器，没有
逐任务 prompt 文件。RoboCasa 增加 ``move_base`` ，RoboTwin 增加解剖学左右臂
选择与双臂指令，两者使用 ``episode`` 。

:download:`工作流指南 <../../../../tools/rl/WORKFLOWS_zh.md>` 说明初始化与重置、
完整工具、SAM3 池和固定任务对比。 ``--image_size`` 为 VLM 图片选择 1024、512
或 256，原始 RGB-D/SAM3 始终为1024，显示像素和矩形会映射回原始坐标。

规划器配置
----------------------------------------------------------------

修改 ``tools/rl/run.sh`` 的 ``run_args`` 参数区即可配置单条运行和批处理默认值；
每个参数独立一行并附中文说明。优先级是：显式命令行参数、``RPENT_*`` 环境变量、
脚本中的默认值。只查看默认 argv，不启动模型或仿真：

.. code-block:: bash

   bash tools/rl/run.sh --print-defaults

``run.sh`` 和 ``python -m tools.rl.run_batch run`` 均支持下列参数。
批处理从同一脚本读取默认值，向每个子进程显式传入解析后的值，并固定在
``manifest.json`` 中；改变参数时需使用新的输出目录。

.. list-table::
   :header-rows: 1

   * - 参数
     - 默认值
     - 含义
   * - ``--max-tokens``
     - 4096
     - 单轮总输出预算，包含 reasoning 和工具协议。
   * - ``--context-window``
     - 262144
     - 计数使用的服务端上下文容量；不会修改服务端启动配置。
   * - ``--compaction-fraction``
     - 0.7
     - 当前输入超过上下文容量的该比例时触发压缩。
   * - ``--compaction-max-messages``
     - 80
     - 独立的消息数量触发上限；0 关闭这条规则。
   * - ``--compaction-keep-messages``
     - 10
     - 压缩后保留的尾部消息数，另保留开场、pinned 内容和完整工具配对。
   * - ``--max-turns`` / ``--max-episode-steps``
     - 80 / 10000
     - 每条轨迹的模型请求次数／物理环境步数上限。
   * - ``--planner-timeout-s``
     - 7200
     - planner 墙钟时限；批处理 ``--episode-timeout`` 默认14400秒。

一次正常模型回复及其工具结果是两条消息，保留10条尾部消息通常约为5轮。
token 与消息数量条件为“或”；若只需 token 触发，设置
``--compaction-max-messages 0``。保留后的窗口仍可能超过 token 阈值，审计会记录。

采样默认保持 Qwen3.8-27B 官方 thinking 配置：
``--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repetition-penalty 1.0``。
六个值均显式写入模型请求；``--top-k -1`` 禁用 top-k。
默认 ``--thinking``；``--no-thinking`` 关闭本轮思考，但不会自动改变采样。
27B 启动器默认 ``--reasoning-effort xhigh``，也可选 ``low`` 或 ``medium``；
``none`` 只省略 effort 字段，不等于关闭 thinking。

同一参数区还列出 ``--reasoning-history``、``--image-history``、
``--action-mode``、``--max-actions-per-call`` 和 ``--image_size``。
默认历史为 -1/-1、动作为 single、batch 长度为3、VLM 图像为1024。
历史裁剪之后仍会进行压缩判断。

例如在单条或批处理命令后追加 ``--max-tokens 8192 --temperature 0.8
--compaction-max-messages 0``。实际输入与请求的输出仍须满足服务端上下文容量。
:download:`参考文档 <../../../../tools/rl/REFERENCE_zh.md>` 说明服务部署及保存的数据。

查看结果
----------------------------------------------------------------

``result.json`` 保存原生成功标记、错误、轮数和 tokens， ``states.json`` 保存
实测状态及原生终止标记。相机/深度数据、 ``episode.mp4`` 、 ``rollout.json`` 、
原生 transcript 和 ``api_trace/`` 保留执行证据。进程正常退出并不证明任务成功或
轨迹可用于训练；批量启动器审计完整请求/响应及图像哈希后才发布记录。

采集任务与 seed
----------------------------------------------------------------

参考文档说明有界并发、valid/evaluated 索引、基础设施重试与中断恢复。使用新的
批次目录，先从每 GPU 一个 episode 开始，再根据实测场景显存增加并发。
``tools/rl/jobs`` 提供示例任务/seed 清单。

远程仿真后端
----------------------------------------------------------------

Adaptive episode 可通过 ``python -m robots.rl.remote_manager`` 和
``RPENT_REMOTE_TOOLKIT_ENDPOINT`` 将动作、渲染、SAM3 放到另一台后端机器。
两侧源码必须一致。参考文档覆盖两侧安装、私有网络连接、GPU 容量、源码/任务/
预算校验、文件哈希与租约清理。远程 ZAP 暂不支持。

RLinf 训练
----------------------------------------------------------------

参考文档保留离线安装、RLinf/SFT 集成说明。RLinf 负责策略训练和 checkpoint，
RPent 负责仿真器与工具执行；episode 入口为 ``robots.rl.rl_episode`` 。
数据编码位于 ``ms-swift/examples/rpent`` ，应保留实际请求与图片顺序。


图片反馈、动作列表与历史保留
------------------------------------------------

LIBERO 新增 ``--workflow episode_image`` ：初始及动作后返回最新双相机图，
没有 ``view_env`` 。 ``segment`` / ``back_project`` 不接收 ``step`` ，
固定使用当前观测；原 ``episode`` 保留历史帧查询。

LIBERO 使用绝对 grip-site ``xyz`` 和 ``orientation_rpy=[roll,pitch,yaw]`` ，
单位弧度，固定轴 XYZ： ``R=Rz(yaw)@Ry(pitch)@Rx(roll)`` 。
至少提供一个目标，省略项保持动作开始时的实测值。
``--action-mode batch --max-actions-per-call 3`` 使用顺序 ``execute_actions`` ；
先校验整个列表，成功、截断、partial 或 error 立即停止。
``--reasoning-history``、``--image-history`` 支持 -1（不额外裁剪）、
0（全部移除）或 N（最近 N 个回复／观测组）；双相机共同一组，默认均为 -1。
原有压缩继续生效；这些开关支持本机 LIBERO episode 模式。
完整接口与可配置路径命令见 ``tools/rl/WORKFLOWS_zh.md`` 。


多模态 token 计数
------------------------------------------------

API planner 在 reasoning／图片历史裁剪后，对当前请求重新计数。
Qwen 使用服务端 checkpoint 的本地副本读取 chat template、tokenizer 和
image processor；本机 SGLang 服务可自动提供 checkpoint 路径。
也可以显式指定：

.. code-block:: bash

   export RPENT_TOKENIZER_PATH=/path/to/served-checkpoint

远程 SGLang 服务的 checkpoint 如果挂载在本地，还需设置
``RPENT_TOKEN_COUNT_BACKEND=sglang``。默认 ``auto`` 自动识别本机 SGLang；
``chat_template`` 只使用 checkpoint 模板，不补服务端工具默认字段。
与服务端 usage 不一致时会输出警告。

图片按 processor 的缩放、patch 和空间合并规则计数，无需运行视觉编码器。
文本、工具 schema、reasoning 和模板标记按 chat adapter 的实际渲染计入。
图片元数据按内容 SHA 与 processor 配置缓存；图片每次出现在上下文中都占用
token，缓存命中不会减少上下文长度。历史裁剪或压缩后重新计算，不沿用旧 usage。

启动器采用 70% token 阈值与 80→10 消息窗口，保留工具调用／结果配对及开场提示词。
如果保留窗口本身仍然很大，压缩后可能仍高于阈值；审计会明确记录，不自动修改
输出或上下文预算。

开启 ``RPENT_RECORD_MODEL_REQUESTS=1`` 后，``api_trace/token_counts`` 记录压缩
前后计数、图片尺寸／SHA／视觉 token、触发原因、计数耗时，以及与服务端 input
usage 的差值。缺少本地 processor 时会警告，并在审计中标明文本／几何估算；
不会将图片字节转成文本计数。修改 checkpoint、模板或服务端图片预处理参数后，
应使用一致配置重新执行真实服务核对：

.. code-block:: bash

   RPENT_RUN_TOKEN_COUNT_INTEGRATION=1 \
   RPENT_TOKEN_COUNT_ENDPOINTS=http://127.0.0.1:30000/v1 \
   pytest tests/e2e_tests/libero/test_token_counting.py -v

此检查需要已运行的 SGLang Qwen 服务和本地 processor 文件，对照真实 input
usage 验证三种图片尺寸、工具回传图片、thinking／历史设置及两种压缩触发条件。
每次请求只允许输出一个 token，不执行仿真动作，不计入正式评测。

Episode 感知列表
----------------

本地 ``episode`` 可选 ``--perception-mode batch``，用
``observe(calls=[...])`` 替换三个独立感知工具。默认 ``single`` 保留原接口。
Episode-image、ZAP 和远程 toolkit 启动前拒绝此选项。

每项有局部 ``id`` 和 ``op``。view 不传 source；测量通过 ``source`` 选择已返回的
图片／mask ID，或 ``@v.agentview``、``@cup`` 这类本次引用。ID 固定帧与相机。
整个列表先校验；代码从来源推导依赖，独立节点并行，不在工具内部额外调用模型。
动作批处理可独立选择。

``--max-observe-calls 8`` 限制列表长度，``--observe-concurrency 8`` 限制并发。
两个启动器都支持，恢复时固定参数。详见
:download:`感知列表指南 <../../../../tools/rl/OBSERVE_zh.md>` 的启动、source、示例、
返回和审计说明。
