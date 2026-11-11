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

四种模式使用 API planner，共用机器人、坐标和夹爪说明。

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

默认预算：100 次模型请求、10000 仿真步、单次输出 32768 tokens、262144-token
上下文、累积模式在 70% 时压缩历史、7200 秒 planner 超时。批量启动器另有
14400 秒 episode 超时。四种 LIBERO 模式共用 ``simple`` 、 ``base`` 、
``base_gripper`` 、 ``base_gripper_camera`` 四种 prompt 变体。

启动器显式默认发送 presence penalty0，示例设置 repetition penalty1.0；
覆盖值记录在实际模型请求中。Thinking 采样为 temperature 1.0、top-p 0.95、top-k 20，
Qwen3.8 默认 effort 为 ``xhigh`` 。
:download:`参考文档 <../../../../tools/rl/REFERENCE_zh.md>` 列出服务、planner、
批量采集的完整参数，以及实际请求的记录方式。

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
