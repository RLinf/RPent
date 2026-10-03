单臂 Franka
=============

.. Product image: https://store.clearpathrobotics.com/products/franka-research-3

.. figure:: https://cdn.shopify.com/s/files/1/1750/5061/products/FR3_image3_x700.png?v=1663341441
   :alt: Franka Research 3 机械臂与夹爪全貌
   :figclass: rpent-robot-figure
   :align: center

   用于真机实验的 Franka 机械臂。

RPent 可以通过 RLinf 的 ``RealWorldEnv`` worker 控制单台 Franka 机械臂。

安装
----

请先按照 `libfranka 官方快速安装指南 <https://docs.ros.org/en/humble/p/libfranka/__README.html#quick-install>`_ 安装 libfranka 0.19.0。

克隆 RPent 并安装 Python 依赖。若已有仓库，进入仓库后执行 ``uv sync``：

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent
   uv sync --extra franka

该命令将 RLinf ``release/v0.4``、``rpent-openpi`` 以及 Franka 的相机、夹爪和遥操作依赖安装到 ``.venv``。其中的 ``franky-control`` wheel 已包含 libfranka 0.19.0。

标定（Calibration）
----------------------

手眼标定使用 ROS 的 `easy_handeye <https://github.com/IFL-CAMP/easy_handeye>`_ 完成。它为每台相机生成一个 YAML （外部相机为 eye-on-base，腕部相机为 eye-on-hand），默认保存在 ``~/.ros/easy_handeye/`` 下。

RPent 会直接加载这些 YAML 文件：在机器人配置的 ``perception.calibration`` 下，将每台相机映射到对应的 easy_handeye YAML 即可（仓库中的 ``robots/franka/config/example.yaml`` 已经包含该映射）：

.. code-block:: yaml

   perception:
     calibration:
       external: ~/.ros/easy_handeye/fr3_external_apriltag_eye_on_base.yaml
       wrist: ~/.ros/easy_handeye/fr3_wrist_apriltag_ee_eye_on_hand.yaml

路径可以是绝对路径、以 ``~`` 开头的路径或相对路径；相对路径会相对启动 RPent 时的工作目录解析。

开发配置
--------

仓库中给出的值是开发默认值，在启用机械臂运动前必须逐项核对：

* ``robots/franka/config/example.yaml``，包含机器人身份（机器人 IP、相机序列号、夹爪）、工作空间几何（目标/复位位姿、安全边界）和 easy_handeye YAML 映射（见上方标定说明）。示例设置 ``backend: franky`` 和 ``realtime_config: ignore``。如果需要强制 RPent 在 PREEMPT_RT 内核上运行，可将 ``realtime_config`` 改为 ``enforce``，此时若 RPent 无法获得实时保证，将拒绝启动。

RPent 会把这份机器人配置转换成内部的 RLinf cluster 和环境对象。如需改用其他文件，请传入 ``--robot-config /path/to/robot_config.yaml``。

启动 Ray
--------

Ray 在启动时会捕获环境变量，因此必须先设置 node rank 再启动：

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray stop --force
   ray start --head

运行冒烟测试
------------

先按 :doc:`../guides/configure_planner` 配置规划器与模型服务。

冒烟测试用于验证基本的解析式运动和夹爪动作是否正常工作。运行时指定任务 ``0``：

.. code-block:: bash

   # replace --robot-config with your own config
   uv run --extra franka rpent --robot franka --task-id 0 \
     --planner claude_code --model claude-opus-4-8 \
     --robot-config robots/franka/config/example.yaml

RPent 会使用当前解释器启动 ``robots/franka/env_server.py``：加载机器人配置、生成内部的 RLinf 适配器配置、连接 Ray、等待 ``healthz``，并将初始状态记录为第 ``0`` 步。

VLA 抓取演示
-------------

RPent 提供了一个使用 VLA 抓取物品的演示。任务 ``1`` 提供 ``vla_grasp`` 工具。当前单臂 Franka 需要兼容的外部 VLA 服务，其观测布局、动作布局、checkpoint 和归一化统计必须与 Franka 训练配置一致：

.. code-block:: bash

   uv run --extra franka rpent --robot franka --task-id 1 \
     --vla-endpoint http://VLA_HOST:PORT \
     --planner claude_code --model claude-opus-4-8 \
     --robot-config robots/franka/config/example.yaml

目前 VLA 服务需要单独部署。若未设置 ``--vla-endpoint``，解析式运动和夹爪工具仍然可用，但 ``vla_grasp`` 会抛出运行时错误。

工具与状态产物
--------------

Franka 扩展提供 ``view_env_state``、``view_camera_meta``、``move_delta``、 ``rotate_delta``、``open_gripper``、``close_gripper`` 和 ``vla_grasp``。所有会改变环境状态的工具都会把机器人状态、腕部与外部 RGB 图像、可选的对齐深度数组和相机元数据保存到 RPent 统一的 ``EnvState`` 中。

安全要求
--------

操作员必须守在急停按钮旁。在尝试抓取之前，先用极小幅度动作验证任务 ``0``。一旦相机与状态结果不一致、目标运动没有到位，或任何标定存在疑问，应立即停止。

.. _franka-flash:

Franka Flash 任务卡
------------------------

单臂和双臂 Franka 可通过 ``--planner flash`` 回放经过人工审阅的 v1 JSON
任务卡。任务卡固定动作顺序和运动意图；每次平移前，Molmo 在最新相机图像中
选点，再通过已有的深度反投影和标定转换为机器人坐标。回放不调用规划模型。

平移偏移取自示范中 **实际到达的 TCP 终点** 与示范锚点位置之差。回放时，
用当前定位的锚点加上该偏移得到新目标，再减去当前 TCP 位置得到移动量。
动作顺序保持不变，连续运动的插值仍由现有控制器负责。

先按本页配置机器人及 ``perception.calibration``，启动 Molmo 服务，并记录一次
成功操作。检查 ``states.json``，按原始步骤编号填写 ``annotations.json``::

   {
     "1": {"intent": "approach the cup rim", "phrase": "visible cup rim", "camera": "third_person"},
     "2": {"intent": "align the gripper", "rotation_mode": "relative"}
   }

每次平移都需要语义锚点；每次旋转都需要意图及 ``relative`` 或 ``fixed`` 模式。
单臂可使用 ``third_person``、``wrist``；双臂可使用 ``base``、``d455``、
``left_wrist``、``right_wrist``。所选视角必须具有深度和有效标定。
单步平移限制为 0.20 m，单步旋转限制为 0.35 rad。

从审阅后的记录生成任务卡；此操作不连接机器人，但需要可访问的 Molmo 服务::

   python -m robots.franka.flash.generate \
     --robot franka --task franka_t0 \
     --robot-config /path/to/robot.yaml \
     --run-dir /path/to/successful-run --annotations annotations.json \
     --molmo-endpoint http://localhost:9000 --destination plan.json

按提示确认原始操作成功。生成器拒绝不支持的命令，且不会覆盖已有目标文件。
新记录保存 ``recording_fingerprint.json``，包含机器人配置及标定文件的哈希。
生成器在定位锚点前拒绝缺少指纹或录制后配置、标定发生变化的记录；此时需重新录制，
没有来源指纹的旧记录不能使用。回放仍检查任务卡与当前文件是否一致。
已知观察、确认记录及初始确认的场景重置不进入任务卡；任务动作之后的重置会被拒绝，
避免拼接不同尝试。
双臂改用 ``--robot dual_franka --task dual_franka_t0``。

在独立操作员终端中回放::

   rpent --robot franka --planner flash --task-id 0 \
     --robot-config /path/to/robot.yaml --flash-plan plan.json \
     --molmo-endpoint http://localhost:9000

双臂使用 ``--robot dual_franka`` 及相应任务卡和配置。按机器人部署文档配置 Env/VLA
端点；含 ``vla_grasp`` 的单臂任务卡需要 ``--vla-endpoint``。回放在驱动初始化
（可能复位机器人）前、执行前请求确认，结束后由操作员判定任务是否成功。
本模式不能与 ``--interactive``、``--dashboard``、``--explore`` 同用。

像素或深度无效、目标超出工作空间、运动过大、原语执行失败都会停止回放。
左臂工作空间检查使用左臂基座坐标系。RPC 成功不代表任务成功，最终结果由操作员
确认。``flash_outcome.json`` 和 ``flash_recipe.jsonl`` 保存结果和已下发动作。

仅支持审阅后的 v1 任务卡；实验版 v2 的分阶段监督回放、阶段拼接、参考图选点及
历史点复用不属于本接口。运行需要对应机器人环境及操作员验证；离线单测不能证明
真机任务成功。

可选的 Agent 定位回退
------------------------------------------

回放命令增加 ``--grounding-agent-model provider:model``，即可在 Molmo 失败后
尝试一次 Agent 定位。``codex:model`` 使用已有的 Codex CLI 登录及 provider
配置，但使用独立的临时配置目录，不带入用户的 MCP 和插件。启动后若生效配置
仍包含启用的 MCP，发送图片前就会停止。仅使用系统钥匙串登录时，需要改用文件
登录或设置 ``CODEX_API_KEY``。API 模型使用已有的 API 模型工厂。
可通过 ``--grounding-agent-base-url``
覆盖模型端点。所选模型需要支持图像输入及结构化输出。

单臂和双臂共用这条路径：目标未找到、像素或深度无效、Molmo 请求失败后，刷新
观测并让 Agent 在同一目标部位重新选点。Agent 只返回像素坐标，不接入机器人
工具；选点仍须通过深度反投影、工作空间及运动幅度检查。再次失败则停止回放，
不执行本次运动。下次平移仍先用 Molmo。取消不会触发回退；配置错误及工作空间、
运动幅度检查失败会直接停止。

默认关闭此功能；它用于实时回放，不用于离线生成任务卡。每次尝试写入对应步骤的
``flash_grounding.json``。Agent 请求可能产生模型费用，请求超时为 90 秒；
Flash 规划器的 token 统计不包含这些感知请求。

停止运行
------------

在终端按 Ctrl+C 请求结束 RPent；紧急情况下使用硬件急停按钮。确认机械臂与夹爪状态后，再按控制节点的停机流程关闭服务。
