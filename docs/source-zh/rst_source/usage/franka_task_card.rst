Franka Task Card
================

单臂和双臂共用生成、回放实现，复用现有 ``--planner task_card`` 和
``RobotSpec.replay_card``。真机使用带版本的独立 JSON 卡片，包含动作列表及
每步视觉 anchor。LIBERO 的 XY 回放、抓取修正与真机坐标约定不同，因此不复用
其执行算法，也不直接加载仿真卡片。

生成
----

用普通 Franka planner 记录一次成功尝试，保留完整 ``states.json``、RGB-D、
相机元数据，以及录制时使用的机器人 YAML 和标定文件。按源 step 为每个
``move_delta`` 和 ``rotate_delta`` 编写 annotations JSON；录制 Agent 可以
根据可见推理、工具调用和图片编写，由操作员核对。以下 step 编号仅为示例：

.. code-block:: json

   {
     "3": {"intent": "接近杯子", "phrase": "the red cup", "camera": "base"},
     "5": {"intent": "调整腕部视角", "rotation_mode": "relative"}
   }

双臂支持 ``base``、``d455`` 定位相机；单臂支持 ``wrist``、``third_person``。
生成时 Molmo 在动作前的源图片定位，深度和标定转换得到三维点，再计算动作后
TCP 相对此点的偏移。回放时在当前图像重新定位并重算 ``move_delta``；两臂使用
最新版统一的 ``right_base`` 运动坐标。左臂 workspace 检查再转换到 ``left_base``。
缺少明确坐标标签的旧版双臂记录会被拒绝，需要重新录制。

旋转 ``relative`` 保留原 delta，但要求起始姿态与录制时相差不超过 0.15 rad；
``fixed`` 保存目标四元数，根据当前姿态通过旋转组合计算 delta。依赖物体方向
的旋转不能标成固定姿态；目前需要合适的 VLA primitive，尚未实现方向估计。
生成器不会根据位移数值猜测动作目的，缺少标注时拒绝生成。

.. code-block:: bash

   python -m robots.franka.task_card.generate \
     --robot dual_franka --task dual_franka_t1 \
     --run-dir logs/SOURCE_RUN --annotations annotations.json \
     --robot-config /path/to/robot.yaml \
     --calibration-path /path/to/hand_eye_calibration.json \
     --molmo-endpoint http://MOLMO_HOST:PORT \
     --destination memory/dual_franka/task_card/cup.json

生成过程只连接 Molmo，不连接机器人。终端要求人类确认源尝试成功。
Agent 明确报告 failure/stuck、回放结果失败、动作返回失败、不支持的动作或
step 编号断裂都会拒绝生成。连续 step 无法证明未发生漏记的硬件动作，仍需
核对 transcript。源目录保持不变；目标文件已存在时拒绝覆盖。

回放
----

.. code-block:: bash

   rpent --robot dual_franka --task-id 1 --planner task_card \
     --task-card memory/dual_franka/task_card/cup.json \
     --robot-config /path/to/robot.yaml \
     --calibration-path /path/to/hand_eye_calibration.json \
     --molmo-endpoint http://MOLMO_HOST:PORT \
     --vla-endpoint http://VLA_HOST:PORT

单臂生成使用 ``--robot franka --task franka_t1``，回放使用
``--robot franka --task-id 1``。单臂 VLA 仍需外部 endpoint；双臂可使用原有
本地 VLA 配置。机器人、Ray 和控制器部署要求沿用各自文档。

回放有三处终端交互：初始化环境前确认人工复位并允许运动（EnvClient 构造会
reset）；初始化后再次确认开始；完成后人类判定 success/failure。只有人类
success 才令 ``solved()`` 为真。动作或定位失败则直接失败，停止后续步骤。
左右臂依照源 step 顺序调用；双臂 VLA 保持一次整体调用并重新推理，不新增
并行调度器。等待人工确认期间，回放器不下发动作。

输出保留 ``states.json``、图片和 transcript，并增加：

* ``task_card_outcome.json``：人类判定、错误、源 step 到回放 step 的映射。
* ``task_card_recipe.jsonl``：本次实际尝试的 primitive 参数。

失败也保存上述证据；失败 JSONL 不能作为成功 recipe。机器人配置和标定的
hash 必须与生成时一致。解析式位移需在各臂 workspace 内且单步不超过 0.20 m，
旋转不超过 0.35 rad。这些检查不提供碰撞检测或硬件急停。

本次使用终端交互，暂不支持 ``--dashboard`` 或 ``--interactive``。上游新增的
双臂 ``--explore`` 可单独使用，本回放流程不自动发布 memory。双臂 VLA 支持
``vla_right_grasp``、``vla_handoff``、``vla_left_place``，单臂仍用 ``vla_grasp``。
自动化测试使用离线 mock；实际机械臂、
Molmo 和标定组合仍需现场验证。
