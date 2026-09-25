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

.. note::

	以下的步骤只会安装 Python 侧依赖（自定义的 RLinf Franka 分支和 ``rlinf-openpi``），并 **不会** 构建机械臂真正需要的机器人控制栈。在安装 RPent 之前，请先按照 RLinf 单臂 Franka 指南配置控制节点：检查 Franka 固件兼容性、安装实时内核、选择夹爪（Franka hand 或 Robotiq 2F-85/2F-140）与相机，并构建 ROS 控制相关软件包（ROS Noetic、与固件匹配的 libfranka 和 franka_ros，以及 serl_franka_controllers）。参见 `RLinf 单臂 Franka 指南 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/franka.html>`_。

克隆 RPent 并安装 Python 依赖。若已有仓库，进入仓库后执行 ``uv sync``：

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent
   uv sync --extra franka

该命令会把自定义的 RLinf Franka 分支和 ``rlinf-openpi`` 安装到 ``.venv``。

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

* ``robots/franka/config/example.yaml``，包含机器人身份（机器人 IP、相机序列号、夹爪）、工作空间几何（目标/复位位姿、安全边界）和 easy_handeye YAML 映射（见上方标定说明）。

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

先按 :doc:`configure_planner` 配置规划器与模型服务。

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

停止运行
------------

在终端按 Ctrl+C 请求结束 RPent；紧急情况下使用硬件急停按钮。确认机械臂与夹爪状态后，再按控制节点的停机流程关闭服务。
