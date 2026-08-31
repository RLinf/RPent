Franka
======

RPent 可以通过 RLinf ``RealWorldEnv`` worker 控制单台 Franka 机械臂。
这个开发分支将运行配置保存在 RPent 中，并通过 ``franka`` extra 安装所需的
RLinf/OpenPI 分支。不需要单独的 RLinf checkout、虚拟环境或安装步骤。

安装
----

在 RPent 仓库根目录运行：

.. code-block:: bash

	uv sync --extra franka

该命令按照 ``pyproject.toml`` 将自定义 RLinf Franka 分支和
``rlinf-openpi`` 安装到 ``.venv``。

开发配置
--------

仓库中的值是开发默认值，启用机械臂运动前必须逐项检查：

* ``robots/franka/robot_config.yaml`` 仅包含机器身份（机器人 IP、相机序列号、
  夹爪）和工作空间几何（target/reset pose、安全边界）。
* ``robots/franka/config.py`` 存放开发者默认值（primitive 控制、action scale、
  容差、相机处理），覆盖在 RLinf 自身 dataclass 默认值之上。
* ``robots/franka/calibration/hand_eye_calibration.json`` 包含感知工具使用的
  hand-eye calibration。

请替换 robot config 中的 ``ROBOT_IP``、``CAMERA_SERIAL_WRIST`` 和
``CAMERA_SERIAL_EXTERNAL``。不要复用其他工作空间的位姿、边界、序列号或标定。

RPent 会将该机器人配置转换成内部 RLinf cluster 和环境对象。如需使用其他文件，
请传入 ``--robot-config /path/to/robot_config.yaml``。用户不再需要接触 Hydra
或 RLinf 配置流程。

标定（Calibration）
----------------------

Hand-eye calibration 使用 ROS `easy_handeye
<https://github.com/IFL-CAMP/easy_handeye>`_ 完成。它为每个相机生成一个 YAML
（外部相机为 eye-on-base，腕部相机为 eye-on-hand），默认保存在
``~/.ros/easy_handeye/`` 下。

RPent 读取一个 JSON bundle（``hand_eye_calibration.json``），其中包含每个相机的
``source_name``、``parameters`` 和 ``transformation``。生成方式是从每个
``easy_handeye`` YAML 中拷贝这些字段。

bundle 位置可通过 ``--calibration-path`` 配置（默认
``~/.ros/easy_handeye/hand_eye_calibration.json``）。相机内参在运行时从
``camera_meta.json`` 捕获，不在 bundle 中。

使用本地 RLinf checkout
------------------------

如果需要修改 RLinf 并直接测试，而不重新安装 ``franka`` extra，请在启动 Ray
前将 checkout 放到 ``PYTHONPATH`` 最前面，并导出 ``RLINF_REPO_PATH``，让 RPent
的环境子进程加载同一 checkout：

.. code-block:: bash

	export RLINF_REPO_PATH=/path/to/RLinf
	export PYTHONPATH=$RLINF_REPO_PATH:${PYTHONPATH:-}
	export RLINF_NODE_RANK=0
	ray stop --force
	ray start --head

	uv run --extra franka rpent --env franka --task-id 0 \
	  --planner api --model anthropic:claude-sonnet-4-5

``robots/franka/env_server.py`` 会在启动时读取 ``RLINF_REPO_PATH`` 并把它放到
``sys.path`` 最前面。在 ``ray start`` 前导出 ``PYTHONPATH``，可让 Ray worker
使用同一份源码。修改路径后必须重启 Ray；已经运行的 Ray 进程会继续使用启动时
捕获的环境。

可以用以下命令确认实际加载位置：

.. code-block:: bash

	PYTHONPATH=$RLINF_REPO_PATH:$PYTHONPATH \
	  uv run --extra franka python -c \
	  'import rlinf; print(rlinf.__file__)'

若要恢复使用 ``.venv`` 中安装的版本，请取消设置 ``RLINF_REPO_PATH``。

启动 Ray
--------

Ray 启动时会捕获环境变量，因此必须先设置 node rank：

.. code-block:: bash

	export RLINF_NODE_RANK=0
	ray stop --force
	ray start --head

运行 smoke test
---------------

任务 ``0`` 用于测试保守的解析式运动和夹爪 primitives：

.. code-block:: bash

	uv run --extra franka rpent --env franka --task-id 0 \
	  --planner api --model anthropic:claude-sonnet-4-5

RPent 使用当前解释器启动 ``robots/franka/env_server.py``，加载 RPent robot
config 并生成内部 RLinf adapter config，然后连接 Ray，等待
``healthz``，并将初始状态记录为 step ``0``。

VLA 任务
--------

任务 ``1`` 会暴露 ``vla_grasp``。单臂 Franka 当前需要兼容的外部 VLA 服务，
其观测布局、动作布局、checkpoint 和归一化统计必须与当前 Franka 训练配置一致：

.. code-block:: bash

	uv run --extra franka rpent --env franka --task-id 1 \
	  --vla-endpoint http://VLA_HOST:PORT \
	  --planner api --model anthropic:claude-sonnet-4-5

未设置 ``--vla-endpoint`` 时，解析式运动和夹爪工具仍然可用，但
``vla_grasp`` 会抛出运行时错误。LIBERO VLA server 与物理 Franka 不兼容。

连接外部环境服务
----------------

连接已经运行的 Franka 环境服务：

.. code-block:: bash

	uv run --extra franka rpent --env franka --task-id 0 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --planner api --model anthropic:claude-sonnet-4-5

工具与状态产物
--------------

Franka 扩展提供 ``view_env_state``、``view_camera_meta``、``move_delta``、
``rotate_delta``、``open_gripper``、``close_gripper`` 和 ``vla_grasp``。
每个会改变环境的工具都会在 RPent 中央 ``EnvState`` 中保存机器人状态、腕部与
外部 RGB 图像、可选的对齐深度数组和相机元数据。

安全要求
--------

操作员必须留在急停按钮旁。先使用极小动作验证任务 ``0``，再尝试抓取。
当相机与状态结果不一致、目标运动未达到，或任何标定存在疑问时，应立即停止。
