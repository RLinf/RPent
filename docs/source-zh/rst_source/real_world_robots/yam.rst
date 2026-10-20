YAM
===

.. Product: https://i2rt.com/products/yam-box

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/rpent/yam_box.png
   :alt: YAM 双臂整机，包括 YAM Box、双臂和三路相机
   :figclass: rpent-robot-figure
   :align: center

   用于真机实验的 YAM 双臂系统。

RPent 通过 RLinf 的机器人接口控制 YAM 双臂系统。任务视频见 :doc:`YAM 演示 <../real_world_demos/yam>`。

.. _id1:
.. _yam-installation:

安装
----

运行环境使用 Linux、Python 3.11。本指南按两台机器部署：

* **控制机**：连接双臂的 CAN 接口和三路 RealSense 相机，运行 ENV 服务及 RPent。
* **推理机**：使用 NVIDIA GPU 运行 Pi0.5 VLA 服务。

两台机器使用同一份 RPent 代码和相同的 RLinf 提交。先按\ :doc:`../get_started/quickstart`\ 安装 uv，再在两台机器上分别准备环境：

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git /path/to/RPent
   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate

   git clone --branch main https://github.com/RLinf/RLinf.git /path/to/RLinf
   git -C /path/to/RLinf checkout --detach 196d40b6dec6f7a4063edee31cfeaa6ca0ee435a

**控制机安装**

在控制机的 RPent 仓库根目录，同时安装 RPent、RLinf、i2rt SDK 和对应的构建依赖：

.. code-block:: bash

   cd /path/to/RPent
   uv pip install --torch-backend cpu \
     --build-constraints /path/to/RLinf/requirements/embodied/envs/yam-build-constraints.txt \
     -e '.[yam]' -e /path/to/RLinf \
     -r /path/to/RLinf/requirements/embodied/envs/yam.txt
   uv pip check

**推理机安装**

在推理机先运行 RLinf 的安装脚本，安装 YAM 和 OpenPI 依赖，再安装 RPent：

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh embodied --model openpi --env yam \
     --venv /path/to/RPent/.venv --install-rlinf
   cd /path/to/RPent
   uv pip install -e '.[yam]'
   uv pip check

已安装系统依赖时，可在安装命令末尾加上 ``--no-root``。

新开终端时，进入对应机器的 RPent 仓库并激活 ``.venv``。

.. _yam-calibration:

标定
----

本指南使用三路 D405 的标准 YAM 双臂套装，双臂基座间距为 0.61 米。相机和双臂基座外参采用 `i2rt 官方整站模型参数 <https://github.com/i2rt-robotics/i2rt/blob/47fee5e7dec4e30ca054f798bda1c8894b465ed2/i2rt/robot_models/station/yam_station_linear_4310_d405/README.md>`_，无需另行采集和求解。

参数配置可直接使用 ``robots/yam/config/yam_station_d405.json``。将机器人配置中的外参路径设为：

.. code-block:: yaml

   extrinsics_path: /path/to/RPent/robots/yam/config/yam_station_d405.json

坐标系使用左臂基座 ``left_base``。调整双臂间距或相机安装位置后，需更新对应外参。

.. _id2:
.. _yam-robot-config:

机器人配置
----------

在 ``robots/yam/config/example.yaml`` 中按真机实际情况配置以下项目：

* ``robot.left_follower/right_follower``：左右臂 CAN 通道、夹爪类型、重力补偿和控制参数。
* ``cameras``：``top``、``left``、``right`` 三路相机的实际序列号。``top`` 为固定相机，``left/right`` 为腕部相机；示例使用 640x480、30 Hz。
* ``extrinsics_path``：上面的外参 JSON。
* 桌面几何、关节限位与碰撞设置：按真机环境的实测值填写。
* ``reset`` 和 ``park_on_close``：分别设置复位姿态和待机姿态；每种姿态均需填写双臂实测关节位置 ``left_qpos/right_qpos``，每侧各 7 维。

配置好两种姿态后，启用 ``reset.enabled`` 和 ``park_on_close.enabled``。待机姿态未配置时，ENV 无法启动。不要照搬其他真机的关节位置。

如需使用其他文件，请传入 ``--robot-config /path/to/robot_config.yaml``。任务 ID、任务指令、seed 和步数上限通过命令行指定。

.. _yam-tasks:

任务选择
--------

主要示例使用任务 ``1``。任务 ``0`` 用于规划器驱动的基本动作检查，包括小幅平移、腕部旋转和夹爪动作，不需要 VLA。任务通过 ``--task-id`` 选择，左右方向以顶视相机画面为准：

.. list-table::
   :header-rows: 1

   * - ID
     - 内容
     - 是否需要 VLA
   * - 0
     - 规划器驱动的基本动作检查
     - 否
   * - 1
     - 百事放左侧袋子，可口可乐放右侧袋子
     - 是
   * - 2
     - 可口可乐放左侧袋子，百事放右侧袋子
     - 是

任务 ``1`` 和 ``2`` 均需分类三瓶饮料，还要求移开碗、取出被遮挡的勺子，将白勺放入白碗、粉勺放入粉碗，并将碗放回原标记位置：白碗在左、粉碗在右。

以下示例使用任务 ``1``、seed ``0``、步数上限 ``1000``。ENV 与 RPent 需使用相同设置。

.. _yam-vla:

VLA 配置
--------

在推理机的 RPent 环境中，下载 `YAM clean-desk checkpoint <https://huggingface.co/RLinf/RLinf-Pi05-RealWorld-DualYAM-Clean_Desk>`_：

.. code-block:: bash

   cd /path/to/RPent
   hf download RLinf/RLinf-Pi05-RealWorld-DualYAM-Clean_Desk \
     --revision 15a037b444fa36618a56729fd55a69e09c9983f5 \
     --local-dir ./models/pi05-yam-clean-desk
   export YAM_MODEL_PATH="$PWD/models/pi05-yam-clean-desk"

保留下载目录中的 ``model.safetensors`` 和 ``norm_stats.json``。使用自己训练的模型时，也可提供包含 ``full_weights.pt`` 的 RLinf checkpoint 目录，并指定对应的归一化统计文件。

部署到不同真机时，建议单独采集数据并重新训练模型。

VLA 服务使用 ``pi05_yam_joint`` 配置，相机顺序为 ``top/left/right``，每次调用执行前 5 步动作。

.. _yam-training:

数据采集与训练
--------------

示教数据采集见\ `RLinf YAM 数据采集 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/yam.html>`_，模型训练与评估见\ `YAM Pi0.5 训练与评估 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/yam_openpi.html>`_。

.. _yam-codex-isolation:

Codex 配置
----------------

使用 Codex 时，先按\ :ref:`planner-codex-isolation`\ 配置独立的运行目录并登录。

.. _id3:
.. _yam-services:

启动双机服务
------------

在两台机器分别激活安装好的 ``.venv``，按以下分工启动服务：

* **控制机**：运行 RPent、ENV 和人工操作命令，连接双臂 CAN 接口与三路相机。
* **推理机**：运行 Pi0.5 VLA 模型服务。

控制机通过下文的 SSH 端口转发连接推理机的 VLA 服务。

控制机分别使用三个终端运行 ENV、人工操作命令和 RPent。每个终端先执行以下准备命令，并保持任务设置一致：

.. code-block:: bash

   cd /path/to/RPent
   source .venv/bin/activate
   export TASK_ID=1

**1. 控制机：启动 ENV**

在 ENV 终端运行：

.. code-block:: bash

   python -m robots.yam.env_server --robot-config robots/yam/config/example.yaml \
     --task-id "$TASK_ID" --seed 0 --max-episode-steps 1000 \
     --transport socket --host 127.0.0.1 --port 8110

**2. 推理机：启动 VLA**

在推理机激活安装好的环境，设置 ``YAM_MODEL_PATH``，然后运行：

.. code-block:: bash

   python -m rpent.robots.components.pi05_vla_server \
     --embodiment yam --model-path "$YAM_MODEL_PATH" \
     --norm-stats-path "$YAM_MODEL_PATH/norm_stats.json" \
     --transport socket --host 127.0.0.1 --port 8220

在控制机另开一个终端，通过 SSH 转发 VLA 端口；替换下面的推理机账号和地址：

.. code-block:: bash

   ssh -N -L 8220:127.0.0.1:8220 INFERENCE_USER@INFERENCE_HOST

保持 SSH 连接，控制机使用 ``socket://127.0.0.1:8220`` 访问 VLA 服务。

.. _yam-start-episode:

**3. 控制机：初始化并开始回合**

现场准备好后，在人工操作终端运行：

.. code-block:: bash

   python -m robots.yam.operator_control --robot-config robots/yam/config/example.yaml \
     --endpoint socket://127.0.0.1:8110 --event status

首次运行 ``status`` 会启动相机并连接双臂。

先将夹爪中的物体放稳或取下，确认复位路径安全后，执行已配置的复位动作，再读取当前回合 ID：

.. code-block:: bash

   python -m robots.yam.operator_control --robot-config robots/yam/config/example.yaml \
     --endpoint socket://127.0.0.1:8110 --event reset_pose
   python -m robots.yam.operator_control --robot-config robots/yam/config/example.yaml \
     --endpoint socket://127.0.0.1:8110 --event status

恢复桌面并确认现场就绪，将下面的 ``CURRENT_ID`` 替换为刚输出的 ``episode_id``：

.. code-block:: bash

   python -m robots.yam.operator_control --robot-config robots/yam/config/example.yaml \
     --endpoint socket://127.0.0.1:8110 --episode-id CURRENT_ID \
     --event start --note '场景已恢复，可以开始本回合'

.. _id5:
.. _yam-diagnostics:

手动技能测试
------------

测试前停止 Agent，确认现场后，用当前回合 ID 执行上面的 ``start`` 命令。

**基础动作检查**

.. code-block:: bash

   python -m robots.yam.manual --task-id "$TASK_ID" --mode primitives \
     --env-endpoint socket://127.0.0.1:8110 --max-episode-steps 1000

每行输入一个 JSON，例如 ``{"tool":"status"}``。支持 ``move_to``、``rotate_wrist``、``set_gripper`` 和 ``release``，通过 ``arguments`` 传参。

**VLA 预测与执行检查**

.. code-block:: bash

   python -m robots.yam.manual --task-id "$TASK_ID" --mode vla \
     --env-endpoint socket://127.0.0.1:8110 --max-episode-steps 1000 \
     --vla-endpoint socket://127.0.0.1:8220 --output-dir /path/to/diagnostics

先输入 ``{"tool":"infer"}`` 保存预测，此时机械臂不动。人工检查后，再输入 ``{"tool":"execute","arguments":{"use_length":5}}``，执行前五步；每份预测最多执行一次。

预测超过 30 秒，或回合、机器人状态改变时，需要重新预测。执行结果不确定时，先检查现场，不要重复执行。输入 ``quit`` 退出测试并发送停止指令。

退出测试后，确认现场并读取当前回合 ID，再执行一次 :ref:`start 命令 <yam-start-episode>`，然后启动 RPent。

.. _dashboard:
.. _yam-run:

运行一个任务
------------

完成机器人配置和双机服务启动，并按\ :doc:`../guides/configure_planner`\ 配置规划器。使用 Codex 前，先完成\ :ref:`planner-codex-isolation`。将 ``MODEL_ID`` 替换为已配置的模型，并选择该模型支持的推理档位。在控制机的 RPent 终端运行：

.. code-block:: bash

   rpent --robot yam --task-id "$TASK_ID" --seed 0 \
     --planner codex --model MODEL_ID --reasoning-effort low \
     --robot-config robots/yam/config/example.yaml \
     --env-endpoint socket://127.0.0.1:8110 \
     --vla-endpoint socket://127.0.0.1:8220 \
     --max-episode-steps 1000 \
     --memory-profile local --memory-dir ./memory/yam

任务 ``0`` 只使用基本动作；运行该任务时，将 ENV 和 RPent 的任务 ID 均设为 ``0``，并将 RPent 的 VLA 地址参数替换为 ``--without-vla``。

**使用网页**

在同一条命令末尾追加 ``--dashboard --dashboard-host 127.0.0.1 --dashboard-port 8090``。从自己的电脑访问时，先转发网页端口：

.. code-block:: bash

   ssh -N -L 8090:127.0.0.1:8090 CONTROL_USER@CONTROL_HOST

打开 ``http://127.0.0.1:8090``，发送 ``/rpent-task 1 0``。选择其他任务时，同步修改 ENV 和网页中的任务 ID、seed。

切换任务前，按退出流程关闭当前 ENV，再用新任务设置启动 ENV；VLA 可以继续使用。网页操作见\ :doc:`../guides/dashboard`。

.. _id4:
.. _yam-results:

查看结果
--------

运行日志保存在 ``logs/`` 下本次新建的目录，也可通过 ``--output-dir`` 指定新目录。重点查看：

* ``result.json``：环境成功判定、规划器结束请求和运行错误。
* ``transcript_*.json``、``run.log``：消息、工具调用和运行日志。
* ``states.json`` 及其引用的 RGB、深度、三维坐标和状态文件：每一步的观测和动作记录。
* ``episode_*.mp4``：保存的回合视频；探索时的逐会话状态位于 ``sessions/session_<NNN>/``。

现场确认任务完成后，先用 ``status`` 读取当前 ``episode_id``，替换下方的 ``CURRENT_ID``，再提交成功结果：

.. code-block:: bash

   python -m robots.yam.operator_control --robot-config robots/yam/config/example.yaml \
     --endpoint socket://127.0.0.1:8110 --episode-id CURRENT_ID \
     --event success --note '现场已确认本回合任务完成'

失败或中止时，分别使用 ``--event failure`` 或 ``--event abort``，并填写说明。

``eval_success=True`` 表示现场人员已确认当前回合成功。

.. _yam-tools:

工具与状态产物
--------------

YAM 提供 ``view_env_state``、``sample_world_xyz``、``query_world_map``、``render``、``status``，以及 ``move_to``、``rotate_wrist``、``set_gripper``、``release``。连接 VLA 后还提供 ``pi05_act``。

基本动作通过 ``arm=left/right`` 选择一条机械臂。``move_to`` 的 xyz 单位为米，坐标系为 ``left_base``，四元数顺序为 wxyz；``rotate_wrist`` 使用角度；夹爪 ``val=0`` 闭合、``val=1`` 打开。

开合夹爪使用 ``set_gripper`` 或 ``release``。

.. _yam-exploration:

探索模式（人工复位与判定）
--------------------------

在任务命令后追加 ``--explore``，即可探索并积累任务记忆。``--explore-attempts-per-session`` 设置每个会话的尝试次数，``--explore-sessions`` 设置会话数。

终端中需要输入反馈时再加 ``--interactive``；网页交互使用 ``--dashboard``，不能与 ``--interactive`` 同用。

每次尝试结束后，在人工操作终端确认成功、失败或中止。再次尝试前，先将夹爪中的物体放稳或取下，恢复场景，按需执行 ``reset_pose``，再用 ``status`` 读取当前回合 ID 并运行：

.. code-block:: bash

   python -m robots.yam.operator_control --robot-config robots/yam/config/example.yaml \
     --endpoint socket://127.0.0.1:8110 --episode-id CURRENT_ID \
     --event ready --note '场景已恢复，可以进行下一次尝试'

准备好场景并执行上面的命令后，在 RPent 交互界面输入 ``/continue`` 继续。等待人工确认时，任务保持暂停；已结束的任务需重新启动。

``/done``、``/success``、``/failure``、``/abort`` 请在人工操作终端通过 ``operator_control --command`` 执行，并填写当前回合 ID 和说明。``/done`` 表示场景已准备好。

.. _yam-memory:

日志与记忆
~~~~~~~~~~

任务经验保存在 ``--memory-dir`` 指定的目录，默认自动合并任务记忆。使用 ``--no-auto-merge-memory`` 可关闭自动合并。目录结构和记忆用法见\ :doc:`../guides/memory`。

.. _yam-shutdown:

停止与退出
----------

停止任务后，先将夹爪中的物体放稳或取下，确认返回待机姿态的路径安全，再在人工操作终端执行：

.. code-block:: bash

   python -m robots.yam.operator_control --robot-config robots/yam/config/example.yaml \
     --endpoint socket://127.0.0.1:8110 --event shutdown

ENV 返回 ``park_on_close`` 配置的待机姿态，确认到位后关闭电机输出。未能到位时，保留进程并检查现场，不要强行结束。ENV 正常退出后，再关闭 VLA 服务和 SSH 转发。
