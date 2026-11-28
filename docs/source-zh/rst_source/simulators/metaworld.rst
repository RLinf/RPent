MetaWorld
=========

在 `MetaWorld <https://github.com/Farama-Foundation/Metaworld>`_ 中用 RPent 运行桌面操作任务。视觉规划器根据 RGB-D 观测，调用脚本化的笛卡尔运动工具控制 MuJoCo 中的 Sawyer 机械臂，不需要 VLA 模型权重。

概览
----

安装前先确认支持的动作模型、规划器、任务和硬件。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      默认使用脚本化的笛卡尔运动工具；也可以接入兼容的外部 WAM worker。

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``、``gemini_cli``、``opencode``

   .. grid-item-card:: 任务

      MetaWorld 3.0.0 MT10 中的十种任务，每种以一个固定的 MT1 episode 运行。

   .. grid-item-card:: 硬件

      Linux、Python 3.11；支持 EGL 的 GPU，或用 OSMesa 在 CPU 上渲染。

任务
~~~~

每次运行用固定种子跑一种任务的一个 episode，结果不能当作 MT10 或 MT50 基准成绩。

.. list-table::
   :header-rows: 1

   * - 任务
     - 传给规划器的任务指令
   * - ``reach-v3``
     - Move the gripper to the visible red goal marker.
   * - ``push-v3``
     - Push the puck to the visible goal marker on the table.
   * - ``pick-place-v3``
     - Pick up the puck and place it at the visible goal marker.
   * - ``door-open-v3``
     - Pull the handle to open the door.
   * - ``drawer-open-v3``
     - Pull the drawer open using its handle.
   * - ``drawer-close-v3``
     - Push the drawer closed.
   * - ``button-press-topdown-v3``
     - Press the button down from above.
   * - ``peg-insert-side-v3``
     - Insert the peg into the side-facing hole.
   * - ``window-open-v3``
     - Slide the window open using its handle.
   * - ``window-close-v3``
     - Slide the window closed using its handle.

观测与动作
~~~~~~~~~~

下表列出规划器看到的观测、推进仿真的工具和任务结果的判定方式。

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - ``view_env_state`` 返回最新的 480 x 480 RGB 图像、末端位置、夹爪开度、任务指令、仿真步数 ``steps`` 和快照编号 ``step``。``back_project`` 从已记录的深度图读取世界坐标。物体和目标的坐标不提供给规划器。
   * - 动作
     - ``move_to`` 把末端移向世界坐标 XYZ 目标，``set_gripper`` 原地开合夹爪，手腕朝向固定。传入 ``--am-endpoint`` 后，``wam_act`` 执行外部 WAM worker 预测的动作片段。
   * - 奖励与成功判定
     - 以 MetaWorld 原生成功信号为准，写入 ``result.json`` 的 ``success``。episode 达到步数上限（默认 500）或原生成功判定触发时结束。规划器调用 ``finish``、``move_to`` 返回 ``reached`` 都不代表任务成功。
   * - 任务指令
     - 所选任务的固定指令，见上方任务表。

.. dropdown:: 运动与反投影细节

   ``move_to(target_xyz, gripper, max_steps=50)`` 闭环修正末端 XYZ 位置，每步最多移动 0.01 米（MetaWorld 原生动作缩放）。每次调用最多 100 步，返回 ``reached``、``distance_m`` 和 ``steps``。末端距目标不到 5 毫米、episode 结束或达到 ``max_steps`` 时停止。运动不做避障。

   夹爪参数只在实际执行的仿真步上生效：+1 闭合，-1 张开。如果末端已在目标 5 毫米以内，``move_to`` 执行 0 步，夹爪保持不变。原地开合夹爪用 ``set_gripper(gripper, steps=10)``，每次最多 20 步。

   原生成功容差可能宽于 5 毫米，所以 episode 可能在 ``move_to`` 返回 ``reached`` 前就已成功并结束。

   调用 ``back_project`` 时，相机填 ``primary``，选区可写成 ``{"image_shape": [480, 480], "points_rc": [[240, 240]]}``，并传入 ``view_env_state`` 返回的 ``step``。``samples_xyz`` 按输入顺序给出以米为单位的世界坐标。深度无效或落在远裁剪平面上的像素对应 ``null``；所有选中像素都没有有效深度时，结果为 ``status: "need_observation"``。像素坐标超出图像范围时参数校验失败。观测工具不推进仿真。

安装
----

创建独立的 Python 3.11 环境，安装 ``metaworld`` extra。MetaWorld 的任务资产随软件包安装，无需另外下载：

.. code-block:: bash

   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[metaworld]"

.. note::

   这个 extra 固定 MuJoCo 3.14.0，与 ``libero``、``libero-pro`` 和 ``libero-plus`` 要求的 MuJoCo 3.3.0 冲突。

运行一个任务
------------

按 :doc:`../guides/configure_planner` 配置规划器凭据，设置模型名后运行一个任务：

.. code-block:: bash

   export PLANNER_MODEL=openai/your-model
   rpent --robot metaworld --task reach-v3 --seed 0 \
     --planner api --model "$PLANNER_MODEL" \
     --output-dir ./runs/metaworld-reach

``rpent`` 会启动环境服务、运行规划器，结束后关闭环境服务。常用参数：

- ``--task``：上表中的任务之一，默认 ``reach-v3``。
- ``--seed``：0 到 4294967295 之间的整数，默认 0。
- ``--camera``：默认 ``corner2``，也可选 ``corner``、``corner3``、``topview``、``behindGripper`` 和 ``gripperPOV``。
- ``--max-episode-steps``：episode 步数上限，默认 500。
- ``--cuda-device``：EGL 渲染使用的物理 CUDA GPU，例如 ``--cuda-device 0``。

MetaWorld 只使用本地记忆，初始为空；用 ``--memory-dir`` 指定记忆目录。不支持 ``--memory-profile hf``、探索模式和 Flash Mode。

要使用 Dashboard，把 ``--task reach-v3 --seed 0`` 换成 ``--dashboard``，再用 ``/rpent-task reach-v3 0`` 领取任务。Dashboard 为每个任务启动新的环境服务，因此不能与 ``--env-endpoint`` 同时使用。详见 :doc:`../guides/dashboard`。

.. dropdown:: 独立仿真环境与已运行的环境服务

   要在另一个 Python 环境中运行仿真，传入 ``--sim-python /path/to/python``。这个解释器需要已从本仓库安装 ``.[metaworld]``。

   要复用已运行的环境服务，启动时使用与本次运行相同的任务、种子、相机和步数上限：

   .. code-block:: bash

      python -m robots.metaworld.env_server --task reach-v3 --seed 0 \
        --port 18765 --video-dir ./runs/standalone/videos

   然后用相同的任务和种子连接：

   .. code-block:: bash

      rpent --robot metaworld --task reach-v3 --seed 0 \
        --env-endpoint http://127.0.0.1:18765 \
        --planner api --model "$PLANNER_MODEL"

   连接时会重置 episode，所以每次运行应独占一个环境服务。``rpent`` 不会关闭它连接的环境服务；渲染设备和视频目录由启动该服务的人设置。

查看结果
--------

每次运行把记录写入 ``--output-dir``：

- ``result.json``：``success`` 是 MetaWorld 原生成功判定，``planner_finish`` 是规划器调用 ``finish`` 时的报告。环境结果不可用时，``success`` 为 ``null``。文件还记录任务、种子、相机、步数上限和规划器预算。
- ``states.json``，以及每次工具调用后保存的 RGB 图像和深度图。
- ``videos/episode_0000.mp4``：由 ``rpent`` 启动环境服务时录制，包含初始帧和每个已执行仿真步后的一帧。
- ``environment.json``：仿真进程的依赖版本、步数上限、动作缩放和渲染设置。
- ``env_server.log``：环境服务日志。

外部动作模型
------------

要让规划器调用已运行、支持 MetaWorld 的 WAM worker，加上 ``--am-endpoint``：

.. code-block:: bash

   rpent --robot metaworld --task reach-v3 --seed 0 \
     --planner api --model "$PLANNER_MODEL" \
     --am-endpoint http://HOST:PORT \
     --output-dir ./runs/metaworld-wam

规划器可额外调用 ``wam_act(max_chunks=1)``，每次执行最多 ``max_chunks`` 个预测动作片段（上限 20）。``move_to`` 和 ``set_gripper`` 仍可使用。RPent 在启动环境服务前先检查 worker，Dashboard 中 worker 状态显示为 ``ENV`` 旁边的 ``WAM``。

worker 需要自行启动和关闭，并使用针对 MetaWorld 训练的模型；RPent 不附带这类模型，LIBERO 或 RoboTwin 的模型权重也不能直接接入。

.. dropdown:: WAM worker 要求

   worker 需要实现 :doc:`../development/add_action_model` 中的 WAM 接口，并报告 embodiment ``metaworld_sawyer_4d`` 和动作空间 ``metaworld.cartesian.delta.v1``。它只能请求 ``primary`` 相机、``eef_position``\ （3 个值）和 ``gripper_opening``\ （1 个值）。

   每个请求包含 ``images.primary`` 下以左上角为原点的 RGB 图像、``state.eef_position`` 下以米为单位的末端世界坐标、``state.gripper_opening`` 下形状为 ``[1]`` 的归一化夹爪开度，以及任务指令。worker 返回有限的 ``[N, 4]`` 动作，取值在 ``[-1, 1]`` 内。前三维是按 0.01 米缩放的世界坐标 XYZ 增量，第四维是夹爪命令（+1 闭合，-1 张开）。控制频率为 80 Hz，手腕朝向固定。

   worker 不兼容时，在启动环境服务前报错；动作片段无效时，在执行任何一步前报错。取消操作或 episode 结束会停止后续动作。每次运行或每个 Dashboard 任务各自绑定一个 worker 会话，任务结束时释放。

故障排查
--------

先查看输出目录中的 ``env_server.log``。

- **没有支持 EGL 的 GPU，环境服务无法启动。** 安装 OSMesa 库，并在运行 ``rpent`` 前执行 ``export MUJOCO_GL=osmesa``；RPent 会沿用你设置的后端。
- **渲染用错了 GPU。** 用 ``--cuda-device 0`` 指定物理 GPU 序号。如果环境变量中已设置 ``MUJOCO_EGL_DEVICE_ID``，以它为准，``--cuda-device`` 不生效。
- **使用** ``--env-endpoint`` **时连接失败。** 启动环境服务时使用与本次运行相同的 ``--task``、``--seed``、``--camera`` 和 ``--max-episode-steps``。
- **使用** ``--am-endpoint`` **时，环境服务启动前就报错。** worker 在 30 秒内没有响应，或报告的 embodiment 或动作空间不匹配；见上方“WAM worker 要求”。
