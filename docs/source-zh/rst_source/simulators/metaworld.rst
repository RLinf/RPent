MetaWorld
=========

MetaWorld 接入通过 RGB-D 观测和有步数上限的笛卡尔控制操作 Sawyer 机械臂。
支持 MetaWorld 3.0.0 的 MT10 清单中的十种任务，分别作为固定 MT1 episode 运行。
视觉 planner 调用脚本动作原语，无需 VLA 或分割模型权重。

支持的任务
----------

* ``reach-v3``、``push-v3``、``pick-place-v3``
* ``door-open-v3``、``drawer-open-v3``、``drawer-close-v3``
* ``button-press-topdown-v3``、``peg-insert-side-v3``
* ``window-open-v3``、``window-close-v3``

安装与单次运行
--------------

在支持 EGL 的 Linux GPU 机器上使用独立 Python 3.11 环境：

.. code-block:: bash

   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[metaworld]"

按照 :doc:`../guides/configure_planner` 配置 planner 后运行：

.. code-block:: bash

   rpent --robot metaworld --task reach-v3 --seed 0 \
     --planner api --model "$PLANNER_MODEL" \
     --memory-dir ./memory/metaworld \
     --output-dir ./runs/metaworld-reach

将 ``PLANNER_MODEL`` 设为已配置的、带 provider 前缀的模型名称。
默认使用本地 memory，可以使用空目录；显式指定 ``--memory-profile hf``
会在参数校验阶段报错。
CLI 自动启动和清理自有环境服务。``--sim-python`` 可指定独立仿真解释器；
``--env-endpoint`` 可连接已有服务，但任务、seed、相机和步数限制必须匹配。
连接会重置固定 episode，因此应独占该服务。
已有服务由其启动者负责关闭，runner 不会停止它；需要录制时，在启动服务时指定 ``--video-dir``：

.. code-block:: bash

   python -m robots.metaworld.env_server --task drawer-open-v3 --seed 0 \
     --port 18765 --video-dir ./runs/standalone/videos

观测与控制
----------

``view_env_state`` 返回最近一次 480 x 480 RGB 图像、末端位置、
夹爪开度、任务指令和仿真步数。
``back_project(row, col)`` 使用对应深度图和相机标定，将可见表面像素转换为世界坐标（米）。
相机坐标轴为 x 向右、y 向上、z 向后。
内参对应像素中心，RGB-D 使用单采样渲染并关闭多重采样抗锯齿，以保证几何对齐。
越界像素、非正或非有限深度，以及远裁剪平面的背景像素会被拒绝。
两种观测工具均不推进物理仿真，也不更新原生帧历史。
即使后续动作的状态抓取失败，反投影仍使用已保存图像对应的深度与标定；
继续操作前应查看该快照记录的步数。

``move_to(target_xyz, gripper, max_steps=50)`` 根据当前末端位置闭环修正，
使用 MetaWorld 原生的 0.01 米动作缩放。每次最多 100 步，
返回是否到达、剩余距离和实际步数。手腕朝向固定。
运动中的夹爪参数 +1 表示关闭，-1 表示打开；
原地操作夹爪请使用 ``set_gripper(gripper, steps=10)``，每次最多 20 步。
零步结束的 move_to 不会改变夹爪。
动作不具备避障能力，并在 episode 结束时停止。到达一个位置不等于任务成功。

默认相机为 ``corner2``，``--camera`` 还可选择 ``corner``、
``corner3``、``topview``、``behindGripper`` 和 ``gripperPOV``。
默认 episode 上限为 500 步；到达步数上限或满足原生成功判定时结束。
任务的原生容差可能大于动作原语的位置容差。

结果与适用范围
--------------

``result.json`` 分别记录原生成功判定与 planner 的结束声明，
同时保存 seed、相机、episode 和 planner 预算。
``environment.json`` 保存仿真进程实际使用的依赖版本、原生步数限制、
动作缩放、渲染配置和视频路径。即使使用 ``--sim-python``，这些信息也来自仿真解释器。

``states.json`` 及 RGB、深度文件记录各工具调用结束后的状态。
自有服务以原生控制频率流式写入 ``videos/episode_0000.mp4``，
包含初始帧，以及每个已接受仿真步后的 RGB 帧。只读调用和被拒绝的动作不增加帧。
重置会关闭上一个视频并使用新的编号文件名，不覆盖已有视频。
视频帧直接写入编码器，不在 planner 内存中累计。
Planner 不获得 reset 工具或原生专家策略。

传给 planner 的观测只包含相机数据和机器人自身状态。
MetaWorld 扁平观测中额外包含的物体、目标坐标不会被传出。
当前运行是构造时固定 seed、``num_tasks=1`` 的单次 MT1 episode，
不是 MT10/MT50 完整评测，也不代表其基准成功率。
尚未实现探索模式和 Flash 回放。

环境服务直接包装原生 MetaWorld。
RLinf 现有包装器管理训练所需的向量化 worker，没有提供此处所需的标定深度和渲染接口。
接入复用 RPent 的 Env RPC、主线程调度、进程生命周期、Toolkit 和产物管理；
物理仿真、任务定义和成功判定仍由原生 MetaWorld 提供。

验证
----

.. code-block:: bash

   pytest tests/unit_tests/robots/metaworld -q
   CUDA_VISIBLE_DEVICES=0 bash tests/e2e_tests/run_gpu_suite.sh \
     metaworld /path/to/new-results /path/to/new-venvs

干净 GPU 环境测试覆盖十个任务的 RGB-D、所有支持的相机、
与真实倾斜平面渲染对照的像素中心反投影、只读观测不变量、非法及结束后动作的拒绝、
真实且有步数上限的工具调用、finish 记录、重置前后的视频帧数和自有进程清理。
该测试不调用远程 planner，也不要求任务成功。
维护者可以通过 ``/ci-metaworld`` 请求同一套检查。

外部动作模型
------------

在前面的运行命令中追加 ``--am-endpoint http://HOST:PORT``，即可连接兼容的
WAM 服务并启用 ``wam_act(max_chunks=1)``。默认脚本工具仍可使用。
服务由外部管理；工具包连接时重置本次运行的模型会话，关闭时释放会话。

服务须实现共享的 ``wam.*`` RPC 协议，并匹配
``robots/metaworld/control.py`` 中的 ``METAWORLD_CONTROL``。
输入包括左上原点 RGB（``images.primary``）、以米为单位的世界坐标
``state.eef_position``、形状为 ``[1]`` 的归一化夹爪开度
``state.gripper_opening`` 以及任务指令。
输出为有限值 ``[N, 4]`` 动作，范围 ``[-1, 1]``：前三维是缩放系数为
0.01 米的世界坐标 XYZ 增量，第四维为夹爪命令（+1 闭合，-1 张开）；
控制频率为 80 Hz，手腕方向固定。

控制协议不匹配、所需观测不可用或动作片段无效时，会在执行前报错。
取消操作和 episode 结束都会停止后续动作；模型预测图像不会替代真实仿真观测。
此接入不附带 MetaWorld 预训练模型或权重启动器，不能仅更换 endpoint 就使用
LIBERO/RoboTwin 权重。模型适配方法见 :doc:`../development/add_action_model`。

渲染设备选择
------------

使用 ``--cuda-device N`` 为自动启动的环境选择物理 CUDA 设备。
共享映射函数会将 CUDA 序号转换为 EGL 设备索引，并保留已有的
``MUJOCO_EGL_DEVICE_ID`` 设置。``MUJOCO_GL`` 默认为 ``egl``；
显式设置的 ``osmesa`` 等后端会被保留，且不会触发 EGL 设备探测。
借用的外部环境服务须由其启动者配置渲染设备。
