LIBERO
======

`LIBERO <https://libero-project.github.io/>`_ 是 RPent 主要使用的仿真基准，
包含一系列基于 MuJoCo/robosuite 的桌面操作任务。RPent 主要使用四个核心基础
任务族 (``libero_object``、``libero_goal``、``libero_spatial``、
``libero_10``) 和三个变体 (``standard``、``pro``、``plus``)。默认 VLA
是 **Pi0.5**, 由 ``robots/libero/vla_server.py`` 通过 HTTP 提供服务。

VLA 配置
--------

下载推荐的 SFT checkpoint
`RLinf-Pi05-LIBERO-130-fullshot-SFT
<https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT>`_，
再将 ``PI05_CHECKPOINT_PATH`` 指向本地 checkpoint 目录：

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
     --local-dir /path/to/rlinf-pi05-libero-130-fullshot-sft

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

SAM3 配置
---------

每次 LIBERO 运行都默认启用 SAM 3.0 分割。可以用以下任一方式下载
``sam3.pt``，再通过 ``SAM3_CHECKPOINT_PATH`` 指定本地 checkpoint:

.. code-block:: bash

   # Hugging Face（需要先在模型页面申请访问权限）
   hf auth login
   hf download facebook/sam3 sam3.pt --local-dir /path/to/sam3

   # ModelScope（与上面的 Hugging Face 命令二选一）
   modelscope download --model facebook/sam3 sam3.pt --local_dir /path/to/sam3

   export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt

SAM 3.0 checkpoint 可以从以下页面下载:
`Hugging Face: facebook/sam3 <https://huggingface.co/facebook/sam3>`_、
`ModelScope: facebook/sam3 <https://modelscope.cn/models/facebook/sam3>`_。

任务选择
--------

运行 LIBERO 任务时，可通过以下参数选择任务：

- ``--suite`` —— 选择要运行的任务套件。完整核心套件列表见
  :ref:`libero-pro-core-suites`。
- ``--task`` —— 套件内的任务索引。
- ``--seed`` —— environment 种子。
- ``--libero-type`` —— LIBERO 变体: ``standard`` | ``pro`` |
  ``plus``。不填时 RPent 会读环境变量 ``LIBERO_TYPE`` (默认 ``pro``)。

.. _libero-pro-core-suites:

LIBERO-PRO 核心套件一览
~~~~~~~~~~~~~~~~~~~~~~~

下表完整列出 RPent 的四个 LIBERO-PRO 核心任务族及其全部扰动套件。

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - 任务族
     - 基础套件
     - 扰动套件
   * - 物体
     - ``libero_object``
     - ``libero_object_task``、``libero_object_swap``、
       ``libero_object_lan``、``libero_object_object``
   * - 目标
     - ``libero_goal``
     - ``libero_goal_task``、``libero_goal_swap``、
       ``libero_goal_lan``、``libero_goal_object``
   * - 空间
     - ``libero_spatial``
     - ``libero_spatial_task``、``libero_spatial_swap``、
       ``libero_spatial_lan``、``libero_spatial_object``
   * - LIBERO-10
     - ``libero_10``
     - ``libero_10_task``、``libero_10_swap``、``libero_10_lan``、
       ``libero_10_object``

后缀表示 LIBERO-PRO 的扰动类型：``_task`` 是 Task/P1，``_swap`` 是
Position/P2，``_lan`` 是 Semantic，``_object`` 是 Object。

最小命令
--------

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   export LIBERO_TYPE=pro
   export CUDA_VISIBLE_DEVICES=0

   rpent --env libero \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

进程分工
--------

- **env_server** (``robots/libero/env_server.py``) —— 负责运行 LIBERO
  的 MuJoCo 环境并通过 EGL 渲染。它通过 RPC 传输 (默认 HTTP; 加
  ``--transport socket`` 走 pickle-framed socket) 对外暴露
  ``reset``、``step``、``chunk_step``、``render_camera``、
  ``get_camera_meta``、``cached_image``…
- **vla_server** (``robots/libero/vla_server.py``) —— 持有 Pi0.5
  权重, 通过同一套 RPC 传输 (HTTP 或 socket) 暴露 ``predict``。
- **sam3_server** (``robots/libero/sam3_server.py``) —— 持有 SAM 3.0,
  通过同一套 RPC 传输 (HTTP 或 socket) 支持文本或单个正点分割, 只返回
  top-1 压缩 PNG mask。
- **Toolkit** (``robots/libero/toolkit.py``) —— 定义 LLM 能调的工具:
  ``pi0_pick`` (交给 Pi0.5)、``move_to``、``rotate_wrist``、
  ``back_project``、``view_driver_state``、``finish``…

Planner 能看到的工具
--------------------

常用的 LIBERO 工具包括：

- ``pi0_pick(prompt, ...)`` —— 调用 Pi0.5 执行闭环抓取。
- ``pi0_doubled(prompt, ...)`` —— 调用 Pi0.5 执行非抓取类接触动作。
- ``move_to(xyz, ...)`` —— 将末端执行器移动到世界坐标系中的目标位置。
- ``move_pose(xyz, target_pitch=..., target_yaw=..., ...)`` —— 同时调整
  末端位置和姿态。
- ``rotate_wrist(target_yaw=... / delta_yaw=..., ...)`` —— 按绝对或相对
  yaw 旋转腕部。
- ``rotate_pitch(target_pitch=... / delta_pitch=..., ...)`` —— 按绝对或
  相对 pitch 倾斜夹爪。
- ``set_gripper(gripper=..., steps=...)`` —— 保持末端姿态，并在指定步数内
  控制夹爪。
- ``release(...)`` —— 打开夹爪。
- ``back_project(row, col, ...)`` —— 将图像像素反投影到世界坐标。
- ``segment(prompt=... / point=..., ...)`` —— 通过 SAM3 对已有图像进行文本或
  点提示分割。
- ``view_driver_state(step=None)`` —— 读取已有的状态和图像记录。
- ``view_camera_meta(camera=..., step=None)`` —— 读取已有的相机元数据。
- ``finish(status, summary)`` —— 结束当前运行。

物理动作工具执行后会记录新的状态和图像；只读工具不会推进环境。

Dashboard
---------

加上 ``--dashboard`` 可启动本地监控服务。系统会自动选择一个空闲端口，
并在终端输出访问 URL：

.. code-block:: bash

   rpent --env libero --dashboard \
     --suite libero_goal_task --task 1 --seed 0 \
     --planner claude_code --model claude-opus-4-8

Dashboard 会实时展示推理过程、agentview 视图、腕部相机视图、Pi0.5
叠加信息和动作时间线。使用 ``--dashboard-language zh-cn`` 切换中文 UI。

接入自定义 VLA
----------------

如果你有一个与 LIBERO 兼容、但并非 Pi0.5 的 VLA，可以在不修改环境实现的
情况下替换 model client：

1. 写一个新的 ``vla_server.py``, 暴露相同的 ``predict`` RPC 契约
   (http 或 socket 皆可)。
2. 用 ``--vla-endpoint [protocol://]host:port`` 指向它。
3. 如果 tool surface 要变 (比如 ``pi0_pick`` 改成 ``mymodel_pick``),
   相应更新 ``robots/libero/toolkit.py``。

完整流程见 :doc:`../development/add_primitive`。
