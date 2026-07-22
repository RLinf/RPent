LIBERO
======

`LIBERO <https://libero-project.github.io/>`_ 是 RPent 默认的 environment:
一个基于 MuJoCo/robosuite 的桌面操作基准, 包含四个套件
(``libero_object``、``libero_goal``、``libero_spatial``、``libero_10``)
和三个变体 (``standard``、``pro``、``plus``)。默认 VLA 是 **Pi0.5**,
由 ``robots/libero/vla_server.py`` 通过 HTTP 提供服务。

VLA 配置
--------

Pi0.5 只需要一件事: 磁盘上的 checkpoint。下载推荐的 SFT checkpoint，
再通过 ``PI05_CHECKPOINT_PATH`` 指向它:

.. code-block:: bash

   hf download RLinf/rlinf-pi05-libero-130-fullshot-sft \
     --repo-type dataset \
     --local-dir /path/to/rlinf-pi05-libero-130-fullshot-sft

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

checkpoint 页面:
`rlinf-pi05-libero-130-fullshot-sft
<https://huggingface.co/datasets/RLinf/rlinf-pi05-libero-130-fullshot-sft>`_。

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

每一次 LIBERO 运行都要指定:

- ``--suite`` —— 四个套件之一, 可选地带上变体后缀 (见下)。例:
  ``libero_object_task``、``libero_object_swap``、
  ``libero_goal_lan``、``libero_spatial_task``、
  ``libero_10_swap``。
- ``--task`` —— 套件内的任务索引。
- ``--seed`` —— environment 种子。
- ``--libero-type`` —— LIBERO 变体: ``standard`` | ``pro`` |
  ``plus``。不填时 RPent 会读环境变量 ``LIBERO_TYPE`` (默认 ``pro``)。

套件 × 变体一览
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - 套件
     - 变体
     - 用途
   * - ``libero_object``
     - ``_task`` / ``_swap`` / ``_lan``
     - 面向物体的任务, 支持目标 swap 或语言扰动。
   * - ``libero_goal``
     - ``_task`` / ``_swap`` / ``_lan``
     - 目标条件任务, 支持 swap / 语言扰动。
   * - ``libero_spatial``
     - ``_task`` / ``_lan``
     - 空间关系任务。
   * - ``libero_10``
     - ``_task`` / ``_swap`` / ``_lan``
     - 长时序的 LIBERO-10 套件。

最小命令
--------

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   export LIBERO_TYPE=pro
   export CUDA_VISIBLE_DEVICES=0

   rpent \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner api --model anthropic:claude-opus-4-8 \
     --max-tokens 8192

进程分工
--------

- **env_server** (``robots/libero/env_server.py``) —— 持有 LIBERO
  MuJoCo env 与 EGL 渲染。通过 pickle-framed socket RPC 对外暴露
  ``reset``、``step``、``chunk_step``、``render_agentview``、
  ``get_camera_meta``、``cached_image``…
- **vla_server** (``robots/libero/vla_server.py``) —— 持有 Pi0.5
  权重, 通过 HTTP 暴露 ``/predict``。
- **sam3_server** (``robots/libero/sam3_server.py``) —— 持有 SAM 3.0,
  通过同一个 ``/segment`` endpoint 支持文本或单个正点分割, 只返回
  top-1 压缩 PNG mask。
- **Toolkit** (``robots/libero/toolkit.py``) —— 定义 LLM 能调的工具:
  ``pi0_pick`` (交给 Pi0.5)、``move_to``、``rotate_wrist``、
  ``back_project``、``view_driver_state``、``finish``…

Planner 能看到的工具
--------------------

LIBERO toolkit 默认暴露:

- ``pi0_pick(target)`` —— 调用 Pi0.5 生成一次针对 ``target`` (自然语言
  描述的物体) 的抓取动作块。
- ``move_to(dx, dy, dz)`` —— 脚本化 Cartesian 运动 (确定性, 不走 VLA)。
- ``rotate_wrist(delta_rad)`` —— 脚本化的腕关节旋转。
- ``release()`` —— 打开夹爪。
- ``back_project(pixel_x, pixel_y)`` —— 把 agentview 图像上的像素点
  反投影到世界坐标 3D 点。
- ``segment(prompt=..., point=...)`` —— 文本提示与单个 ``[row, col]``
  正点二选一, 用 SAM3 的 top-1 mask 反投影得到 ``world_xyz``。mask 只在
  服务端与客户端之间使用, planner 只收到摘要和 artifact 路径。
- ``view_driver_state()`` —— 强制刷新一次状态 dump (图像、深度、
  camera meta、``states.json``)。
- ``finish(status)`` —— 以 ``success`` / ``failure`` / ``stuck``
  结束当前 episode。

每个工具跑完后都会重新渲染世界, 所以下一轮 agent 上下文反映的是
动作后的状态。

Dashboard
---------

给 LIBERO 运行加上 ``--dashboard`` 打开本地监控页:

.. code-block:: bash

   rpent --dashboard \
     --suite libero_goal_task --task 1 --seed 0 --planner claude_code

Dashboard streams reasoning、agentview + 腕部相机 + Pi0.5 叠加视图,
以及动作时间线。用 ``--dashboard-language zh-cn`` 切换中文 UI。

自带 VLA
--------

如果你有一个与 LIBERO 兼容、但不是 Pi0.5 的 VLA, 可以在不动 env 的
情况下把 model client 换掉:

1. 写一个新的 ``vla_server.py``, 暴露相同的 HTTP ``/predict``
   契约 (或改走 socket RPC, 如果那更合适)。
2. 用 ``--vla-endpoint http://host:port`` 指向它。
3. 如果 tool surface 要变 (比如 ``pi0_pick`` 改成 ``mymodel_pick``),
   相应更新 ``robots/libero/toolkit.py``。

完整流程见 :doc:`../development/add_primitive`。
