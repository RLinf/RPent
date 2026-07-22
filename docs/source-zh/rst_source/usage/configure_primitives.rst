Action Primitives
=================

Planner 决定 *做什么*, 而 **action primitive** 决定 *怎么做*。所谓 primitive
就是把一次 tool 调用 (``pi0_pick``、``move_to``、``open_drawer``…) 变成
一段可以直接送给 environment 执行的动作。

RPent 内置支持两大类 primitive:

- **VLA 策略** (Vision-Language-Action 模型)。跑在专门的 ``vla_server``
  进程里, 把 GPU 权重与物理引擎隔离; toolkit 通过 per-env 的 model
  client 调用它。例如 Pi0.5 (LIBERO)、RLDX-1 (RoboCasa)。
- **脚本化 primitive**。确定性运动, 如 ``move_to``、``rotate_wrist``、
  ``release`` 或 ``back_project``。它们放在 agent 侧 (不需要 VLA
  权重), 通过 ``env_server`` 的 RPC 调用。

具体到每一种机器人的配置 (哪个 VLA、checkpoint 路径、tool surface),
参见对应的 environment 页: :doc:`libero`、:doc:`robocasa`、
:doc:`franka`、:doc:`so101`。

不同 environment 用哪个 VLA
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Environment / 机器人
     - 默认 VLA
     - 传输协议
     - Server
   * - LIBERO (仿真)
     - Pi0.5
     - HTTP 或 socket RPC (``--transport``)
     - ``robots/libero/vla_server.py``
   * - RoboCasa (仿真)
     - RLDX-1
     - pickle-framed socket RPC
     - ``robots/robocasa/vla_server.py`` *(规划中)*
   * - Franka (真机)
     - Pi0.5 或 RLDX-1 (依任务而定)
     - HTTP 或 socket
     - ``robots/franka/vla_server.py`` *(规划中)*
   * - SO-101 (真机)
     - RLDX-1 (依任务而定)
     - socket RPC
     - ``robots/so101/vla_server.py`` *(规划中)*

VLA server 用同一套 ``predict`` / ``healthz`` 方法, 同时支持 HTTP (JSON)
与 socket (pickle-framed) 两种传输, 通过 ``--transport {http,socket}``
选择 (默认 ``http``)。设计理由参见 :doc:`../development/add_robot`。

复用一个已在运行的 VLA server
-----------------------------

每一个 VLA server 都设计成 **可跨 run 复用**。用 ``--vla-endpoint``
指向已在跑的实例, 而不是每次都启动新实例:

.. code-block:: bash

   rpent --env libero --vla-endpoint http://localhost:8000 \
     --suite libero_object_swap --task 2 --seed 0 --planner api \
     --model anthropic:claude-opus-4-8

``--vla-endpoint`` 接受 ``[protocol://]host:port`` 格式, protocol 可为
``http`` (默认) 或 ``socket``。同样的规则适用于 ``--env-endpoint``
(复用已有的 env_server)。

新增全新的 primitive 家族
-------------------------

如果要接入的既不是 VLA 也不是脚本化运动 —— 比如一个
WAM (World Action Model)、扩散规划器、或 MPC primitive ——
参见 :doc:`../development/add_primitive`。
