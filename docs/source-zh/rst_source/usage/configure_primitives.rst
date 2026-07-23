Action Primitives
=================

Planner 决定执行什么操作，而 **action primitive** 决定如何执行。每个 primitive
都会将一次工具调用（如 ``pi0_pick``、``move_to`` 或
``open_drawer``）转换成一段可由环境直接执行的动作。

RPent 内置两类 primitive：

- **VLA 策略**：Vision-Language-Action 模型运行在独立的 ``vla_server``
  进程中，将 GPU 权重与物理引擎隔离。toolkit 通过各环境对应的 model
  client 调用模型，例如 Pi0.5（LIBERO）和 RLDX-1（RoboCasa）。
- **脚本化 primitive**：用于执行 ``move_to``、``rotate_wrist``、
  ``release`` 和 ``back_project`` 等确定性动作。这类 primitive 位于
  agent 侧，不需要加载 VLA 权重，并通过 RPC 调用 ``env_server``。

各环境的具体配置，例如使用哪个 VLA、checkpoint 路径以及对外提供的工具，
请参考对应的环境页面：:doc:`libero`、:doc:`robocasa`、
:doc:`franka`、:doc:`so101`。

各环境使用的 VLA
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - 环境 / 机器人
     - 默认 VLA
     - 传输协议
     - 服务实现
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

VLA server 通过统一的 ``predict`` 和 ``healthz`` 方法提供服务，并支持
HTTP（JSON）和 socket（pickle-framed）两种传输方式。可通过
``--transport {http,socket}`` 选择，默认为 ``http``。设计理由参见
:doc:`../development/add_robot`。

复用正在运行的 VLA server
-------------------------

VLA server 可以在多次运行之间复用。使用 ``--vla-endpoint`` 连接已经运行的
实例，无需每次重新启动：

.. code-block:: bash

   rpent --env libero --vla-endpoint http://localhost:8000 \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

``--vla-endpoint`` 接受 ``[protocol://]host:port`` 格式，其中 protocol
可以是 ``http``（默认）或 ``socket``。``--env-endpoint`` 采用相同格式，
用于复用正在运行的 ``env_server``。

新增 primitive 类别
-------------------

如果要接入的既不是 VLA 也不是脚本化运动 —— 比如一个
WAM (World Action Model)、Diffusion Policy 或 MPC primitive ——
参见 :doc:`../development/add_primitive`。
