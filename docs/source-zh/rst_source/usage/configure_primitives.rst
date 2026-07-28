动作原语
========

planner 决定执行什么操作，而 **动作原语** 决定如何执行。每个原语
都会将一次工具调用（如 ``pi0_pick``、``move_to`` 或
``rotate_wrist``）转换成一段可由环境直接执行的动作。

RPent 内置两类原语：

- **VLA 策略**：VLA 模型运行在独立的 ``vla_server``
  进程中，将 GPU 权重与物理引擎隔离。toolkit 通过各环境对应的 model
  client 调用模型，例如 Pi0.5（LIBERO）和 RLDX-1（RoboCasa）。
- **脚本化原语**：用于执行 ``move_to``、``rotate_wrist``、
  ``release`` 和 ``back_project`` 等确定性动作。这类原语位于
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
     - HTTP 或 socket RPC
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
HTTP（JSON）和 socket（pickle-framed）两种传输方式。直接启动 VLA server
时，可通过服务端的
``--transport {http,socket}`` 选项选择传输方式，默认为 ``http``。
设计理由参见 :doc:`../development/add_robot`。

独立服务、远程 endpoint 和跨运行复用模型的方法参见
:doc:`advanced_deployment`。

新增原语类别
------------

如果要接入的既不是 VLA 也不是脚本化运动 —— 比如一个
WAM (World Action Model)、Diffusion Policy 或 MPC 原语 ——
参见 :doc:`../development/add_primitive`。
