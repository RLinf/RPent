RoboCasa
========

`RoboCasa <https://robocasa.ai>`_ 是厨房尺度、长时序的操作 environment。
在 RPent 中它由 **RLDX-1** VLA 策略驱动, 通过 pickle-framed socket RPC
(而非 LIBERO 用的 HTTP) 提供服务 —— 因为 RLDX 的观测是历史堆叠的嵌套
numpy dict, socket 天然承载, HTTP 反而需要额外设计 wire 格式。

一次性安装
----------

RoboCasa 需要资产、robosuite / composite-controller 配置, 以及 RLDX-1
checkpoint。执行:

.. code-block:: bash

   bash scripts/setup_robocasa.sh

脚本会下载 RoboCasa365 资产、patch robosuite、并把 RLDX-1 checkpoint
放到 ``vla_server`` 期望的位置。完整流程见
`docs/SETUP_ROBOCASA.zh.md
<https://github.com/RLinf/RPent/blob/main/docs/SETUP_ROBOCASA.zh.md>`_。

最小命令
--------

RoboCasa 有自己的便捷入口:

.. code-block:: bash

   bash scripts/run_robocasa.sh <task> <gpu> <seed>

例如:

.. code-block:: bash

   bash scripts/run_robocasa.sh PickPlaceCounterToCabinet 0 0

背后启动的是和 LIBERO 一样的三个进程 —— agent、env_server、vla_server ——
只是接入了 RoboCasa 的 client 类和 RLDX-1 的 model client。

可用任务家族
------------

RPent 中的 RoboCasa 支持标准厨房 benchmark:

- ``PickPlace*`` —— 把物体从起始位置搬到目标位置 (灶台 → 橱柜、水槽
  → 灶台…)。
- ``Open*`` / ``Close*`` —— 开合橱柜门、抽屉、家电。
- ``TurnOn*`` / ``TurnOff*`` —— 操作灶台旋钮、微波炉按钮、水壶开关等。

具体列表取决于你装的 RoboCasa 版本; 一旦 ``robots/robocasa/`` 落地,
到那里 (或 RoboCasa 上游) 查看当前目录。

Toolkit 与 LIBERO 的差异
------------------------

RoboCasa toolkit 的工具 *形状* 和 LIBERO 相同 (一次 primitive 调用、
一次状态查看、一次 ``finish``), 但有两处是 RoboCasa 特有的:

- **Env 侧的辅助方法。** ``check_grasp`` 和 ``assemble_action``
  —— 即 eval 时的 ``unmap_action`` + composite-controller
  split-index 组装 —— 需要活着的 robosuite env, 所以它们是
  env_server 的 RPC。Agent 侧的 skill (``RLDXSkill``) 因此同时
  持有 **两个** client: env client 做 render/step/grasp/assemble,
  model client 做 RLDX-1 推理。理由参见
  :doc:`../development/add_robot`。
- **观测形状。** RLDX-1 看到的是 3 路相机 video 张量
  ``(1, T, H, W, 3)``, 按历史 ``T`` 堆叠, 加上 ``state.*``、annotation、
  session / reset_memory。这也是为什么 RoboCasa vla_server 选 socket
  RPC 而不是 HTTP。

复用一个已在运行的 vla_server
-----------------------------

RLDX-1 加载一次, 跑很多次任务:

.. code-block:: bash

   # 终端 1: 起 vla_server
   python -m robots.robocasa.vla_server --host 0.0.0.0 --port 8100

   # 终端 2..N: 指向它
   VLA_ENDPOINT=socket://host:8100 bash scripts/run_robocasa.sh \
     PickPlaceCounterToCabinet 0 0

env server 起得快 (几秒), VLA 不快。

Dashboard
---------

RoboCasa 也支持 ``--dashboard``, 用法和 LIBERO 一致。把
``--dashboard`` 加到 ``run_robocasa.sh`` 底下调的 ``rpent/cli/main.py`` 命令
上即可。
