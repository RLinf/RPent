RoboCasa
========

`RoboCasa <https://robocasa.ai>`_ 是厨房尺度、长时序的操作 environment。
在 RPent 中由 **RLDX-1** VLA 策略驱动，默认通过 HTTP RPC 提供服务
（与 LIBERO 一致），也支持 pickle-framed socket 传输。详见
``robots/robocasa/vla_server.py`` 与 ``robots/robocasa/__init__.py``
中的传输选择逻辑。

安装
----

RoboCasa365 不在 ``.[full]`` 里。``.[robocasa]`` extra 装齐整个 stack
—— MuJoCo 3.3.1、ARISE-Initiative robosuite fork、pinned lerobot commit、
protobuf，以及 ``robocasa`` 包本身（从 ``github.com/qurakchin/robocasa``
fork 的 ``v1.0.1_rlinf`` 分支装的 wheel）。fork 改造过，让
``macros_private`` 和 ``assets`` 都从 env var 加载，所以非 editable 的
wheel 装也能用，不需要本地 clone。

.. code-block:: bash

   uv pip install -e ".[robocasa]"
   bash scripts/robocasa/install_robocasa.sh

helper 脚本（最后一行）把 ``macros_private.py`` 写到项目内的 ``.robocasa/``，
并 print 出你需要设的两个 env var（``ROBOCASA_MACROS_PATH``、
``ROBOCASA_ASSETS_PATH``）。它本身 **不** 跑任何 ``pip install``，
安装全走 extra。详见 :doc:`../installation`。

RLDX-1 checkpoint 是单独的 —— 见下面运行命令的 ``--vla-model-path``。

可用任务家族
------------

RoboCasa 覆盖标准厨房 benchmark:

- ``PickPlace*`` —— 把物体从起始位置搬到目标位置 (灶台 → 橱柜、水槽
  → 灶台…)。
- ``Open*`` / ``Close*`` —— 开合橱柜门、抽屉、家电。
- ``TurnOn*`` / ``TurnOff*`` —— 操作灶台旋钮、微波炉按钮、水壶开关等。

具体列表取决于 RoboCasa 版本；当前目录参见
`RoboCasa <https://robocasa.ai>`_ 上游。

运行一个任务
------------

RoboCasa 的 CLI 参数由 ``robots/robocasa/__init__`` 注册，可通过
``rpent --env robocasa --help`` 查看:

.. code-block:: bash

   rpent --env robocasa \
         --robocasa-env OpenDrawer \
         --robocasa-split target \
         --seed 0 \
         --vla-model-path /path/to/rldx \
         --planner claude_code \
         --model claude-opus-4-8

使用 ``--env-endpoint`` / ``--vla-endpoint`` 指向已运行的服务器
(``[protocol://]host:port``)；不指定时，RPent 会就地启动 env 和 VLA
子进程，日志分别写到 ``<output_dir>/env_server.log`` 和
``<output_dir>/vla_server.log``。

Toolkit 与 LIBERO 的差异
------------------------

RoboCasa toolkit 的工具 *形状* 和 LIBERO 相同 (一次原语调用、
一次状态查看、一次 ``finish``), 但有两处是 RoboCasa 特有的:

- **Env 侧的辅助方法。** 抓取检测与动作组装需要活着的仿真 env, 所以
  它们是 env_server 的 RPC。Agent 侧的 skill 因此同时持有 **两个**
  client: env client 做 render/step, model client 做 RLDX-1 推理。
  理由参见 :doc:`../development/add_robot`。
- **观测形状。** RLDX-1 看到的是 3 路相机 video 张量
  ``(1, T, H, W, 3)``, 按历史 ``T`` 堆叠，加上 ``state.*``、annotation、
  以及一个 session id (用于 ``reset_session`` / ``predict``)。
