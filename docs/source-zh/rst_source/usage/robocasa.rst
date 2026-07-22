RoboCasa
========

`RoboCasa <https://robocasa.ai>`_ 是厨房尺度、长时序的操作 environment。
在 RPent 中它由 **RLDX-1** VLA 策略驱动, 通过 pickle-framed socket RPC
(而非 LIBERO 用的 HTTP) 提供服务 —— 因为 RLDX 的观测是历史堆叠的嵌套
numpy dict, socket 天然承载, HTTP 反而需要额外设计 wire 格式。

可用任务家族
------------

RoboCasa 覆盖标准厨房 benchmark:

- ``PickPlace*`` —— 把物体从起始位置搬到目标位置 (灶台 → 橱柜、水槽
  → 灶台…)。
- ``Open*`` / ``Close*`` —— 开合橱柜门、抽屉、家电。
- ``TurnOn*`` / ``TurnOff*`` —— 操作灶台旋钮、微波炉按钮、水壶开关等。

具体列表取决于 RoboCasa 版本; 当前目录参见
`RoboCasa <https://robocasa.ai>`_ 上游。

Toolkit 与 LIBERO 的差异
------------------------

RoboCasa toolkit 的工具 *形状* 和 LIBERO 相同 (一次 primitive 调用、
一次状态查看、一次 ``finish``), 但有两处是 RoboCasa 特有的:

- **Env 侧的辅助方法。** 抓取检测与动作组装需要活着的仿真 env, 所以
  它们是 env_server 的 RPC。Agent 侧的 skill 因此同时持有 **两个**
  client: env client 做 render/step, model client 做 RLDX-1 推理。
  理由参见 :doc:`../development/add_robot`。
- **观测形状。** RLDX-1 看到的是 3 路相机 video 张量
  ``(1, T, H, W, 3)``, 按历史 ``T`` 堆叠, 加上 ``state.*``、annotation、
  session / reset_memory。
