RoboDojo
========

RoboDojo 是 RPent 的一个可插拔仿真后端（``rpent --env robodojo``），把
Isaac Sim / IsaacLab 上的双臂 ARX-X5 与 Pi_05 策略接入 RPent 的
LLM-in-the-loop runner，与 LIBERO 等后端并存。Planner（LLM）、工具协议、
SAM3 感知与 memory 层完全复用，只替换"身体"（仿真器/机器人）。

主要模块
--------

* ``robots/robodojo/env_server.py`` —— Isaac Sim RPC 服务（主线程渲染、
  三相机 + 深度 + 标定、joint/ee 动作、逐相机视频录制）。
* ``robots/robodojo/env_client.py`` —— 继承 ``BaseEnvClient`` 的 rpent
  侧客户端。
* ``robots/robodojo/vla_server.py`` / ``vla_client.py`` —— Pi_05 策略服务
  （XPolicyLab WebSocket）适配到共享 ``BaseVLAFacade`` / ``BaseVLAClient``
  协议。
* ``robots/robodojo/toolkit.py`` / ``tools.py`` —— view_env_state /
  back_project / segment / move_to / set_gripper / pi0_pick / stabilize /
  place_in_bin / get_reward_details 等原语。
* ``robots/robodojo/robot_spec.py`` —— RobotSpec 工厂（CLI、RunConfig、
  运行时编排）。
* ``robots/robodojo/tasks.py`` —— 54 个 RoboDojo 任务的动态映射。

快速开始
--------

.. code-block:: bash

   cd <rpent checkout>
   export PATH="$PWD/.venv/bin:$PATH" \
     SAM3_CHECKPOINT_PATH=$PWD/checkpoints/sam3/sam3.pt \
     HF_HUB_DISABLE_XET=1 CELL_TIMEOUT_S=3600
   rpent --env robodojo --task put_bottles_into_dustbin --layout 1 \
     --sim-device 0 --planner codex --model deepseek-v4-flash --max-turns 30

运行输出（含 reward_details 审计、三相机 mp4、transcript）写到
``logs/<timestamp>_robodojo_<task>_l<layout>/``。

从零搭建完整环境请见 :doc:`installation`；裸策略 vs Harness 的对照口径见
:doc:`ab_protocol`；集成过程记录见 :doc:`integration_log`。
