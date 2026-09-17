RoboDojo
========

.. toctree::
   :maxdepth: 1
   :hidden:

   installation

RoboDojo 是 RPent 的一个可插拔仿真后端（``rpent --robot robodojo``），把
Isaac Sim / IsaacLab 上的双臂 ARX-X5 与 Pi_05 策略接入 RPent 的
LLM-in-the-loop runner，与 LIBERO 等后端并存。Planner（LLM）、工具协议、
SAM3 感知与 memory 层完全复用，只替换"身体"（仿真器/机器人）。

主要模块
--------

* ``robots/robodojo/env_server.py`` —— Isaac Sim RPC 服务（主线程渲染、
  三相机 + 深度 + 标定、joint/ee 动作、逐相机视频录制）。
* ``robots/robodojo/env_client.py`` —— 继承 ``BaseEnvClient`` 的 rpent
  侧客户端。
* 共享的 ``rpent/robots/components/xpolicylab_vla_server.py`` 和
  ``robots/robodojo/vla_client.py`` —— Pi_05 策略服务
  （XPolicyLab WebSocket）适配到共享 ``BaseVLAFacade`` / ``BaseVLAClient``
  协议。
* ``robots/robodojo/toolkit.py`` / ``tools.py`` —— view_env_state /
  back_project / segment / move_to / set_gripper / pi0_pick / stabilize /
  place_in_bin / get_reward_details 等原语。
* ``robots/robodojo/robot_spec.py`` —— RobotSpec 工厂（CLI、RunConfig、
  运行时编排）。
* ``robots/robodojo/tasks.py`` —— 从指定源码目录读取任务列表。

快速开始
--------

.. code-block:: bash

   cd <rpent checkout>
   export PATH="$PWD/.venv/bin:$PATH" \
     SAM3_CHECKPOINT_PATH=$PWD/checkpoints/sam3/sam3.pt \
     HF_HUB_DISABLE_XET=1 CELL_TIMEOUT_S=3600
   rpent --robot robodojo --task put_bottles_into_dustbin --layout 1 \
     --source-root /path/to/RoboDojo \
     --sim-python /path/to/sim-env/bin/python \
     --pi05-python /path/to/pi05-env/bin/python \
     --cuda-device 0 --planner codex --model deepseek-v4-flash --max-turns 30

运行输出（含 reward_details 审计、三相机 mp4、transcript）写到
``logs/<timestamp>_robodojo_<task>_l<layout>/``。

源码、资产与运行时配置请见 :doc:`installation`。

工具与信息访问
--------------

planner 直接提供工具，按工具列表中的名称调用即可。垃圾桶放置与瓶子评分指导仅在
``put_bottles_into_dustbin`` 的任务上下文中提供，不放入通用 system prompt。
状态记录读取和标定深度反投影使用共享感知函数；控制与双臂监控保留在后端。

``robots.robodojo.tools.TOOL_GROUPS`` 将工具的直接输出分为 ``general``
（深度、分割与运动）、``privileged``（``get_reward_details`` 和
``get_safety_status``）与 ``mixed``（``view_env_state``、``set_gripper``、
``place_in_bin``）。安全告警暴露物体真值世界坐标，reward 明细暴露逐物体成功谓词；
混合输出包含成功标志、状态或历史结果，可能携带特权信息。

Python toolkit 工厂接受 ``allowed_tool_groups``，例如传入
``frozenset({"general"})`` 只注册 general 组的机器人工具。默认 ``None``
保留已有工具集合。该挂钩同时过滤工具 schema 和处理函数，但不处理动作后的自动状态、
原始观测字段、日志、memory 或通用文件工具，因此不是评估隔离模式。
