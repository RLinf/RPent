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

开发与冻结重放
--------------

普通 planner 保留开发工具集合和特权反馈。``--planner flash`` 选择 eval-fair，
由 RPent 原生 Flash planner 调用 ``RobotSpec.run_flash``，LLM 不进环。
RoboDojo 仍不支持 ``--explore``；这里的开发指普通 planner 循环，并非该 CLI 模式。

开发运行在输出目录记录 ``flash_trace.json``。记录可迁移航点时，先对
``cam_head`` 调用 ``segment``，再对框中心取整后的像素调用 ``back_project``，
然后执行动作。每次动作前重复这组感知查询，在评测前导出：

.. code-block:: bash

   python -m robots.robodojo.flash.generate \
     --trace /path/to/dev-run/flash_trace.json \
     --task put_bottles_into_dustbin \
     --destination /path/to/memory/robodojo/flash

版本 1 JSON 包含任务、SAM3 符号查询、可选腕部精定位设置，以及有序动作、参数与
三维相对偏移，不保存参考物体坐标、分数或谓词结果。
导出支持 ``move_to``、``set_gripper`` 和 ``pi0_pick``；不支持的动作
（包括 ``place_in_bin``、``stabilize``）、失败调用或缺少锚点的航点会被拒绝，
应在开发期将这些操作记录为支持的基础动作。导出不会覆盖已有计划。
评测前审核并冻结计划；重放期间不筛选候选计划，也不写入计划。

使用正常运行时参数并指定：

.. code-block:: bash

   rpent --robot robodojo --task put_bottles_into_dustbin --layout 1 \
     --planner flash --memory-profile local --memory-dir /path/to/memory/robodojo \
     --source-root /path/to/RoboDojo \
     --sim-python /path/to/sim-env/bin/python \
     --pi05-python /path/to/pi05-env/bin/python

计划位于选定机器人 memory 的 ``flash/<task>_plan.json``。
HF 模式复用原生 ``robodojo/flash/**`` 同步过滤；仓库不附带 RoboDojo 计划，
也不保证 HF 已发布这些计划。计划缺失或无效时拒绝重放。

重放先在开头头部画面定位全部锚点，航点由实时锚点加记录偏移得到，偏移长度上限
0.5 m。导出时可传 ``--refine-camera cam_left_wrist`` 或 ``cam_right_wrist``，
要求运动前用腕部画面精定位；与头部定位相差超过 5 cm 则停止。
精定位不会主动移动腕部以获得视野；应记录能提供该视野的接近动作，或不启用精定位。
``pi0_pick`` 语义不变，重放额外检查夹爪闭合且腕部定位的物体距任一末端不超过
12 cm。未保持抓取时重新用头部定位并重放前一接近动作，最多尝试抓取三次。
调用失败、定位丢失、航点不可达或步数耗尽均停止，不重置场景；这些检查不保证碰撞安全。

eval-fair 移除特权工具与通用文件/memory 工具。服务元数据标明模式，拒绝 reset
和诊断 RPC，返回零 reward 且不返回任务成功反馈；观测只含 RGB-D、相机标定、
公开指令和双臂本体状态，不启用瓶子真值告警。eval-fair 拒绝连接 dev 服务；
启动时创建回合，eval 客户端不再重复 reset。自动状态日志因此只包含公开观测与动作诊断。
精简后的 eval prompt 不含评分指导，Flash 也不会使用这些 prompt。

Flash 的 ``done`` 和 planner 完成状态仅表示冻结动作序列执行完毕，不表示官方任务
谓词通过。官方评分必须与重放上下文隔离；宣称 benchmark 兼容或成功前仍需验证
GPU、真实策略服务与仿真端到端。
