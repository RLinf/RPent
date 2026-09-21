RoboDojo
========

.. toctree::
   :maxdepth: 1
   :hidden:

   installation

RoboDojo 将 Isaac Sim / IsaacLab、双臂 ARX-X5 和 XPolicyLab Pi_05 策略接入
RPent 共享的 planner、工具与 memory 基础设施。该接入仍属实验性：
离线契约测试不代表仿真兼容性或任务成功。安装与可运行的 CLI 示例见
:doc:`installation`。

共享后端接口见 :doc:`../../development/add_robot`。

主要模块
--------

* ``robots/robodojo/env_server.py`` —— Isaac Sim RPC 服务（主线程渲染、
  三相机 + 深度 + 标定、joint/ee 动作、逐相机视频录制）。
* ``robots/robodojo/env_client.py`` —— 继承 ``BaseEnvClient`` 的 rpent
  侧客户端。
* 共享的 ``rpent/robots/components/xpolicylab_vla_server.py`` 和
  ``rpent/robots/components/xpolicylab_vla_client.py`` —— Pi_05 策略服务
  （XPolicyLab WebSocket）适配到共享 ``BaseVLAFacade`` / ``BaseVLAClient``
  协议。
* ``robots/robodojo/toolkit.py`` / ``tools.py`` —— view_env_state /
  back_project / segment / move_to / set_gripper / pi0_pick / stabilize /
  place_in_bin / get_reward_details 等原语。
* ``robots/robodojo/robot_spec.py`` —— RobotSpec 工厂（CLI、RunConfig、
  运行时编排）。
* ``robots/robodojo/tasks.py`` —— 从指定源码目录读取任务列表。

实现细节
--------

环境服务先初始化 Isaac Sim，再导入仿真模块；为保证相机渲染正常，仿真请求在
主线程串行执行。每个进程持有一个仿真应用。reset 返回观测字典，step 返回
``(obs, reward, done, info)`` 四元组。不支持 chunk stepping，原语通过环境
动作接口逐步执行，保留该接口的边界检查与计数。

策略通过 ``rpent.robots.components.xpolicylab_vla_server``
在独立 Python 环境中运行。``--policy-root`` 指向配置源码目录中的
``XPolicyLab/policy/Pi_05``。适配器原样传递观测与动作，并将
``update_obs``/``get_action`` 与 ``reset`` 串行化，不提供会话隔离。
RoboDojo 要求三相机输入和 14-DoF 关节动作；切换策略后端不会转换这些格式。

运行时将本次输出目录分别通过环境服务的 ``--save-dir`` 和策略入口的
``--output-dir`` 传入，内层策略日志写入该目录的 ``vla_server.log``。
直接启动服务且省略这些参数时，均使用当前工作目录。并发运行应使用不同输出目录。

工具与信息访问
--------------

planner 直接提供工具，按工具列表中的名称调用即可。垃圾桶放置与瓶子评分指导仅在
``put_bottles_into_dustbin`` 的任务上下文中提供，不放入通用 system prompt。
状态记录读取和标定深度反投影在 ``robots/robodojo/tools.py`` 内实现；反投影采用
Isaac 的负光轴 Z 约定，分割使用共享 SAM3 client。``view_env_state``、
``back_project``、``segment``、``get_reward_details`` 和 ``get_safety_status``
均为只读调用，不推进环境，也不触发动作后的状态采集。

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
``cam_head`` 调用 ``segment``，再对返回的 ``centroid_rc``（行、列）调用
``back_project``，然后执行动作。该像素是掩码前景坐标均值向下取整，不是框中心。
录制与重放共用此推导；记录的掩码面积和坐标和用于拒绝被篡改的质心。
不要先批量分割多个对象再批量反投影。每次动作前重复这组感知查询，在评测前导出：

.. code-block:: bash

   python -m robots.robodojo.flash.generate \
     --trace /path/to/dev-run/flash_trace.json \
     --task put_bottles_into_dustbin \
     --destination /path/to/memory/robodojo/flash

版本 2 JSON 包含任务、SAM3 符号查询、显式的 ``sam3_mask_centroid_floor_v1``
推导方法、可选腕部精定位设置，以及有序动作、参数与
三维相对偏移，不保存参考物体坐标、分数或谓词结果。
导出支持 ``move_to``、``set_gripper`` 和 ``pi0_pick``；不支持的动作
（包括 ``place_in_bin``、``stabilize``）、失败调用或缺少锚点的航点会被拒绝，
应在开发期将这些操作记录为支持的基础动作。导出不会覆盖已有计划。
评测前审核并冻结计划；重放期间不筛选候选计划，也不写入计划。
版本 1 框中心计划和缺少掩码矩的轨迹必须重新录制，不会静默转换。
共享 XPolicyLab facade 只持有自己启动的策略进程，并在 close、启动失败、
SIGTERM 和解释器正常退出时终止它们；借用的策略服务保持运行。

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

任务语言
~~~~~~~~

任务语言 RPC 与公开观测使用 RoboDojo 的 description manager，而不是原始
``gen_instruction`` 模板。管理器不可用或返回未解析语言时，使用环境 0 的
``get_label_descriptions`` 逐标签填充，确定性地选择第一条描述。缺失标签、空语言
或残留模板标记均明确报错，不用任务名替代。官方 instruction 在 eval-fair 中仍是公开信息。
省略 ``pi0_pick.prompt`` 即使用这条填好的官方语言。接触动作仍可显式覆盖指令，
但应指明目标身份；未解析标记会在策略推理前被拒绝。多物体官方任务不一定指定抓取顺序，
描述性覆盖也不保证 checkpoint 能选择任意实例，必须检查实际抓住的目标。

能力范围与限制
--------------

RoboDojo 提供双臂运动与夹爪原语、三相机 RGB-D、SAM3 感知、XPolicyLab Pi_05
及冻结 Flash 重放。任务名称来自配置的源码目录，例如
``put_bottles_into_dustbin``、``fill_pen_holder`` 和 ``stack_bowls_random``，
并非已验证成功的任务套件。``place_in_bin`` 仅为 ``put_bottles_into_dustbin`` 注册。
尚未实现 handover。低 Z 桌面级与侧向脚本 IK 存在可达性限制，应检查
``reached`` 和 ``dist_to_target``，不要假设指令位姿已到达。
共享的仅评测 planner 见 :doc:`../flash`；重放会执行动作，并非对机器人只读。

有界冒烟与退出诊断
------------------

``fill_pen_holder`` 的冒烟显式使用
``--planner-timeout-s 1500 --max-turns 40``，外层使用
``timeout --signal=INT --kill-after=20s 1700s``。这是验证参数，不是常规默认值；
外框为启动和清理留出时间，并将单次运行限制在 30 分钟内。超时或门禁失败后停止，
先检查最后完成的工具调用与模型服务延迟，再安排下一次尝试。

CLI 将 planner 错误写入 ``transcript_<cell>.json`` 的 ``error`` 字段，
并将包含收尾失败的最终错误写入 ``run_diagnostics.json``。
dev 与 Flash 都应检查这些工件；退出码为零不代表官方任务成功。

完整解码三路视频，并检查每个自有服务的退出码。从
``[robodojo-env] shutdown begin`` 到进程退出，不允许出现 ``[Error]``、
traceback 或 ``Fatal Python error``。Headless GLFW warning 是预期噪音，
不能据此忽略关闭错误。环境在主线程依次释放录像 writer、相机 annotator/render
product、syntheticdata 图句柄，停止 Replicator，最后关闭 stage/app。
