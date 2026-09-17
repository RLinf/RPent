新增机器人后端
==============

.. toctree::
   :maxdepth: 1
   :hidden:

   installation

本指南以 RoboDojo 为完整示例，介绍如何在 ``robots/<name>/`` 新增后端。
它将 Isaac Sim / IsaacLab、双臂 ARX-X5 和 XPolicyLab Pi_05 策略接入
RPent 共享的 planner、感知、工具与 memory 基础设施。该接入仍属实验性：
离线契约测试不代表仿真兼容性或任务成功。安装与可运行的 CLI 示例见
:doc:`installation`。

共享接口请先阅读 :doc:`../../development/add_robot` 和
:doc:`../../development/add_primitive`。仿真接入可对照 :doc:`../libero`、
:doc:`../robocasa` 和 :doc:`../robotwin`；硬件部署与安全要求可参考
:doc:`../dual_franka`。

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

1. 注册后端并管理运行时
------------------------------------------------------------

在 ``robots/<name>/__init__.py`` 导出 ``get_robot_spec`` 和 ``get_toolkit``。
``rpent/robots/base.py`` 按需导入包，无需修改中央注册表。
用户通过 ``rpent --robot <name>`` 选择后端。

在 ``robot_spec.py`` 实现 ``RobotSpec``：提供名称和 prompts，用
``add_cli_args`` 注册参数，``parse_config`` 生成 ``RunConfig``，
``init_runtime`` 启动所需组件。通过 ``primitives_kwargs`` 将运行时参数传入
toolkit。复用 ``try_spawn_server``、``try_wait_server`` 和 ``ProcessDaemon``，
返回自己创建的进程供清理，不停止借用的服务。仅在支持冻结重放时实现
``run_flash(toolkit, cell_tag, note)``。

仿真器和模型应在使用处才导入。源码目录、Python 解释器与服务地址由 CLI/配置
显式传入，不读取开发者工作区文件，不硬编码本地路径。RoboDojo 的
``--source-root``、``--sim-python`` 和 ``--pi05-python`` 展示了这种分离。

2. 定义环境契约
---------------

``env_client.py`` 继承 ``BaseEnvClient``，``env_server.py`` 继承
``BaseEnvFacade``，复用 RPC 路由、元数据校验与进程生命周期管理。
结合调用方测试观测字段、动作单位、reset 归属和返回值结构。
``BaseEnvClient`` 缓存 step 观测，但不转换元组长度：通用文档描述五元组，
RoboDojo 调用方实际使用 ``(obs, reward, done, info)`` 四元组。
RoboDojo reset 返回观测字典，``chunk_step`` 抛出 ``NotImplementedError``，
原语逐步调用 step。不要将这些后端特有结构原样套到不同的调用方。

需要主线程渲染时，参考 RoboDojo，将 ``MainThreadServeMixin`` 放在
``BaseEnvFacade`` 前继承。先初始化 Isaac，再导入仿真模块；仿真操作派发到
主线程，不在 RPC 工作线程中执行。显式测试连接时 reset 与模式元数据，
eval 不得悄悄连接 dev 服务。

3. 组装工具、提示词与任务
------------------------------------------------------------

``toolkit.py`` 负责注册、状态和产物管理，``tools.py`` 实现原语。
状态记录读取、分割与标定深度反投影复用共享 ``perception_tools.py`` 和
SAM3 client。核对相机坐标约定：RoboDojo 使用负光轴 Z，并非所有后端都如此。
运动、夹爪控制和双臂监控保留在所属后端。

非修改型工具标记 ``@readonly``，避免执行后自动追加一次状态采集。
读取缓存观测或执行分割不是机器人动作；RoboDojo 的 ``view_env_state``、
``back_project`` 和 ``segment`` 均如此标记。这不表示返回信息适合评测，
仍需单独分类工具输出。

通过 ``prompts`` 与 ``prompt_bundle`` 接入共享 prompt 构建器。
planner 直接注入工具，按名称调用；不要要求 agent 查找 MCP URL 或发送
JSON-RPC。任务专属放置或评分指导应放在任务上下文，而非通用 system prompt。
在 ``tasks.py`` 实现任务发现；RoboDojo 读取
``<source_root>/task/RoboDojo/config/*.yml``，排除 ``_task.yml``。
发现某个任务不代表已验证其 benchmark 成绩。

4. 复用策略服务
---------------

使用 ``BaseVLAClient`` 和共享入口
``python -m rpent.robots.components.pi05_vla_server``，通过
``--policy-backend rlinf`` 或 ``--policy-backend xpolicylab`` 选择实现。
前者在进程内加载 RLinf 策略，后者通过 ``BaseVLAFacade`` 适配外部
XPolicyLab WebSocket 服务。在 ``robot_spec.py`` 配置入口，不新增后端私有服务。
切换实现不会转换 checkpoint 或观测：RoboDojo 需要 14-DoF 关节动作和三相机输入。
适配轻量客户端时保留 ``pi0_pick`` 的双臂监控语义。

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

能力范围与限制
--------------

RoboDojo 提供双臂运动与夹爪原语、三相机 RGB-D、SAM3 感知、XPolicyLab Pi_05
及冻结 Flash 重放。任务名称来自配置的源码目录，例如
``put_bottles_into_dustbin``、``fill_pen_holder`` 和 ``stack_bowls_random``，
并非已验证成功的任务套件。``place_in_bin`` 仅为 ``put_bottles_into_dustbin`` 注册。
尚未实现 handover。低 Z 桌面级与侧向脚本 IK 存在可达性限制，应检查
``reached`` 和 ``dist_to_target``，不要假设指令位姿已到达。
共享的仅评测 planner 见 :doc:`../flash`；重放会执行动作，并非对机器人只读。

后端接入检查清单
----------------

1. 创建包导出与 ``RobotSpec``，确认未安装仿真器/模型依赖时仍能发现后端并查看
   CLI 帮助。参考 ``tests/unit_tests/robots/test_robodojo_runtime_contracts.py``。
2. 在上述运行时契约测试中，用 fake client/facade 覆盖 reset、step 返回结构、
   不支持的 chunk stepping、元数据与自有进程清理。
3. 注册工具 schema，标记只读调用并分类信息访问。在
   ``tests/unit_tests/robots/test_tool_schema_contracts.py`` 覆盖默认与过滤后的工具组；
   共享感知测试位于
   ``tests/unit_tests/rpent/robots/components/test_perception_tools_contracts.py``。
4. 添加任务上下文和 prompt bundle，测试无关任务不会收到专属指导。
   策略后端选择测试参考
   ``tests/unit_tests/rpent/robots/components/test_pi05_vla_server_contracts.py``。
5. 实现 Flash 时，参考 ``tests/unit_tests/robots/robodojo/test_flash_contracts.py``，
   用 fake state/toolkit 覆盖冻结计划校验、锚点偏移、有界重试与特权隔离。
   只过滤工具名不够，还要审计观测、自动日志和 memory。
6. 在 ``docs/source-en/rst_source/usage/`` 与 ``docs/source-zh/rst_source/usage/``
   添加双语页面、导航，并更新两份 README 和 overview 功能矩阵。
   运行 ``pre-commit run --all-files``、``pytest tests/unit_tests -q``，以及严格构建
   ``make -C docs html LANG=en SPHINXOPTS='-W --keep-going -E'`` 和 ``LANG=zh``。
   依赖与运行时验证遵循 ``CONTRIBUTING.md`` 和 ``tests/README.md``；
   单独报告 GPU、真实策略与仿真验证结果。
