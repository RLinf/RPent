RoboDojo
========

RoboDojo 将 Isaac Sim / IsaacLab、双臂 ARX-X5 和 RLinf Pi0.5 策略接入
RPent 的规划器、工具与记忆系统。该接入仍属实验性，仿真兼容性和任务成功率
需要 GPU 验证。仿真器、CUDA 与策略依赖请按 RoboDojo 和 XPolicyLab 官方说明
安装，并下载所需资产与 checkpoint。共享后端接口见 :doc:`../development/add_robot`。

Python 环境
-----------

目标是将 RoboDojo 安装在同一个环境中。``robodojo-sim`` extra 声明仿真栈：Isaac Sim 5.1、
RoboDojo 使用的 IsaacLab fork、cuRobo 及配套的运行时版本约束；``robodojo`` 在其
之上增加 SAM3 感知、提供 ``pi05_robodojo_arx_x5`` 的 RLinf 版本，以及 openpi
运行时。每个机器人 extra 都带自己那套运行时钉版，因此一个环境只装一个。

最终希望通过 ``uv pip install -e ".[robodojo]"`` 一键安装。目前 Isaac Sim
仍有三处上游钉版与 RPent/RLinf 依赖冲突：``uvicorn==0.29.0`` 与
``mcp`` 要求的 ``uvicorn>=0.31.1``；``wrapt==1.16.0`` 与 RLinf 引入的
``swanlab>=0.6.11`` 要求的 ``wrapt>=1.17.0``；以及 ``filelock==3.13.1`` 与
``rpent-openpi`` 要求的 ``filelock>=3.16.1``。此外，``rpent-openpi==0.2.2``
钉住了 ``torch==2.7.1``，而仿真运行时需要 ``torch==2.7.0``。
这些约束需经 ``rlinf`` 前缀的 fork 在上游修正，才能打通一键安装；
不应在 RPent 中继续添加 override 来绕过。

在上游放宽之前，仿真环境按两步准备：先安装 ``robodojo-sim`` 声明的仿真栈，
再用 ``uv pip install --no-deps -e .`` 将 RPent 安装到同一环境。
这一临时方案不安装完整的 agent/策略依赖栈。请在项目根目录运行 uv，
以读取 ``pyproject.toml`` 中已有的 ``override-dependencies``。

IsaacLab 本身也需要可编辑安装：非 editable 的 VCS 子目录安装只包含
``__init__.py``，会丢失 ``config/extension.toml``，而 ``isaaclab/__init__.py``
通过 ``ISAACLAB_EXT_DIR`` 加载该文件。上述命令只将 RPent 设为可编辑安装，
不会让依赖也变为可编辑安装；还需在同一环境中以可编辑方式安装 IsaacLab 源码包，
才能正确加载配置并使源码修改生效。

这些 Git 引用目前指向 RoboDojo 所依赖的 fork 分支；对应的上游 PR 合并后，
应改回官方分支。

源码与资产
----------

克隆官方仓库及子模块前，先安装 Git LFS：

.. code-block:: bash

   git lfs install
   git clone --recurse-submodules https://github.com/RoboDojo-Benchmark/RoboDojo.git
   cd RoboDojo
   git lfs pull
   git submodule foreach --recursive 'git lfs pull'
   git lfs fsck

对于官方 RoboDojo release 链接的资产和 checkpoint 仓库，同样依次执行克隆、
``git lfs pull`` 和 ``git lfs fsck``，并按该 release 的说明放置文件。
启动仿真器前，确认所需文件是实际数据而非 LFS 指针文本。
release 的 LFS 属性不完整时，请向数据集发布方核实。

RPent 配置
----------

默认的 ``--policy-backend rlinf`` 需要策略解释器提供
``pi05_robodojo_arx_x5`` 配置及其 openpi 依赖；官方 main 尚未包含该配置。
通过 ``PI05_CHECKPOINT_PATH`` 指定兼容的 RLinf checkpoint；
``robodojo`` extra 已提供该策略运行时。使用另行准备的 XPolicyLab 运行时时，
选择 ``--policy-backend xpolicylab``。

通过 ``SAM3_CHECKPOINT_PATH`` 配置 SAM3 checkpoint，并导出摆放稳定步数。
默认值会让物体在 official 模式下不稳定；该变量由 RoboDojo 源码读取，而非
RPent，CLI 会把它传给启动的子服务：

.. code-block:: bash

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000

传入 RoboDojo 源码目录：

.. code-block:: bash

   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0 \
     --source-root /path/to/RoboDojo

``--xpolicylab-root`` 默认使用 ``SOURCE_ROOT/XPolicyLab``；独立克隆时请指定。
单环境下各服务默认使用当前解释器；需要指向其他解释器时，仍可通过
``--sim-python`` 或 ``--pi05-python`` 覆盖。
CLI 构造子进程导入路径，不读取工作区的 ``config/runtime.env``，也不修改父进程环境。
子进程继承已有 shell 环境变量。省略 ``--cuda-device`` 时保留
``CUDA_VISIBLE_DEVICES`` 的原值（包括未设置的状态）；显式传入时，为本地启动的服务选择 GPU。
XPolicyLab 在省略 GPU 参数时直接用 ``--pi05-python`` 启动 Python 策略入口；
显式指定时使用其 shell 启动脚本。

通过 ``--env-endpoint``、``--vla-endpoint`` 和 ``--sam3-endpoint`` 可连接已有服务。
连接已有服务时，该组件不需要本地源码或 Python 路径。
CLI 默认启动共享的 ``rpent.robots.components.pi05_vla_server --embodiment robodojo``。
选择 ``--policy-backend xpolicylab`` 时，改为启动 ``xpolicylab_vla_server``，
并通过 ``--policy-root`` 指定 ``XPolicyLab/policy/Pi_05``。
连接已有 VLA 服务时也应选择匹配的后端。切换后端不会转换 checkpoint；
RLinf 客户端会将原生观测编码为 openpi wire 格式。

每个自有服务的日志与输出都落在本次运行的输出目录：CLI 以 ``--save-dir``
传给环境服务、以 ``--output-dir`` 传给可选的 XPolicyLab 策略入口，内层策略日志为
``vla_server.log``。直接启动服务且省略这些参数时使用当前目录；并发运行应指定不同输出目录。

验证安装
--------

跑一次有界的开发模式 episode，确认规划器接管之前各服务已就绪：

.. code-block:: bash

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000
   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0 \
     --planner codex --model <planner-model> --max-turns 1 \
     --source-root /path/to/RoboDojo \
     --sim-python /path/to/sim-env/bin/python \
     --pi05-python /path/to/pi05-env/bin/python \
     --output-dir /path/to/run-output

预期现象：

* ``/path/to/run-output`` 下出现 ``robodojo_env_server.log``、
  ``sam3_server.log``、``robodojo_vla_server.log``，XPolicyLab 策略服务启动后还会出现
  ``vla_server.log``。
* 环境服务报告 ready，第一条观测包含 ``cam_head``、``cam_left_wrist`` 与
  ``cam_right_wrist`` 的内参、外参，以及关节与夹爪状态。
* 本次运行在 ``/path/to/run-output/videos`` 下为每路相机写一个 MP4。
* 退出后没有遗留的自有子进程，GPU 回到空闲。

启动阶段某个服务退出是最常见的失败形式，先去运行输出目录读它的日志。
Isaac Sim 启动需要数十秒，首次运行还要编译 shader。

主要模块
--------

* ``robots/robodojo/env_server.py`` —— Isaac Sim RPC 服务（主线程渲染、
  三相机 + 深度 + 标定、joint/ee 动作、逐相机视频录制）。
* ``robots/robodojo/env_client.py`` —— 继承 ``BaseEnvClient`` 的 rpent
  侧客户端。
* 共享的 ``rpent/robots/components/pi05_vla_server.py`` 和
  ``rpent/robots/components/pi05_vla_client.py`` —— 默认 RLinf openpi 策略，
  使用 ``robodojo`` embodiment。
* 可选的 ``rpent/robots/components/xpolicylab_vla_server.py`` 和
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

RLinf 客户端将头部、左腕、右腕 RGB 分别映射到
``main_images``、``wrist_images``、``extra_view_images``；``states`` 按左臂 6 维、
右臂 6 维、左夹爪 1 维、右夹爪 1 维拼接，夹爪保留观测原值（1=张开，0=闭合），
``task_descriptions`` 携带指令。

XPolicyLab 适配器原样传递观测与动作，并将
``update_obs``/``get_action`` 与 ``reset`` 串行化，不提供会话隔离。
RoboDojo 要求三相机输入和 14-DoF 关节动作；两种后端均不转换 checkpoint，
也不会将关节动作转换成末端位姿动作。

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

开发运行通过共享的 ``states.json`` 保存动作和观测，将只读感知结果附加到下一个
动作记录，供 Flash 导出使用。已有 ``flash_trace.json`` 仍可导出。
记录可迁移航点时，先对
``cam_head`` 调用 ``segment``，再对返回的 ``centroid_rc``（行、列）调用
``back_project``，然后执行动作。该像素是掩码前景坐标均值向下取整，不是框中心。
录制与重放共用此推导；记录的掩码面积和坐标和用于拒绝被篡改的质心。
不要先批量分割多个对象再批量反投影。每次动作前重复这组感知查询，在评测前导出：

.. code-block:: bash

   python -m robots.robodojo.flash.generate \
     --trace /path/to/dev-run/states.json \
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

任务语言 RPC 与公开观测使用 RoboDojo 已初始化的 description manager。
空指令或残留模板标记会明确报错。官方 instruction 在 eval-fair 中仍是公开信息。
省略 ``pi0_pick.prompt`` 即使用这条填好的官方语言。接触动作仍可显式覆盖指令，
但应指明目标身份；未解析标记会在策略推理前被拒绝。多物体官方任务不一定指定抓取顺序，
描述性覆盖也不保证 checkpoint 能选择任意实例，必须检查实际抓住的目标。

能力范围与限制
--------------

RoboDojo 提供双臂运动与夹爪原语、三相机 RGB-D、SAM3 感知、RLinf Pi0.5（或可选 XPolicyLab）
及冻结 Flash 重放。任务名称来自配置的源码目录，例如
``put_bottles_into_dustbin``、``fill_pen_holder`` 和 ``stack_bowls_random``，
并非已验证成功的任务套件。``place_in_bin`` 仅为 ``put_bottles_into_dustbin`` 注册。
尚未实现 handover。低 Z 桌面级与侧向脚本 IK 存在可达性限制，应检查
``reached`` 和 ``dist_to_target``，不要假设指令位姿已到达。
共享的仅评测 planner 见 :doc:`flash`；重放会执行动作，并非对机器人只读。

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
