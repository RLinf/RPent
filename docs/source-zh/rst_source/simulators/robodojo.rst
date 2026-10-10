RoboDojo
========

RoboDojo 将 Isaac Sim / IsaacLab、双臂 ARX-X5 和 RLinf Pi0.5 策略接入
RPent 的规划器、工具与记忆系统。该接入仍属实验性，仿真兼容性和任务成功率需\
要 GPU 验证。仿真器、CUDA 与策略依赖请按 RoboDojo 和 XPolicyLab 官方说明安\
装，并下载所需资产与 checkpoint。共享后端接口见 :doc:`../development/add_robot`。

Python 环境
-----------

在 Python 3.11 环境中，从 RPent 根目录安装运行时与 agent 依赖。
``robodojo-sim`` extra 安装 ``rlinf-robodojo-runtime`` 和 RLinf 环境适配器，
Isaac Sim / IsaacLab 的版本约束由运行时包维护。``robodojo`` 还会安装 SAM3
与 openpi 策略运行时。每个环境只安装一个机器人 extra。在此目录运行 uv，以读取\
根项目的 cuRobo 构建依赖，并显式传入本仓库的仿真 override：

.. code-block:: bash

   uv pip install -e ".[robodojo]" --extra-index-url https://pypi.nvidia.com \
     --override requirements/robodojo-override.txt

Blackwell GPU 需在命令末尾添加 ``--torch-backend=cu128``，选择支持
``sm_120`` 的 PyTorch 构建。PyTorch 2.7.0 默认的 CUDA 12.6 构建不支持这\
类 GPU。重新安装该 extra 时也需保留此选项。
uv 会把已装的 ``2.7.0+cu126`` 视作满足钉住的 ``torch==2.7.0`` 而不替换，\
因此在已有环境上换后端还需补 ``--reinstall-package``，对 torch 与 torchvision \
各传一次；或直接重建环境。

``requirements/robodojo-override.txt`` 汇总了仿真栈与 agent 栈、rpent-openpi 冲突的八条钉\
版本；用 ``--override`` 传入后，其它机器人仍按各自验证过的版本解析。这些 override 使依\
赖可以解析，不代表仿真任务成功。RLinf 集成分支同时提供\
环境适配器和 ``pi05_robodojo_arx_x5``。RoboDojo 预设选择 OpenPI 的 ``eval`` loader，\
预测长度为 50，模型动作补齐到 32 维，环境动作保持 14 维。\
真实权重的 RPC 推理已返回形状为 ``(1, 50, 14)`` 且全部有限的动作；这不保证任务成功。

仿真桥需要运行时包 0.3.0 或更新版本。发布版本号之前，暂时使用 Git 引用。全\
新依赖安装与 GPU rollout 仍需分别验证；桥的离线测试不代表仿真兼容性已获验证。
IsaacLab 上游的非 editable 打包可能遗漏扩展配置。如果所安装的版本仍有此问题，需\
另行准备 editable IsaacLab 安装；运行时 wheel 不修复 IsaacLab 的打包问题。当\
前固定的 ``afca7b09`` 版本在 wheel 安装后确实会因缺少
``config/extension.toml`` 失败。将运行时依赖指定的
`yuechen0614/IsaacLab <https://github.com/yuechen0614/IsaacLab.git>`_ 及对应版本检\
出到独立的可写目录，然后在 RPent 根目录、同一环境中执行：

.. code-block:: bash

   uv pip install --no-deps \
     -e /path/to/IsaacLab/source/isaaclab \
     -e /path/to/IsaacLab/source/isaaclab_assets \
     -e /path/to/IsaacLab/source/isaaclab_tasks

该步骤在安装后手工执行一次；使用此环境期间需保留该源码目录。

源码与资产
----------

运行时 wheel 已包含验证过的 RoboDojo 代码树，无需克隆 RoboDojo。环\
境默认使用 ``robodojo_runtime.source_root()``；仍可用 ``--source-root``
显式指定本地 checkout。场景数据独立于 Python 依赖下载。将 \
``ROBODOJO_ASSETS_ROOT`` 指向包含 ``Assets/`` 的目录：

* **RoboDojo 机器人、物体、材质和布局资产：** 只下载
  `RoboDojo 数据集 <https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo>`_
  中的 ``Assets/**``。已有 Hugging Face CLI 时可执行：

  .. code-block:: bash

     hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
       --include 'Assets/**' --local-dir /data/robodojo
     export ROBODOJO_ASSETS_ROOT=/data/robodojo

  运行时读取 ``ROBODOJO_ASSETS_ROOT/Assets``，无需向安装目录创建符号链接。
  确认 ``Robots``、``Object``、``Material`` 和 ``Eval_Layout`` 下是实际文件，
  而非 LFS 指针。
* **IsaacLab 引用的 NVIDIA USD/材质资产：** 它们不在 ``isaaclab_assets`` 包内。
  按 `NVIDIA 资产下载说明 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_faq.html#isaac-sim-setup-assets-content-pack>`_
  获取对应版本。RoboDojo 的 ``utils/ensure_usd_path.py`` 仅改写
  ``Assets/Isaac/5.0`` 前缀的 URL：Isaac Sim 5.1 仍在引用这批 5.0 资产路径，
  因此这些引用需要 **5.0** 资产包；直接引用 5.1 路径的场景或扩展则需要
  5.1 资产包。两个资产包的文件树不同，不能互相顶替。对前者，
  保留 ``/data/nvidia/Assets/Isaac/5.0`` 目录结构，并导出：

  .. code-block:: bash

     export ROBODOJO_USD_ASSET_PREFIX=/data/nvidia

  这一上游已有变量只适用于上述 URL 前缀，并不能覆盖所有 IsaacLab 资产或
  5.1 资产目录。其他 IsaacLab 引用使用 Kit 的
  ``/persistent/isaac/asset_root/cloud`` 设置；离线运行时需另行配置匹配的本地资产包。
  5.1 资产包不能替代这批 5.0 路径。

依赖中这两个名称相近的条目 **并不是两份场景数据包**：
``isaacsim[all,extscache]`` 提供仿真器二进制和扩展缓存（Linux x86-64 /
CPython 3.11 的 5.1.0 Kit 与 Kit-SDK 缓存 wheel 就分别有
3,021,340,845 和 1,345,115,764 字节）。保留这部分运行时安装，不能用数据路径环\
境变量替代。``afca7b09`` 的 ``isaaclab_assets`` 仅含 119,143 字节源码和配置，没\
有 USD 文件，因此也保留。IsaacLab 和 cuRobo 的 Git 文件树分别约为
53.6 MB、129.9 MB（后者包含机器人网格）；这是未压缩文件总量，不是实测克隆体积。
cuRobo 随包提供的网格仍属于其运行时。因\
此 uv 仍会下载较大的仿真运行时，但不会下载上述独立场景数据集或策略 checkpoint。
checkpoint 也需单独获取，按下节用 ``PI05_CHECKPOINT_PATH`` 和
``SAM3_CHECKPOINT_PATH`` 指定。

策略与感知 checkpoint
------------------------------

RoboDojo 发布的 Pi_05 权重是 orbax/JAX checkpoint，而 RLinf 的 openpi loader 读的是\
PyTorch 版本，因此下载后需要转换一次。感知使用 SAM 3，需要单独的 checkpoint。

.. code-block:: bash

   # 1. 下载发布版 Pi_05 checkpoint（约 7 GB）。下载根目录与资产保持一致：
   #    hf download 会保留仓库内的路径，写成 /data/robodojo/ckpt 会多套一层 ckpt/。
   hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
     --include 'ckpt/RoboDojo/Pi_05/**' --local-dir /data/robodojo

   # 2. 转换。发布目录名为 Pi_05，而转换器靠路径中是否含小写 "pi05" 选择分支，
   #    所以先建一个含该字符串的链接再转换。
   ln -s /data/robodojo/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0 \
     /data/robodojo/pi05_robodojo_arx_x5
   python -m rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python \
     --checkpoint-dir /data/robodojo/pi05_robodojo_arx_x5/59999 \
     --config-name pi05_aloha \
     --output-path /data/robodojo/pi05_robodojo_arx_x5_torch

   # 3. 指向转换后的 checkpoint 与 SAM 3。
   export PI05_CHECKPOINT_PATH=/data/robodojo/pi05_robodojo_arx_x5_torch
   export SAM3_CHECKPOINT_PATH=/data/sam3/sam3.pt

转换器属于 RLinf，随 ``robodojo-sim`` 安装的 ``rlinf`` 包一起落地，因此用模块方式调\
用；RPent 检出里没有 ``rlinf/utils/`` 目录。``--checkpoint-dir`` 要指向 ``59999`` 这一\
步目录，因为转换器读取 ``<checkpoint-dir>/params``。发布权重是 Pi_05 aloha 结构（32 \
维动作、预测长度 50），正是随 extra 一同安装的 openpi 注册表里 ``pi05_aloha`` 所描述的\
结构；XPolicyLab 自带的 openpi 里那个 ``..._arx-x5_seed_0`` 名字对应同一结构，但转\
换器导入的不是那份注册表。

发布版随包携带的归一化统计量要与转换后的权重放在一起，客户端会一并加载。\
``--inspect_only`` 只打印 orbax 参数键、不做转换，是检查下载结果最快的方式。

RPent 配置
----------

默认的 ``--policy-backend rlinf`` 需要策略解释器提供
``pi05_robodojo_arx_x5`` 配置及其 openpi 依赖；官方 main 尚未包含该配置。通\
过 ``PI05_CHECKPOINT_PATH`` 指定兼容的 RLinf checkpoint 及其归一化统计量。
``robodojo`` extra 选择包含所需预设的集成分支。使用另行准备的 XPolicyLab 运行时时，选\
择 ``--policy-backend xpolicylab``。

通过 ``SAM3_CHECKPOINT_PATH`` 配置 SAM3 checkpoint，并导出摆放稳定步数。默\
认值会让物体在 official 模式下不稳定；该变量由 RoboDojo 源码读取，而非
RPent，CLI 会把它传给启动的子服务：

.. code-block:: bash

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000

使用包内代码启动：

.. code-block:: bash

   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0

运行时 wheel 不含 XPolicyLab。``--xpolicylab-root`` 默认使用
``SOURCE_ROOT/XPolicyLab``；独立克隆时请指定。单\
环境下各服务默认使用当前解释器；需要指向其他解释器时，仍可通过
``--sim-python`` 或 ``--pi05-python`` 覆盖。
CLI 构造子进程导入路径，不读取工作区的 ``config/runtime.env``，也不修改父进程环境。子\
进程继承已有 shell 环境变量。省略 ``--cuda-device`` 时保留
``CUDA_VISIBLE_DEVICES`` 的原值（包括未设置的状态）；显式传入时，\
为本地启动的服务选择 GPU。
XPolicyLab 在省略 GPU 参数时直接用 ``--pi05-python`` 启动 Python 策略入口；显\
式指定时使用其 shell 启动脚本。

通过 ``--env-endpoint``、``--vla-endpoint`` 和 ``--sam3-endpoint`` 可连接已有服务。连\
接已有服务时，该组件不需要本地源码或 Python 路径。
CLI 默认启动共享的 ``rpent.robots.components.pi05_vla_server --embodiment robodojo``。选\
择 ``--policy-backend xpolicylab`` 时，改为启动 ``xpolicylab_vla_server``，并\
通过 ``--policy-root`` 指定 ``XPolicyLab/policy/Pi_05``。连\
接已有 VLA 服务时也应选择匹配的后端。切换后端不会转换 checkpoint；
RLinf 客户端会将原生观测编码为 openpi wire 格式。

每个自有服务的日志与输出都落在本次运行的输出目录：CLI 以 ``--save-dir``
传给环境服务、以 ``--output-dir`` 传给可选的 XPolicyLab 策略入口，内层策略日志为
``vla_server.log``。直接启动服务且省略这些参数时使用当前目录；\
并发运行应指定不同输出目录。

验证安装
--------

跑一次有界的开发模式 episode，确认规划器接管之前各服务已就绪：

.. code-block:: bash

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000
   export OMNI_KIT_ACCEPT_EULA=YES
   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0 \
     --planner codex --model <planner-model> --max-turns 1 \
     --sim-python /path/to/sim-env/bin/python \
     --pi05-python /path/to/pi05-env/bin/python \
     --output-dir /path/to/run-output

``OMNI_KIT_ACCEPT_EULA=YES`` 用于回应 Kit 的许可询问；非交互启动时不设置它会停在询问\
处并直接退出。

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

* ``robots/robodojo/rlinf_env.py`` —— 基于 RLinf ``RoboDojoEnv`` 的 agent 适配器，
  提供视频录制、相机元数据和回合诊断。
* ``robodojo_runtime/bridge.py`` —— 仿真器创建、重置、观测与动作执行，
  由 RLinf 惰性导入。
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
  place_in_bin 等原语。
* ``robots/robodojo/robot_spec.py`` —— RobotSpec 工厂（CLI、RunConfig、
  运行时编排）。
* ``robots/robodojo/tasks.py`` —— 从指定源码目录读取任务列表。

实现细节
--------

环境服务先初始化 Isaac Sim，再导入仿真模块；为保证相机渲染正常，仿真请求在主\
线程串行执行。每个进程持有一个仿真应用。reset 返回观测字典，step 返回
``(obs, reward, done, info)`` 四元组。不支持 chunk stepping，原语通过环境动\
作接口逐步执行，保留该接口的边界检查与计数。

RLinf 客户端将头部、左腕、右腕 RGB 分别映射到
``main_images``、``wrist_images``、``extra_view_images``；``states`` 按左臂 6 维、右\
臂 6 维、左夹爪 1 维、右夹爪 1 维拼接，夹爪保留观测原值（1=张开，0=闭合），
``task_descriptions`` 携带指令。

XPolicyLab 适配器原样传递观测与动作，并将
``update_obs``/``get_action`` 与 ``reset`` 串行化，不提供会话隔离。
RoboDojo 要求三相机输入和 14-DoF 关节动作；两种后端均不转换 checkpoint，也\
不会将关节动作转换成末端位姿动作。

工具与信息访问
--------------

planner 直接提供工具，按工具列表中的名称调用即可。垃圾桶放置与瓶子恢复操作指导仅在
``put_bottles_into_dustbin`` 的任务上下文中提供，不放入通用 system prompt。状\
态记录读取和标定深度反投影在 ``robots/robodojo/tools.py`` 内实现；反投影采用
Isaac 的负光轴 Z 约定，分割使用共享 SAM3 client。``view_env_state``、
``back_project`` 和 ``segment``
均为只读调用，不推进环境，也不触发动作后的状态采集。

``robots.robodojo.tools`` 中声明的每个工具都针对所有任务注册，不按任务名过滤。任何 planner 工具都不暴露 reward 明细或基于真值的安全告警，\
planner 只根据观测判断进展。底层 ``env.get_reward_details`` 和
``env.get_safety_status`` RPC 仍在 dev 服务中保留，供外部评测与诊断使用，不注册为 \
planner 工具。reward 与官方 success 仅由评测路径在 planner 动作通道关闭后读取。
``finalize_run`` 写入 runner 提供的结果，本身不调用这些 RPC。

工具可见性不是隔离机制：共享 toolkit 暴露的 schema、处理函数、动作后的自动状态、原始\
观测字段、日志、memory 与通用文件工具都保持不变；只有 ``--planner flash`` 选择的 \
eval-fair 模式会把机器人工具收窄到重放所需的一组。

开发与冻结重放
--------------

普通 planner 保留开发工具集合，但不提供评分或安全诊断。``--planner flash`` 选择 eval-fair，由 \
RPent 原生 Flash planner 调用 ``RobotSpec.run_flash``，LLM 不进环。
RoboDojo 仍不支持 ``--explore``；这里的开发指普通 planner 循环，并非该 CLI 模式。

开发运行通过共享的 ``states.json`` 保存动作和观测，将只读感知结果附加到下一个动\
作记录，供 Flash 导出使用。已有 ``flash_trace.json`` 仍可导出。记\
录可迁移航点时，先对
``cam_head`` 调用 ``segment``，再对返回的 ``centroid_rc``（行、列）调用
``back_project``，然后执行动作。该像素是掩码前景坐标均值向下取整，不是框中心。录\
制与重放共用此推导；记录的掩码面积和坐标和用于拒绝被篡改的质心。不\
要先批量分割多个对象再批量反投影。每次动作前重复这组感知查询，在评测前导出：

.. code-block:: bash

   python -m robots.robodojo.flash.generate \
     --trace /path/to/dev-run/states.json \
     --task put_bottles_into_dustbin \
     --destination /path/to/memory/robodojo/flash

版本 2 JSON 包含任务、SAM3 符号查询、显式的 ``sam3_mask_centroid_floor_v1``
推导方法、可选腕部精定位设置，以及有序动作、参数与三\
维相对偏移，不保存参考物体坐标、分数或谓词结果。导\
出支持 ``move_to``、``set_gripper`` 和 ``pi0_pick``；不支持的动作（\
包括 ``place_in_bin``、``stabilize``）、失败调用或缺少锚点的航点会被拒绝，应\
在开发期将这些操作记录为支持的基础动作。导出不会覆盖已有计划。评\
测前审核并冻结计划；重放期间不筛选候选计划，也不写入计划。版\
本 1 框中心计划和缺少掩码矩的轨迹必须重新录制，不会静默转换。共\
享 XPolicyLab facade 只持有自己启动的策略进程，并在 close、启动失败、
SIGTERM 和解释器正常退出时终止它们；借用的策略服务保持运行。

使用正常运行时参数并指定：

.. code-block:: bash

   rpent --robot robodojo --task put_bottles_into_dustbin --layout 1 \
     --planner flash --memory-profile local --memory-dir /path/to/memory/robodojo \
     --sim-python /path/to/sim-env/bin/python \
     --pi05-python /path/to/pi05-env/bin/python

计划位于选定机器人 memory 的 ``flash/<task>_plan.json``。
HF 模式复用原生 ``robodojo/flash/**`` 同步过滤；仓库不附带 RoboDojo 计划，也\
不保证 HF 已发布这些计划。计划缺失或无效时拒绝重放。

重放先在开头头部画面定位全部锚点，航点由实时锚点加记录偏移得到，偏移长度上限
0.5 m。导出时可传 ``--refine-camera cam_left_wrist`` 或 ``cam_right_wrist``，要\
求运动前用腕部画面精定位；与头部定位相差超过 5 cm 则停止。精\
定位不会主动移动腕部以获得视野；应记录能提供该视野的接近动作，或不启用精定位。
``pi0_pick`` 语义不变，重放额外检查夹爪闭合且腕部定位的物体距任一末端不超过
12 cm。未保持抓取时重新用头部定位并重放前一接近动作，最多尝试抓取三次。调\
用失败、定位丢失、航点不可达或步数耗尽均停止，不重置场景；这些检查不保证碰撞安全。

eval-fair 额外移除通用文件/memory 工具与任务专用辅助工具。服务元数据标明模式，拒绝 reset
和诊断 RPC，返回零 reward 且不返回任务成功反馈；观测只含 RGB-D、相机标定、公\
开指令和双臂本体状态，不启用瓶子真值告警。eval-fair 拒绝连接 dev 服务；启\
动时创建回合，eval 客户端不再重复 reset。自动状态日志因此只包含公开观测与动作诊断。精\
简后的 eval prompt 不含评分指导，Flash 也不会使用这些 prompt。

Flash 的 ``done`` 和 planner 完成状态仅表示冻结动作序列执行完毕，不表示官方任务谓\
词通过。官方评分必须与重放上下文隔离；宣称 benchmark 兼容或成功前仍需验证
GPU、真实策略服务与仿真端到端。

任务语言
~~~~~~~~

任务语言 RPC 与公开观测使用 RoboDojo 已初始化的 description manager。空\
指令或残留模板标记会明确报错。官方 instruction 在 eval-fair 中仍是公开信息。省\
略 ``pi0_pick.prompt`` 即使用这条填好的官方语言。接触动作仍可显式覆盖指令，但\
应指明目标身份；未解析标记会在策略推理前被拒绝。多物体官方任务不一定指定抓取顺序，描\
述性覆盖也不保证 checkpoint 能选择任意实例，必须检查实际抓住的目标。

能力范围与限制
--------------

RoboDojo 提供双臂运动与夹爪原语、三相机 RGB-D、SAM3 感知、RLinf Pi0.5（或可选 XPolicyLab）及\
冻结 Flash 重放。任务名称来自配置的源码目录，例如
``put_bottles_into_dustbin``、``fill_pen_holder`` 和 ``stack_bowls_random``，并\
非已验证成功的任务套件。``place_in_bin`` 仅为 ``put_bottles_into_dustbin`` 注册。尚\
未实现 handover。低 Z 桌面级与侧向脚本 IK 存在可达性限制，应检查
``reached`` 和 ``dist_to_target``，不要假设指令位姿已到达。共\
享的仅评测 planner 见 :doc:`../guides/flash`；重放会执行动作，并非对机器人只读。

有界冒烟与退出诊断
------------------

``fill_pen_holder`` 的冒烟显式使用
``--planner-timeout-s 1500 --max-turns 40``，外层使用
``timeout --signal=INT --kill-after=20s 1700s``。这是验证参数，不是常规默认值；外\
框为启动和清理留出时间，并将单次运行限制在 30 分钟内。超时或门禁失败后停止，先\
检查最后完成的工具调用与模型服务延迟，再安排下一次尝试。

CLI 将 planner 错误写入 ``transcript_<cell>.json`` 的 ``error`` 字段；运行结果\
由各 sim 自己的收尾过程写出（RoboDojo 在运行输出目录写 ``result.json``），\
收尾过程抛错会记录到日志并反映在进程退出码上。
dev 与 Flash 都应检查这些工件；退出码为零不代表官方任务成功。

完整解码三路视频，并检查每个自有服务的退出码。从
``[robodojo-env] shutdown begin`` 到进程退出，不允许出现 ``[Error]``、
traceback 或 ``Fatal Python error``。Headless GLFW warning 是预期噪音，不\
能据此忽略关闭错误。环境在主线程依次释放录像 writer、相机 annotator/render
product、syntheticdata 图句柄，停止 Replicator，最后关闭 stage/app。
