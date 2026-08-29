RoboCasa 全量复现
=================

本页定义 RoboCasa365 Harness VLA 的 fail-closed 正式复现协议。普通单任务开发
流程仍见 :doc:`robocasa`，两者不能混用结果口径。

.. warning::

   当前可用的 local hybrid runtime 只属于 **preliminary** 阶段，可用于接入和
   smoke test，但不是已经发布的不可变复现环境。正式 release 必须发布并冻结
   完整 runtime tree、Python 环境、RLDX checkpoint、VLM metadata、
   RoboCasa365 revision 与 robosuite revision。来自未版本化本地目录的结果不能
   声称为论文复现结果。

冻结的评测矩阵
--------------

协议严格包含 340 个 held-out cell：

.. list-table::
   :header-rows: 1

   * - Split
     - 任务数
     - 评测 seed
     - Cell 数
   * - Atomic
     - 18
     - 1--10
     - 180
   * - Composite seen
     - 16
     - 1--5
     - 80
   * - Composite unseen
     - 16
     - 1--5
     - 80
   * - **合计**
     - **50**
     - 见上
     - **340**

seed 0 从不作为评测 cell。Memory pack 对 43 个任务各保存一对成功的 seed-0
audit/command trace；协议白名单中的另外 7 个 composite-unseen 任务使用刻意为空的目录：
``HeatKebabSandwich``、``PanTransfer``、``PortionHotDogs``、
``SeparateFreezerRack``、``WaffleReheat``、``WashFruitColander``、
``WeighIngredients``。协议没有 Global Memory、``Task.md``、experience
Markdown、跨任务 memory，也不允许为空任务寻找替代 memory。seed-0 两个文件只
提供程序性先验，当前 RGB-D 观测和环境 success 始终具有更高优先级。
schema v3 中 audit JSON 只保存元数据，禁止内嵌任何 command object；配对 JSONL
是唯一命令权威。打包时会移除旧 ``command_sequence``，并根据 JSONL 重新计算
所有已有命令计数。

评测严格禁止 reset。每个 held-out seed 只有一个 live episode，canonical command
trace 仅允许八种物理动作：``navigate_to``、``move_base``、``move_to``、
``move_delta``、``rotate_pitch``、``set_gripper``、``release``、``vla_act``。
Planner 退出不代表成功，正式结果由仿真器和 artifact validator 决定。
仿真器 success 是硬物理终止边界：reset 会采样初始 success predicate，首个成功
step 会锁存 success，之后任何 step 都被拒绝；analytic 与 VLA 多步 helper 也必须
在首个成功 step 立即停止。

版本化 navview patch
--------------------

Runtime 要求
``robosuite/models/assets/bases/omron_mobile_base.xml`` 中有且仅有一台
base-mounted ``navview`` 相机。RPent wheel 内置
``robots/robocasa/patches/robosuite_navview.patch``。该补丁基于
``RLinf/robosuite@85abee228d1c43ab1939bce33028099945d453b4`` 审计，只新增：

.. code-block:: xml

   <camera name="navview" mode="fixed" pos="0.2 0 1.6"
           xyaxes="0 -1 0 0.643 0 0.766" fovy="75"/>

定位 wheel 里的补丁，对干净 checkout 先检查再应用一次：

.. code-block:: bash

   NAVVIEW_PATCH=$(python -c 'from importlib.resources import files; print(files("robots.robocasa").joinpath("patches/robosuite_navview.patch"))')
   ROBOSUITE_CHECKOUT=/path/to/runtime/external_dependencies/robocasa365/robosuite
   git -C "$ROBOSUITE_CHECKOUT" apply --check "$NAVVIEW_PATCH"
   git -C "$ROBOSUITE_CHECKOUT" apply "$NAVVIEW_PATCH"

若 ``--check`` 失败，禁止强制应用。先用 ``git apply --reverse --check`` 判断
是否已经应用；若也失败，说明 checkout 与审计基线不同，必须作为版本问题处理。
下面的 ``doctor`` 会解析安装后的 XML，要求恰好一台 navview，且它必须是
``worldbody/body[@name='base']`` 的直接子节点，并逐项核对上述四个属性。

Memory 与 planner 认证
----------------------

从已审计的 migration 输出构建 task-isolated memory pack，再校验身份、空任务
白名单、精确文件名和 SHA-256 manifest：

.. code-block:: bash

   rpent-reproduce robocasa memory-pack \
     --migration-root /path/to/runtime/migration \
     --output /path/to/memory/harness_vla_v1
   rpent-reproduce robocasa memory-validate \
     --memory-dir /path/to/memory/harness_vla_v1

正式 runner 默认使用 ``RLinf/RPent-memory`` dataset，并自动下载
``robocasa/harness_vla_v1/**``。它仍强制要求不可变的 40 位小写 commit SHA，
通过 ``--memory-revision`` 或 ``RPENT_ROBOCASA_MEMORY_REVISION`` 提供。
如需 staging dataset，可使用 ``--memory-repo-id`` 或
``RPENT_ROBOCASA_MEMORY_REPO_ID`` 明确指定。

RPent 会将该 revision 实体化到私有、无 symlink 的 cache，不直接执行 Hugging
Face blob cache 中的 symlink；可用 ``RPENT_ROBOCASA_MEMORY_CACHE`` 指定 cache
根目录。下载失败、revision 非法或 pack 校验失败都会直接终止，不会隐式回退到
服务器本地 memory。``--memory-dir`` 仅作为显式离线/开发覆盖，且不能与
Hugging Face 参数同时使用。

Planner 认证必须显式选择以下两种互斥模式之一。两种模式都冻结使用
``gpt-5.5`` 与 ``xhigh``；认证方式不会改变 340-cell 矩阵或计分协议。

``api-key`` 是 provider-neutral 模式，可接入任意经过审计、支持冻结 model 与
effort 的 Responses-compatible endpoint。必须同时提供凭据文件和 API base，协议
没有内置 provider 或 endpoint 默认值：

.. code-block:: bash

   API_AUTH_ARGS="--planner-auth-mode api-key \
   --api-key-file /run/secrets/responses-api-key \
   --base-url https://provider.example/v1"

``chatgpt-subscription`` 模式依赖另行部署在同一个 loopback HTTP listener 上的
可信 broker。RPent 只接收一行式 broker client capability；ChatGPT subscription
OAuth 凭据由 broker 在评测 cell 外部持有并刷新：

.. code-block:: bash

   SUBSCRIPTION_AUTH_ARGS="--planner-auth-mode chatgpt-subscription \
   --broker-credential-file /run/secrets/broker-client-capability \
   --broker-base-url http://127.0.0.1:8765/v1 \
   --broker-health-url http://127.0.0.1:8765/health"

上面的 loopback 值只演示 RPent client contract；本仓库不负责部署 subscription
broker，也不声称任何当前 broker 已达到 release-ready。外部部署必须单独完成安全
加固、审计与冻结。启动 cell 前，``doctor`` 要求 health response 明确证明
``provider_profile=chatgpt_subscription_broker``、
``auth_mode=chatgpt_broker``、``credential_broker=true``、
``credential_broker_ready=true``、
``credential_broker_protocol=root_oauth_injection_v1``、``model=gpt-5.5``
以及 ``reasoning_effort=xhigh``；request 与 health URL 必须位于相同的
``127.0.0.1`` 或 ``::1`` listener。

两种模式下，RPent 可见的 credential 都必须是 reproduction 用户拥有的普通
非 symlink 文件，权限精确为 ``0600``，并且只含一个非空行。禁止把 credential
值放进命令、提交到仓库的配置、结果 artifact 或日志。订阅模式下，OAuth token
和 ``auth.json`` 绝不能进入 cell、结果、命令行参数或 RPent credential 文件；
也不能复制、挂载或 symlink 到 cell。Broker client capability 不属于 OAuth
凭据。

RPent 会为两种认证模式冻结完全相同的 Codex provider 重试参数：
``request_max_retries=12``、``stream_max_retries=120`` 与
``stream_idle_timeout_ms=330000``。重试不会延长既有的 cell 绝对 deadline：atomic
cell 仍限制为 1800 秒，composite cell 仍限制为 3600 秒。Broker 与 network-egress
链路仍须持续监管；长期的 provider、broker 或 egress 中断属于基础设施事故，不能
作为 benchmark 证据。

预检与 checkpoint hash
----------------------

Results 根目录保持精确 ``0700``。Rollout 根目录必须由 reproduction 用户拥有，
权限精确为 ``0711``：隔离后的 planner UID 只能穿越已知路径，不能列目录或写入。
Rollout 根的所有祖先也必须允许 other UID 穿越；如果它位于某个 ``0700`` 祖先下，
``doctor`` 会拒绝。每个 cell 的 workdir 仍由独立 planner group 和 Landlock 边界
保护。Results 与 rollout 根目录不得相同，也不得互为祖先。

启动任何 cell 之前先运行 ``doctor``。``--verify-checkpoint`` 会计算三个 RLDX
checkpoint shard 的 hash，并与协议冻结的 SHA-256 对比。Doctor 还检查 runtime
脚本、Codex binary、VLM metadata、memory manifest、secret 文件策略和 navview
XML：

.. code-block:: bash

   export RPENT_ROBOCASA_MEMORY_REVISION="$PUBLISHED_MEMORY_COMMIT"

   COMMON_ARGS="--runtime-root /path/to/runtime \
   --results-root /path/to/results \
   --rollout-root /path/to/rollouts \
   --planner-profile codex-gpt55-xhigh \
   --preliminary-local-runtime"

   # API_AUTH_ARGS 与 SUBSCRIPTION_AUTH_ARGS 必须且只能选择一个。
   PLANNER_AUTH_ARGS="$API_AUTH_ARGS"

   rpent-reproduce robocasa doctor $COMMON_ARGS $PLANNER_AUTH_ARGS \
     --verify-checkpoint --verify-isolation

任何 doctor problem 都是配置失败。应把 JSON 输出与不可变 release manifest 一起
保存，禁止绕过 checkpoint hash 或 navview 不匹配。
``doctor`` 和 ``run`` 对当前 local hybrid snapshot 强制要求
``--preliminary-local-runtime`` acknowledgement。这个 flag 只用于明确标注本地初测，
绝不能用来暗示该 snapshot 已成为正式 release runtime。

当前开发容器不具备 bubblewrap 所需的 mount propagation。外层 RLDX launcher
因此会在 Codex 启动前安装权威 Landlock 文件系统边界，降权到 cell 专属高 UID，
并且只让固定 mailbox 和私有 scratch 目录可写。普通观测与 prompt 文件只以只读
方式开放给该 cell group；``_deadline_commit.gate`` 与
``_deadline_contract.json`` 在整个 workdir 准备过程中保持 ``root:root`` 和精确
``0600``，planner 无法读取。若精确双文件集合、任一 inode 或元数据变化，adapter
会在 Codex 启动前 fail closed。Doctor 仅接受版本精确为
``codex-cli 0.147.0`` 且 SHA-256 与冻结 native binary 一致的 Codex，同时拒绝非空
``/etc/codex`` managed configuration，并要求每个外部 runtime 脚本匹配冻结
SHA-256。固定 Codex 使用命名 profile ``rpent_outer_landlock``。其中 full-write
只是 Codex 自身看到的权限模型，
用于避免再次叠加 bubblewrap 或第二层 Landlock；它不能扩大已经生效且不可撤销的
内核边界和 Unix ownership 检查。``network.enabled=false`` 仍使 Codex 对每条模型
生成命令安装 restricted-network seccomp；direct local tools 则由显式 capability
集合与外层文件系统边界共同约束。兼容性测试必须证明 scratch 可写、cell
根目录与 ``/usr`` 写入被拒、RPent/RLDX/``/proc`` 读取被拒、planner 看不到凭证，
并且 shell 网络 socket 被拒。该 profile 身份写入 run manifest；更换 Codex
binary 后必须重新验证。主动 preflight 还会捕获精确的 direct-tool capability
集合；其 attestation 将 Codex 与 launcher 哈希、profile 参数、kernel release、
Landlock ABI 和全部兼容性检查一起绑定进 run manifest。
默认 executable 固定为 native binary
``/usr/local/libexec/rpent/codex-0.147.0``，其 SHA-256 为
``cb0a15567e9a60a5820d54b0f6ae86d504dc3805c1eab21a47f70e3eb7b73a40``。
运行前必须安装这份由运行用户拥有且 group/other 不可写的精确文件；JavaScript
``/usr/bin/codex`` wrapper 不是等价默认项。

GPU 调度、运行与 resume
--------------------------------

Full runner 在每张 GPU 上启动一个串行 worker。双卡命令为：

.. code-block:: bash

   rpent-reproduce robocasa run $COMMON_ARGS \
     $PLANNER_AUTH_ARGS \
     --selection full --gpus 0,1 \
     --max-attempts 3 --retry-backoff-seconds 60

单卡通过 ``--gpus 0`` 运行。若要求 split 之间存在严格阶段边界，使用同一个
results root，并按以下顺序分别执行：

.. code-block:: bash

   rpent-reproduce robocasa run $COMMON_ARGS $PLANNER_AUTH_ARGS \
     --selection atomic --gpus 0
   rpent-reproduce robocasa run $COMMON_ARGS $PLANNER_AUTH_ARGS \
     --selection composite_seen --gpus 0
   rpent-reproduce robocasa run $COMMON_ARGS $PLANNER_AUTH_ARGS \
     --selection composite_unseen --gpus 0

``full`` 同样按 atomic、composite-seen、composite-unseen 顺序入队。若不允许
上一 split 的尾部 cell 与下一 split 在不同 GPU 上短暂重叠，应使用上述三条独立命令。

preliminary parity 配置使用 2 张 RTX 4090，每张卡仅运行一个串行 simulator/VLA
worker。这是参考配置而不是最低要求；并行度只影响吞吐，不改变冻结矩阵或计分契约。
仓库目前尚未发布经过审计的 340-cell 总耗时或 LLM API 总成本；二者会随 cell 结果、
重试、provider 定价和 token 用量变化。发布这些数据前，应保留 runner JSON 中的
``gpus``、``started_at``、``finished_at``、``elapsed_seconds``，记录精确 GPU
型号、显存和驱动，并以 provider billing export 为成本依据，不能仅按请求数估算。

runner 会在所有 worker 退出前持续持有整个 results root 和每张请求 GPU 的非阻塞
OS 锁。若另一个进程使用相同 results root 或 GPU，它会在调度任何 cell 前失败，
从而避免 held-out episode 重复执行和后写产物覆盖。持久 lock file 不承载状态；
所有权只由仍存活的内核锁决定。

每一次 run（包括单 cell 与 smoke selection）都会在调度前自动执行冻结 checkpoint
hash 校验和主动 isolation preflight。生成的私有 attestations 与
``_run_manifest.json`` 会把 checkpoint、memory、planner、isolation、navview patch
和 runtime implementation 身份绑定到 results root。
这些身份会在每个 cell 前重新计算，而且 cell audit 会逐项与 root manifest
交叉校验。因此长跑期间 runtime、adapter、memory、checkpoint 或 navview 发生
漂移时会立即停止，不会混合 provenance。

同一条命令也是 resume 命令。重启时 runner 以 validator 为唯一事实来源，只跳过
具有 canonical completion manifest、audit/trace 身份匹配、动作均在白名单内且
hash 正确的 cell；缺失、未完成或无效 cell 会重新调度。若存在持久化的
``_FATAL_STOP.json``，必须先调查，不能自动删除。
任何已绑定输入发生变化都必须使用新的空 results root；runner 会拒绝混合 provenance
的 resume，而不是覆盖旧清单。

RPent 自有的 deadline supervisor 会从校验过的 source bytes 加载冻结 planner wrapper，
并让 planner 与 driver 共享同一个绝对 monotonic deadline。primitive 执行完成且
state、log、trace 均已持久化后，driver 必须取得仅 root 可访问的 commit gate 才能发布
``done_NN``。driver 会记录发布后的 monotonic receipt；若发布过程跨过 deadline，
marker 会被撤销，在 deadline 后才到达 gate 的命令也会被拒绝。到期时 supervisor
封印同一 gate、停止 driver，并对连续已提交前缀生成哈希绑定的 freeze attestation。
因此，planner 子进程清理期间才完成的 in-flight primitive 不可能进入 canonical 轨迹。

benchmark 结果由该冻结前缀的最终状态决定。即使 Codex turn 未在 deadline 前退出，
前缀内的 ``success=true`` 仍是 canonical success；否则 timeout 是 canonical benchmark
failure。audit、trace、planner status、deadline contract 与 freeze attestation 必须相互
一致；其中 timeout 证据还必须包含 freeze 所列的全部 state、done marker、command
log、commit receipt 和原始 trace。离线校验会重新计算这些文件的哈希并重建冻结前缀，
而不是只信任汇总字段。timeout 日志可以没有 ``turn.completed``，也可以包含语法受
认可的 reconnect event。非 timeout 场景中的 reconnect 则必须最终恢复到
``turn.completed``；未知 event 或显式失败 event 仍然无效。

Validate、summarize 与发布门槛
----------------------------------------

可以验证单个 cell 或完整矩阵，并输出稳定 summary：

.. code-block:: bash

   rpent-reproduce robocasa validate \
     --results-root /path/to/results \
     --split atomic --task OpenDrawer --seed 1
   rpent-reproduce robocasa validate --results-root /path/to/results
   rpent-reproduce robocasa validate --results-root /path/to/results \
     --require-publication-ready
   rpent-reproduce robocasa summarize \
     --results-root /path/to/results \
     --output /path/to/results/summary.json \
     --require-publication-ready

未加 ``--require-publication-ready`` 时，退出码 0 只表示完整性/一致性验证通过，
不代表获得发布资格。只有完整 validator 还明确报告 ``publication_ready: true``、
``root_release_ready: true``、``expected: 340``、``complete: true``，并且不存在 invalid
或 incomplete cell，结果才具备发布资格。当前 checked-in schema 只接受
preliminary external runtime，因此 ``root_release_ready`` 与
``publication_ready`` 被明确硬置为 false，``--require-publication-ready`` 必须失败。
只有另行审计 formal-runtime schema 与不可变 release assets 后才能开放发布门槛；
修改自签 hash manifest 不能开启它。Canonical failure 也是有效观测，不能
删除或静默重跑来提高分数。必须等全部 340 个 cell 通过 validator 后，才可计算
task-weighted 结果并与论文值比较。Smoke、部分矩阵、resume 尚未完成或 preliminary
local-hybrid 的结果必须明确标注，禁止与 :doc:`../awesome_works/harnessvla`
所述论文 RoboCasa365 task-weighted 结果 55.4% 比较。
