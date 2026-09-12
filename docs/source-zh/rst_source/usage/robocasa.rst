RoboCasa
========

`RoboCasa <https://robocasa.ai>`_ 是面向厨房场景的长时序操作仿真环境。
在 RPent 中由 **RLDX-1** VLA 策略驱动，默认通过 HTTP RPC 提供服务
（与 LIBERO 一致），也支持 pickle-framed socket 传输。详见
``robots/robocasa/vla_server.py`` 与 ``robots/robocasa/robot_spec.py``
中的传输选择逻辑。

.. note::

   公开的 Target50 协议固定在 ``robots/robocasa/eval/target50.json`` 中。
   340 个 cell 均使用普通的单任务 ``rpent --robot robocasa`` 命令。

运行流程
--------

RoboCasa365 使用 PandaOmron 移动机械臂与冻结的 RLDX-1 策略。
接入层与 planner 无关，API planner、Claude Code 和 Codex 使用相同的 RoboCasa
工具接口。后端配置参见 :doc:`configure_planner`；凭据由用户在仓库外提供。

.. code-block:: text

   rpent CLI -> task-memory sync -> environment and VLA servers
             -> planner toolkit -> final environment state.success

未指定外部 endpoint 时，RPent 为每次运行启动环境服务器和 VLA 服务器。
Planner 根据实时任务语言和观测选择 primitive，RLDX-1 执行操作技能。
评测成功只取环境自身 ``_check_success()`` 暴露的 ``state.success``。
公开协议使用普通单-cell 命令，不包含批量启动器；Harness VLA 的总体说明参见
:doc:`../awesome_works/harnessvla`。

安装
----

RLDX-1 要求 Python ``3.10``。请创建独立环境，并通过 ``.[robocasa]``
安装完整的 RoboCasa365 运行栈：

.. code-block:: bash

   uv venv --python 3.10
   source .venv/bin/activate
   uv pip install -e ".[robocasa]" \
      --constraint robots/robocasa/eval/target50-constraints.txt \
      --override robots/robocasa/eval/target50-overrides.txt \
      --torch-backend=cu126
   uv pip check

RoboCasa 专用 constraints 文件固定经 Target50 复现验证的兼容性敏感包版本，
同时不会收窄 RPent 中 LIBERO 或 RoboTwin 的共享依赖。配套 override 文件让
正式环境直接解析到不可变的 Robosuite revision，而普通 ``robocasa`` extra
仍跟随维护中的 ``rpent`` 分支。该命令让 uv 只为 Torch 选择官方 CUDA wheel，
避免把 PyTorch wheel 源作为通用 ``--index`` 后，在默认 first-index 策略下误选
其中的旧版无关依赖。上面的 ``cu126`` 是已验证的 CUDA 安装方式；仅当宿主机
确有需要时才切换到其他受支持的 Torch backend。

国内网络可使用 PyPI 镜像加速：\

.. code-block:: bash

   uv pip install -e ".[robocasa]" \
      --constraint robots/robocasa/eval/target50-constraints.txt \
      --override robots/robocasa/eval/target50-overrides.txt \
      --default-index https://mirrors.aliyun.com/pypi/simple \
      --torch-backend=cu126

.. note::

   flash-attn 是可选的，缺少时 RLDX-1 会回退到 PyTorch SDPA。若要加快策略
   前向，可安装预编译 wheel —— PyPI 上只有 sdist，直接
   ``pip install flash-attn`` 会源码编译 10-20 分钟：

   .. code-block:: bash

      uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

   该 wheel 只带 SM_80 与 SM_90 kernel；Blackwell (``sm_120``) 需从源码编译，
   或继续使用 SDPA。

**安装后处理**

将厨房 assets（约 10 GB）下载到 ``site-packages`` 之外，重装不会丢。
Target50 不使用 RoboCasa dataset 或 teleop macros，因此跳过可选的 private
macros 配置：

.. code-block:: bash

   robocasa-download-assets --assets-path ~/.robocasa/assets --no-macros -y

命令结束时会打印需要导出的环境变量，把它加到启动 ``rpent`` 的 shell 里：

.. code-block:: bash

   export ROBOCASA_ASSETS_PATH=~/.robocasa/assets

外置根目录需要同时包含六类下载资源，以及发行包内的 scene、arena 和 fixture
静态文件。修正后的安装器会补齐静态文件，不覆盖内容不同的已有文件。
``--skip-existing`` 会检查成功下载的文件清单，不再把非空目录当作完整安装。
请保留官方 attribution 文件，并在实验开始前完成中断的下载。

资源以原子方式发布，中断复制不会留下半写入的正式文件。若旧安装器已经留下
内容冲突的资源，可在原命令中显式增加覆盖权限进行修复：

.. code-block:: bash

   robocasa-download-assets --assets-path ~/.robocasa/assets --no-macros --overwrite -y

``--overwrite`` 优先于 ``--skip-existing``，仅替换本次安装范围内的资源文件，
不删除无关文件或整个目录；未指定时，内容不同的已有文件仍受保护。
默认原子不覆盖发布要求目标文件系统支持硬链接。强制终止遗留的临时文件不会
阻止重试。

新资源集合主要需要 ZIP 与一份解压数据的空间，staging 发布时不再复制 payload；
替换已有安装时还需为旧资源占用预留空间。``--skip-existing`` 会跳过已完成集合的
下载和内容比较，随包静态文件单独检查。

**移动相机**

``robocasa`` extra 会安装 ``RLinf/robosuite`` 的 ``rpent`` 分支，该分支
包含 Omron 底盘固定的 ``navview`` 相机，其组合后的 MuJoCo 相机名为
``mobilebase0_navview``。导航 RGB-D 与 world map 渲染会在首次请求时验证该
相机，并在缺失时明确报错。无需手工修改
``site-packages`` 中的 XML。Target50 将 Robosuite 固定为
``97cfbde4b68d8ec43dad20cf4747297866a6ca2e``；上面安装命令中的 Target50
override 会直接选中这一 revision。

**RLDX-1 checkpoint**

下面运行命令的 ``--vla-model-path`` 期望一个本地 ``RLDX-1-FT-RC365``
checkpoint 路径（RoboCasa365 微调版）。从 HuggingFace 下载:

.. code-block:: bash

   hf download RLWRLD/RLDX-1-FT-RC365 \
      --revision 587e9ecdcc5e7184fcc17f58713908edff5af041 \
      --local-dir ./checkpoints/rldx-1-ft-rc365

下载慢的话用 HF 镜像:

.. code-block:: bash

   HF_ENDPOINT=https://hf-mirror.com hf download RLWRLD/RLDX-1-FT-RC365 \
      --revision 587e9ecdcc5e7184fcc17f58713908edff5af041 \
      --local-dir ./checkpoints/rldx-1-ft-rc365

**RLDX-1 backbone 支持文件**

FT checkpoint 虽包含权重，仍引用 ``RLWRLD/RLDX-1-VLM`` 的架构、processor
和 tokenizer。Target50 将此第四类资源固定到
``4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7``，共 15 个非权重文件，约
16.4 MB（包括模型文档和图片）。下载到启动 RPent 时使用的同一缓存：

.. code-block:: bash

   export HF_HOME="$PWD/.cache/huggingface"
   export HF_HUB_CACHE="$HF_HOME/hub"
   hf download RLWRLD/RLDX-1-VLM \
      --revision 4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7 \
      --include "*.json" "*.txt" "*.jinja" "*.md" "*.png" ".gitattributes" \
      --exclude "*.safetensors.index.json"

无需额外下载基础模型权重。启动时保留上述缓存变量，不要通过
``TRANSFORMERS_CACHE`` 指向空缓存。正式命令使用 ``--vla-backbone-revision``
固定实际加载版本，不依赖缓存中的 ``main`` 引用。连接外部 VLA worker 时，应在
该服务上配置 ``--backbone-revision``；普通在线运行可以省略此参数。
模型和 assets 的许可证独立于 RPent 代码许可证。

**任务 Memory**

通过 ``--memory-profile hf`` （默认值）启用自动同步。每次以该 profile 普通
运行前，RPent 都会通过统一 memory manager，从
`RLinf/RPent-memory 数据集
<https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/robocasa/results>`_
自动同步 ``robocasa/**`` 到 ``memory/robocasa``，因此在线普通运行无需单独下载
memory。当前任务只能读取 ``results/`` 下与该任务对应的 memory：

.. code-block:: text

   memory/robocasa/results/<Task>_s0.json
   memory/robocasa/results/recipe_<Task>_s0.jsonl
   memory/robocasa/results/<Task>.md  # 可选

最终发布的 corpus 包含 43 个 audit JSON、43 个 recipe JSONL 和 25 个任务
Markdown，共 111 个文件且不含 global memory。JSON/JSONL pair 保存经过审核的
seed-0 证据。可选 Markdown 保存同任务探索 memory，可能汇总多次尝试；全部
16 个 Composite-Seen 和 9 个 Composite-Unseen 任务包含该文件。Prompt 要求
planner 在开始动作前主动通过 ``read_text_file`` 读取当前任务所有存在的文件；
RPent 不会把 Markdown 内容强制注入 prompt。

RoboCasa 不要求 planner 使用 global memory，也不会退回读取其他任务的 memory。
7 个 Composite-Unseen 任务完全没有 task memory，但仍计入评测：
``HeatKebabSandwich``、``PanTransfer``、``PortionHotDogs``、
``SeparateFreezerRack``、``WaffleReheat``、``WashFruitColander`` 和
``WeighIngredients``；这些任务基于实时观测继续。Memory 仅是策略证据，历史
坐标、位姿、像素和子任务 prompt 不能替代当前定位与完整实时任务语言。

普通运行同步 Hugging Face ``main``；正式 Target50 使用固定 memory snapshot
``551fc3157b3e56b40a3d3a3b4c7ff81721ebe89b``：

.. code-block:: bash

   hf download RLinf/RPent-memory \
      --repo-type dataset \
      --revision 551fc3157b3e56b40a3d3a3b4c7ff81721ebe89b \
      --include "robocasa/**" \
      --local-dir ./target50-memory

随后选择 local profile，并传入固定的 RoboCasa memory 根目录：

.. code-block:: bash

   rpent --robot robocasa --task-name OpenDrawer --seed 1 \
         --vla-model-path ./checkpoints/rldx-1-ft-rc365 \
         --planner claude_code --model claude-opus-4-8 \
         --memory-profile local \
         --memory-dir ./target50-memory/robocasa

Harness VLA Target50 复现协议
-----------------------------

``robots/robocasa/eval/target50.json`` 是 Harness VLA 在 RoboCasa Target50
上的规范复现清单。它固定 ``target`` 环境 split、依赖 revision、memory 边界、
task/seed 矩阵、cell 时限、成功来源与重试规则；协议 ID 为
``robocasa-harness-vla-v1``：

.. list-table:: RoboCasa Target50 矩阵
   :header-rows: 1
   :widths: 30 15 20 20 15

   * - Split
     - 任务数
     - 每任务 seed
     - Cell 时限
     - Cells
   * - Atomic
     - 18
     - 1--10
     - 1800 秒
     - 180
   * - Composite-Seen
     - 16
     - 1--5
     - 3600 秒
     - 80
   * - Composite-Unseen
     - 16
     - 1--5
     - 3600 秒
     - 80
   * - **总计**
     - **50**
     -
     -
     - **340**

50 个任务分三组:

- **Atomic (18)** —— 单步原语的开合与搬运任务: ``CloseBlenderLid``、
  ``CloseFridge``、``CloseToasterOvenDoor``、``CoffeeSetupMug``、
  ``NavigateKitchen``、``OpenCabinet``、``OpenDrawer``、
  ``OpenStandMixerHead``、``PickPlaceCounterToCabinet``、
  ``PickPlaceCounterToStove``、``PickPlaceDrawerToCounter``、
  ``PickPlaceSinkToCounter``、``PickPlaceToasterToCounter``、
  ``SlideDishwasherRack``、``TurnOffStove``、``TurnOnElectricKettle``、
  ``TurnOnMicrowave``、``TurnOnSinkFaucet``。
- **Composite seen (16)** —— 训练时见过的厨房布局上的多步任务:
  ``ScrubCuttingBoard``、``StackBowlsCabinet``、``WashLettuce``、
  ``RinseSinkBasin``、``PreSoakPan``、``StirVegetables``、
  ``LoadDishwasher``、``SteamInMicrowave``、``SetUpCuttingStation``、
  ``GetToastedBread``、``DeliverStraw``、``KettleBoiling``、
  ``PrepareCoffee``、``StoreLeftoversInBowl``、``SearingMeat``、
  ``PackIdenticalLunches``。
- **Composite unseen (16)** —— 训练时 **没** 见过的布局上的多步任务
  （泛化测试）: ``ArrangeBreadBasket``、``ArrangeTea``、
  ``BreadSelection``、``CategorizeCondiments``、
  ``CuttingToolSelection``、``GarnishPancake``、``GatherTableware``、
  ``HeatKebabSandwich``、``MakeIceLemonade``、``PanTransfer``、
  ``PortionHotDogs``、``RecycleBottlesByType``、
  ``SeparateFreezerRack``、``WaffleReheat``、``WashFruitColander``、
  ``WeighIngredients``。

任选一个传给 ``--task-name`` 即可。RoboCasa 完整目录更大，参见
`RoboCasa <https://robocasa.ai>`_ 上游。

运行一个任务
------------

HTTP RPC endpoint 的主机名为 ``127.0.0.1`` 或 ``localhost`` 时一律直连，无论
worker 由 RPent 启动还是由用户指定。其他主机名与 IP 均遵循标准代理环境。Codex
只在其子进程环境中为本地 MCP 应用相同的两个主机名例外。如果 Hugging Face、
远程 planner 或其他远程服务需要 ``HTTP_PROXY`` 或 ``HTTPS_PROXY``，请保持原有
代理；默认运行不要求 shell 统一配置 ``NO_PROXY``。

如果用户指定的本地服务使用其他主机名或 IP 且应当直连，请将该准确值加入用户
已有的 ``NO_PROXY`` 与 ``no_proxy`` 配置。

RoboCasa 的 CLI 参数由 ``robots/robocasa/__init__`` 注册，可通过
``rpent --robot robocasa --help`` 查看:

.. code-block:: bash

   rpent --robot robocasa \
         --task-name OpenDrawer \
         --split target \
         --seed 1 \
         --vla-model-path /path/to/rldx \
         --planner claude_code \
         --model claude-opus-4-8

RoboCasa 不绑定具体 planner；RPent 支持的任意 planner 都可用于该机器人。
配置方式参见 :doc:`configure_planner`。

正式 Target50 先按上文下载固定资源，再为 manifest 中每个 cell 调用一次普通
命令。Codex 参考 profile 为 ``gpt-5.5``、``xhigh``、``max_turns=100``；
RoboCasa 运行时本身仍与 planner 解耦。场景身份直接使用普通 ``--seed`` 参数，
不要设置 ``RLDX_RESET_SEED``。普通 RoboCasa 使用 ``max_chunks=70``，只有
Target50 将其覆盖为 40。运行前固定 Target50 的 RLDX 执行参数：

.. code-block:: bash

   export RLDX_MAX_CHUNKS=40
   export RLDX_SETTLE_PATIENCE=999
   export RLDX_ACTION_STEPS_PER_CHUNK=8
   unset RLDX_RESET_SEED

第一个 ``OpenDrawer`` Atomic cell 示例：

.. code-block:: bash

   rpent --robot robocasa \
         --task-name OpenDrawer --split target --seed 1 \
         --vla-model-path ./checkpoints/rldx-1-ft-rc365 --cuda-device 0 \
         --vla-backbone-revision 4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7 \
         --planner codex --model gpt-5.5 --reasoning-effort xhigh \
         --max-turns 100 --planner-timeout-s 1800 \
         --memory-profile local \
         --memory-dir ./target50-memory/robocasa \
         --output-dir ./runs/target50/atomic/OpenDrawer_s1

Composite-Seen 与 Composite-Unseen 使用 ``--planner-timeout-s 3600``。执行顺序为
Atomic、Composite-Seen、Composite-Unseen。成功只认最终环境记录中的
``state.success=true``，planner 的 ``finish(status=...)`` 不是评测标签。有效任务
失败与 planner timeout 不重跑；只有没有产生有效环境结果的基础设施失败允许重跑。

每条完成的命令都会原子写入 ``<output-dir>/result.json``，成功值只来自最终环境
``state.success``。文件记录有效协议参数，但不保存 provider 错误原文或凭据。全部
cell 按 ``<results-root>/<manifest-split>/<Task>_s<seed>/result.json`` 落盘后，
用下面命令校验固定分母并输出任务加权指标：

.. code-block:: bash

   python -m robots.robocasa.eval.validate_target50 ./runs/target50

.. note::

   使用 ``--env-endpoint`` / ``--vla-endpoint`` 指向已运行的服务器
   (``[protocol://]host:port``)；不指定时，RPent 会就地启动 env 和 VLA
   子进程，日志分别写到 ``<output_dir>/env_server.log`` 和
   ``<output_dir>/vla_server.log``。

已发布的 Target50 结果
-----------------------

已发布 Codex 复现覆盖全部 340 cells，任务级汇总如下：

.. list-table:: Codex Target50 复现结果
   :header-rows: 1
   :widths: 30 20 20 30

   * - Split
     - 成功 cells
     - 成功率
     - Harness VLA 参考值
   * - Atomic
     - 163/180
     - 90.56%
     - 165/180 (91.67%)
   * - Composite-Seen
     - 49/80
     - 61.25%
     - 45/80 (56.25%)
   * - Composite-Unseen
     - 12/80
     - 15.00%
     - 11/80 (13.75%)
   * - 总体（任务加权）
     - 不适用
     - 57.00%
     - 55.40%

`完整逐任务结果表
<https://github.com/RLinf/RPent/blob/main/robots/robocasa/eval/target50_codex_results.md>`_
给出每个任务的成功次数和准确率。当前发布内容是任务级聚合数据，不包含 seed 级
trace、原始轨迹或失败分类，因此不属于逐 cell 审计产物。

常见错误
--------

资源自检无需 planner 凭据。安装 ``pytest`` 和 ``pytest-timeout`` 后，现有环境
测试只保留四项代表检查：``OpenDrawer``、``NavigateKitchen`` 和
``PickPlaceCounterToCabinet`` 各 seed 1，加一项移动相机检查。它们验证构造/reset、
12D action、操作相机、导航 RGB-D/world map、成功条件和关闭流程，不执行完整
评测矩阵；首次检查可增加 ``-k OpenDrawer``：

.. code-block:: bash

   uv pip install pytest pytest-timeout
   RPENT_RUN_ROBOCASA_INTEGRATION=1 MUJOCO_GL=egl \
      python -m pytest -q tests/integration_tests/robots/robocasa/test_target50_runtime_smoke.py

四类资源下载完成后，再检查模型加载和首次推理：

.. code-block:: bash

   RPENT_RUN_RLDX_INTEGRATION=1 \
   RPENT_RLDX_CHECKPOINT=./checkpoints/rldx-1-ft-rc365 \
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
   MUJOCO_GL=egl python -m pytest -q \
      tests/integration_tests/robots/robocasa/test_rldx_support_smoke.py

这些检查不会启动 planner 或生成 benchmark 成绩，默认 skip 不能当作通过。
离线变量仅作用于该自检命令；普通 HF memory 同步仍需要网络。保持远程 planner
的代理配置不变。
轻量协议测试仍检查全部 50 个任务和固定的 340-cell 分母，全量评测单独执行。

- 下载慢时可使用 ``UV_HTTP_TIMEOUT=600``，并将缓存和临时目录放在空间足够的
  文件系统上。重新执行固定 revision 的 HF 下载命令；不能只看 shard 文件大小
  判断完整性，不要禁用 TLS 校验。
- 只读 assets 报错需要安装修正后的 RoboCasa 依赖，不要开放资源目录写权限；
  转换后的 XML 应写入临时目录。
- RLDX 离线缓存缺失时检查上述支持文件和缓存变量。
  ``NO_ALBUMENTATIONS_UPDATE=1`` 只关闭导入时的更新检查，不改变图像处理；
  保持现有 image geometry fallback 参数。
- CUDA 兼容性应通过 GPU 运算和 EGL render 验证，不能只看驱动显示版本；推荐
  Torch/CUDA 组合见安装部分。
- 共享只读环境应将 ``NUMBA_CACHE_DIR`` 设置到当前用户可写目录，不要修改包的
  代码权限。

- 导航 RGB-D 或 world map 渲染报告缺少 ``mobilebase0_navview`` 时，应重新
  安装 ``.[robocasa]`` 以刷新 ``RLinf/robosuite`` 的 ``rpent`` 分支；不要手工
  修改已安装的 XML。
- ``read_text_file`` 报告缺少当前任务结果时，请检查
  ``memory/robocasa/results/`` 目录或所选本地目录。RPent 不会读取其他任务的
  memory 作为替代。
  Markdown 为可选文件；Atomic 任务没有发布 ``<Task>.md``。
- 环境与 VLA 启动错误会分别记录在 ``<output_dir>/env_server.log`` 和
  ``<output_dir>/vla_server.log``；运行级错误也可检查 ``<output_dir>/run.log``。
- 只有准确的 ``127.0.0.1`` 与 ``localhost`` 主机名会自动绕过 HTTP 代理。其他
  主机名与 IP 均遵循标准代理环境；只有该服务应当直连时，才需要把准确主机名
  加入 ``NO_PROXY`` 与 ``no_proxy`` 配置。

Toolkit 与 LIBERO 的差异
------------------------

RoboCasa toolkit 提供的工具 *形式* 与 LIBERO 相同（一次原语调用、
一次状态查看、一次 ``finish``），但有两处 RoboCasa 特有的差异:

- **Env 侧的辅助方法。** 抓取检测与动作组装需要运行中的仿真 env, 所以
  它们是 env_server 的 RPC。Agent 侧的 skill 因此同时持有 **两个**
  client: env client 做 render/step, model client 做 RLDX-1 推理。
  理由参见 :doc:`../development/add_robot`。
- **观测形状。** RLDX-1 看到的是 3 路相机的视频张量
  ``(1, T, H, W, 3)``, 按历史 ``T`` 堆叠, 加上 ``state.*`` 与
  ``annotation.*`` 字段。session id 不在观测里——它由 RPC 框架自动
  管理: ``RpcClient`` 生成 ``rpc_`` + uuid hex 的私有 session id,
  ``wait_for_ready`` 在连接时注册到服务端; 服务端跟踪每个 session
  的空闲时间, 后台 sweep 线程定期回收超时 (默认 3600 秒) 的 session,
  进程退出时客户端通过 atexit 发送 ``session.close``。业务代码
  (``rldx_skill`` / ``vla_client``) 从不直接看到 session id, 服务端
  把它注入到 ``predict`` / ``reset_session`` 中, 按客户端隔离
  RLDX memory/RTC 策略状态。
