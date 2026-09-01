RoboCasa
========

`RoboCasa <https://robocasa.ai>`_ 是面向厨房场景的长时序操作仿真环境。
在 RPent 中由 **RLDX-1** VLA 策略驱动，默认通过 HTTP RPC 提供服务
（与 LIBERO 一致），也支持 pickle-framed socket 传输。详见
``robots/robocasa/vla_server.py`` 与 ``robots/robocasa/robot_spec.py``
中的传输选择逻辑。

.. note::

   本页只说明普通的单任务 ``rpent --robot robocasa`` 运行方式，不定义
   Atomic/Seen/Unseen 全量 benchmark 的正式复现协议。

安装
----

RLDX-1 要求 Python ``3.10``。请创建独立环境，并通过 ``.[robocasa]``
安装完整的 RoboCasa365 运行栈：

.. code-block:: bash

   uv venv --python 3.10
   uv pip install -e ".[robocasa]" \
      "torch==2.7.0" "torchvision==0.22.0" \
      --torch-backend=cu126

该命令固定 RLDX-1 使用的 Torch 版本组合，并让 uv 只为 Torch 选择官方 CUDA
wheel，避免把 PyTorch wheel 源作为通用 ``--index`` 后，在默认 first-index
策略下误选其中的旧版无关依赖。上面的 ``cu126`` 是已验证的 CUDA 安装方式；
仅当宿主机确有需要时才切换到其他受支持的 Torch backend。

国内网络可使用 PyPI 镜像加速：\

.. code-block:: bash

   uv pip install -e ".[robocasa]" \
      "torch==2.7.0" "torchvision==0.22.0" \
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

一条命令即可生成 ``macros_private.py`` 并下载厨房 assets（约 10 GB）。建议
放在 ``site-packages`` 之外，重装不会丢：

.. code-block:: bash

   robocasa-download-assets --assets-path ~/.robocasa/assets -y

命令结束时会打印需要导出的环境变量，把它们加到启动 ``rpent`` 的 shell 里：

.. code-block:: bash

   export ROBOCASA_MACROS_PATH=~/.robocasa/macros_private.py
   export ROBOCASA_ASSETS_PATH=~/.robocasa/assets

加 ``--skip-existing`` 重跑会跳过已下载的目录。

**移动相机**

``robocasa`` extra 会安装 ``RLinf/robosuite`` 的 ``rpent`` 分支，该分支
包含 Omron 底盘固定的 ``navview`` 相机，其组合后的 MuJoCo 相机名为
``mobilebase0_navview``。RPent 不再执行单独的 reset-time 相机预检；如果相机
缺失，首次请求导航 RGB-D 或 world map 渲染时会自然报错。此时重新安装
``.[robocasa]`` 以刷新该分支即可，无需手工修改 ``site-packages`` 中的 XML。
本 PR 验证时该分支解析为 commit
``97cfbde4b68d8ec43dad20cf4747297866a6ca2e``；发布实验结果时应记录实际解析
到的 commit。

**RLDX-1 checkpoint**

下面运行命令的 ``--vla-model-path`` 期望一个本地 ``RLDX-1-FT-RC365``
checkpoint 路径（RoboCasa365 微调版）。从 HuggingFace 下载:

.. code-block:: bash

   hf download RLWRLD/RLDX-1-FT-RC365 --local-dir ./checkpoints/rldx-1-ft-rc365

下载慢的话用 HF 镜像:

.. code-block:: bash

   HF_ENDPOINT=https://hf-mirror.com hf download RLWRLD/RLDX-1-FT-RC365 --local-dir ./checkpoints/rldx-1-ft-rc365

**任务 Memory**

默认评测会复用 RPent 的公共资源同步机制，从 Hugging Face 数据集
``RLinf/RPent-memory`` 下载 ``robocasa/**`` 到 ``resources/robocasa``。
当前任务只能使用下面这些同任务 memory：

.. code-block:: text

   resources/robocasa/results/<Task>_s0.json
   resources/robocasa/results/recipe_<Task>_s0.jsonl
   resources/robocasa/results/<Task>.md  # 可选

JSON/JSONL pair 保存经过审核的 seed-0 证据。可选的 Markdown 文件保存同任务
探索 memory，可能汇总多次尝试；当前 16 个 Composite-Seen 和 9 个
Composite-Unseen 任务包含此文件。Prompt 要求 planner 在开始动作前主动通过
``read_text_file`` 读取当前任务所有存在的文件；RPent 不会把 Markdown 内容
强制注入 prompt。

RoboCasa 不要求 planner 使用 global memory，也不会退回读取其他任务的 memory。
运行时不预检 corpus 的完整性；当前任务文件缺失时，``read_text_file`` 会报告
该文件不存在，planner 使用其余可用的同任务文件和实时观测继续。公开的默认
corpus 会在发布阶段单独校验。如需使用经过审核的本地文件，请选择 local
profile 并传入 results corpus 所在目录：

.. code-block:: bash

   rpent --robot robocasa --task-name OpenDrawer --seed 1 \
         --vla-model-path /path/to/rldx --planner claude_code \
         --memory-profile local --memory-dir /path/to/robocasa-results

可用任务列表
------------

RPent 用的 50 个任务分三组:

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

RoboCasa 的 CLI 参数由 ``robots/robocasa/__init__`` 注册，可通过
``rpent --robot robocasa --help`` 查看:

.. code-block:: bash

   rpent --robot robocasa \
         --task-name OpenDrawer \
         --split target \
         --seed 0 \
         --vla-model-path /path/to/rldx \
         --planner claude_code \
         --model claude-opus-4-8

RoboCasa 不绑定具体 planner；RPent 支持的任意 planner 都可用于该机器人。
配置方式参见 :doc:`configure_planner`。

.. note::

   使用 ``--env-endpoint`` / ``--vla-endpoint`` 指向已运行的服务器
   (``[protocol://]host:port``)；不指定时，RPent 会就地启动 env 和 VLA
   子进程，日志分别写到 ``<output_dir>/env_server.log`` 和
   ``<output_dir>/vla_server.log``。

常见错误
--------

- 导航 RGB-D 或 world map 渲染报告缺少 ``mobilebase0_navview`` 时，应重新
  安装 ``.[robocasa]`` 以刷新 ``RLinf/robosuite`` 的 ``rpent`` 分支；不要手工
  修改已安装的 XML。
- ``read_text_file`` 报告缺少当前任务结果时，请检查 Hugging Face 的
  ``robocasa/results`` corpus 或所选本地目录。RPent 不预检 corpus，也不会
  退回读取其他任务。
- 环境与 VLA 启动错误会分别记录在 ``<output_dir>/env_server.log`` 和
  ``<output_dir>/vla_server.log``。

Toolkit 与 LIBERO 的差异
------------------------

RoboCasa toolkit 提供的工具 *形式* 与 LIBERO 相同（一次原语调用、
一次状态查看、一次 ``finish``），但有两处 RoboCasa 特有的差异:

- **Env 侧的辅助方法。** 抓取检测与动作组装需要活着的仿真 env, 所以
  它们是 env_server 的 RPC。Agent 侧的 skill 因此同时持有 **两个**
  client: env client 做 render/step, model client 做 RLDX-1 推理。
  理由参见 :doc:`../development/add_robot`。
- **观测形状。** RLDX-1 看到的是 3 路相机 video 张量
  ``(1, T, H, W, 3)``, 按历史 ``T`` 堆叠，加上 ``state.*``、annotation、
  以及一个 session id (用于 ``reset_session`` / ``predict``)。
