RoboCasa
========

`RoboCasa <https://robocasa.ai>`_ 是厨房尺度、长时序的操作 environment。
在 RPent 中由 **RLDX-1** VLA 策略驱动，默认通过 HTTP RPC 提供服务
（与 LIBERO 一致），也支持 pickle-framed socket 传输。详见
``robots/robocasa/vla_server.py`` 与 ``robots/robocasa/__init__.py``
中的传输选择逻辑。

安装
----

RoboCasa365 不在 ``.[full]`` 里。``.[robocasa]`` extra 装齐整个 stack
—— MuJoCo 3.3.1、ARISE-Initiative robosuite fork、pinned lerobot commit、
protobuf，以及 ``robocasa`` 包本身（从 ``github.com/qurakchin/robocasa``
fork 的 ``v1.0.1_rlinf`` 分支装的 wheel）。fork 改造过，让
``macros_private`` 和 ``assets`` 都从 env var 加载，所以非 editable 的
wheel 装也能用，不需要本地 clone。

**前置依赖 —— PyTorch、torchvision、flash-attn**

RLDX-1 依赖特定版本的 PyTorch、torchvision 和 flash-attn。请在主
``.[robocasa]`` 安装 **之前** 先装好它们，并选择一个有预编译
flash-attn wheel 的 CUDA 版本:

.. code-block:: bash

   # 装 PyTorch + torchvision（选 cu124 可直接用预编译 flash-attn）
   uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu124

   # 装 flash-attn（cu124 的预编译 wheel，无需源码编译）
   uv pip install flash-attn==2.7.4.post1

   # 再装 RPent RoboCasa stack
   uv pip install -e ".[robocasa]"

**注意**: 如果用其他 CUDA 版本（如 cu126），flash-attn 需要从源码编译，
必须有 CUDA 编译器。预编译 wheel 覆盖 cu118、cu121、cu124 —— 推荐
``cu124``，开箱即用。

**安装后处理**

装完 ``.[robocasa]`` 后，RoboCasa 还需要 ``macros_private.py`` 和
厨房 assets 才能运行 ``rpent``:

1. 生成 ``macros_private.py`` 并导出路径:

   .. code-block:: bash

      # 默认写到 <repo_root>/.robocasa/macros_private.py
      export ROBOCASA_MACROS_PATH=$PWD/.robocasa/macros_private.py
      python -m robocasa.scripts.setup_macros

   fork 的 ``macros.py`` 在 import 时读 ``$ROBOCASA_MACROS_PATH``，所以
   任何启动 ``rpent`` 的 shell 都要设这个 env var —— 加到你的
   ``.bashrc`` / ``.zshrc`` 里。

2. 下载厨房 assets（10+ GB），可选地移出 ``site-packages``:

   .. code-block:: bash

      # 下载到 wheel 自带的 robocasa/models/assets/
      python -m robocasa.scripts.download_kitchen_assets --type all

      # 可选: 移到外部目录，避免 wheel 重装时丢失，也可跨 venv 共享
      export ROBOCASA_ASSETS_PATH=$PWD/.robocasa/assets
      WHEEL_ASSETS=$(python -c "import robocasa; print(robocasa.__path__[0])")/models/assets
      mkdir -p "$ROBOCASA_ASSETS_PATH"
      mv "$WHEEL_ASSETS"/* "$ROBOCASA_ASSETS_PATH"/

   不设 ``ROBOCASA_ASSETS_PATH`` 时，robocasa 会 fallback 到 wheel 自带
   的 ``models/assets/`` —— 光下载就够跑。只有移走了 assets 才需要
   导出这个 env var。

3. （可选）验证 import 作为 sanity check:

   .. code-block:: bash

      python -c "import robosuite, robocasa; print(robosuite.__version__, robocasa.__path__[0])"

安装时的默认值见 :doc:`../installation`。

**RLDX-1 checkpoint**

下面运行命令的 ``--vla-model-path`` 期望一个本地 ``RLDX-1-FT-RC365``
checkpoint 路径（RoboCasa365 微调版）。从 HuggingFace 下载:

.. code-block:: bash

   huggingface-cli download RLWRLD/RLDX-1-FT-RC365 --local-dir ./checkpoints/rldx-1-ft-rc365

下载慢的话用 HF 镜像:

.. code-block:: bash

   HF_ENDPOINT=https://hf-mirror.com huggingface-cli download RLWRLD/RLDX-1-FT-RC365 --local-dir ./checkpoints/rldx-1-ft-rc365

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

任选一个传给 ``--robocasa-env`` 即可。RoboCasa 完整目录更大，参见
`RoboCasa <https://robocasa.ai>`_ 上游。

运行一个任务
------------

RoboCasa 的 CLI 参数由 ``robots/robocasa/__init__`` 注册，可通过
``rpent --env robocasa --help`` 查看:

.. code-block:: bash

   rpent --env robocasa \
         --robocasa-env OpenDrawer \
         --robocasa-split target \
         --robocasa-seed 0 \
         --vla-model-path /path/to/rldx \
         --planner claude_code \
         --model claude-opus-4-8

使用 ``--env-endpoint`` / ``--vla-endpoint`` 指向已运行的服务器
(``[protocol://]host:port``)；不指定时，RPent 会就地启动 env 和 VLA
子进程，日志分别写到 ``<output_dir>/env_server.log`` 和
``<output_dir>/vla_server.log``。

Toolkit 与 LIBERO 的差异
------------------------

RoboCasa toolkit 的工具 *形状* 和 LIBERO 相同 (一次原语调用、
一次状态查看、一次 ``finish``), 但有两处是 RoboCasa 特有的:

- **Env 侧的辅助方法。** 抓取检测与动作组装需要活着的仿真 env, 所以
  它们是 env_server 的 RPC。Agent 侧的 skill 因此同时持有 **两个**
  client: env client 做 render/step, model client 做 RLDX-1 推理。
  理由参见 :doc:`../development/add_robot`。
- **观测形状。** RLDX-1 看到的是 3 路相机 video 张量
  ``(1, T, H, W, 3)``, 按历史 ``T`` 堆叠，加上 ``state.*``、annotation、
  以及一个 session id (用于 ``reset_session`` / ``predict``)。
