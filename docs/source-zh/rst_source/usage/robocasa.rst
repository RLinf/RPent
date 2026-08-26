RoboCasa
========

`RoboCasa <https://robocasa.ai>`_ 是面向厨房场景的长时序操作仿真环境。
在 RPent 中由 **RLDX-1** VLA 策略驱动，默认通过 HTTP RPC 提供服务
（与 LIBERO 一致），也支持 pickle-framed socket 传输。详见
``robots/robocasa/vla_server.py`` 与 ``robots/robocasa/__init__.py``
中的传输选择逻辑。

.. note::

   当前代码尚未完全对齐 `harnessvla.github.io <https://harnessvla.github.io>`_
   展示的效果，完整复现将在后续放出。

安装
----

RoboCasa365 已包含在 ``.[full]`` 中。若要单独安装 —— RLDX-1 要求
Python ``3.10``\ ：

.. code-block:: bash

   uv venv --python 3.10
   uv pip install -e ".[robocasa]"

为避免 PyPI 上 torch 的 CUDA build 与本地驱动不匹配，建议通过
``--index`` 指定 PyTorch CUDA 索引，例如
``uv pip install -e ".[robocasa]" --index https://download.pytorch.org/whl/cu126``；
CUDA 13-only 的机器换成 ``cu130``。

国内网络可使用 PyPI 镜像加速：\

.. code-block:: bash

   uv pip install -e ".[robocasa]" \
      --default-index https://mirrors.aliyun.com/pypi/simple \
      --index https://pypi.tuna.tsinghua.edu.cn/simple \
      --index https://mirrors.aliyun.com/pytorch-wheels/cu126

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

**RLDX-1 checkpoint**

下面运行命令的 ``--vla-model-path`` 期望一个本地 ``RLDX-1-FT-RC365``
checkpoint 路径（RoboCasa365 微调版）。从 HuggingFace 下载:

.. code-block:: bash

   hf download RLWRLD/RLDX-1-FT-RC365 --local-dir ./checkpoints/rldx-1-ft-rc365

下载慢的话用 HF 镜像:

.. code-block:: bash

   HF_ENDPOINT=https://hf-mirror.com hf download RLWRLD/RLDX-1-FT-RC365 --local-dir ./checkpoints/rldx-1-ft-rc365

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

.. note::

   使用 ``--env-endpoint`` / ``--vla-endpoint`` 指向已运行的服务器
   (``[protocol://]host:port``)；不指定时，RPent 会就地启动 env 和 VLA
   子进程，日志分别写到 ``<output_dir>/env_server.log`` 和
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
