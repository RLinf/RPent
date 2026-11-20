BEHAVIOR
========

使用 RPent 的视觉语言规划器，在 BEHAVIOR-1K 仿真环境中完成家庭操作任务。规划器根据相机观测调用共享 Pi0.5 和机器人动作工具，动作由仿真中的 R1Pro 双臂移动机器人执行。独立的 RPent Env 进程通过 RLinf 的 BEHAVIOR 3.9.3 适配器推进仿真并读取官方任务指标。

本文从 BEHAVIOR-1K 挑战赛的 100 个任务中选取“制作微波爆米花”（任务 ID：40）作为运行示例，介绍环境准备、任务启动，以及观测和评分结果的查看方式。

.. _behavior-overview:

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      RPent 共享 Pi0.5 的 BEHAVIOR 配置；可选夹爪、手臂和底盘工具。

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``

   .. grid-item-card:: 任务

      100 个挑战赛任务 ID；已公布 3 个任务的对比结果。

   .. grid-item-card:: 硬件

      Linux、NVIDIA GPU，以及兼容的 OmniGibson 图形依赖和模型 CUDA 环境。

每个 Env 进程拥有一个仿真实例和一个回合。规划器连接其 RPC 地址，共享 Pi0.5 在独立进程中推理。

任务
~~~~

任务 ID、官方名称和完整指令见 ``robots/behavior/tasks.py``。选择任务时，这三项须对应同一任务。

.. list-table::
   :header-rows: 1

   * - 范围
     - 选择方式
   * - 100 个挑战赛任务
     - ID 为 ``0..99``。规划器只指定 ``--task-id`` 时使用官方任务指令；Env 也使用相同默认值；显式覆盖时，两端指令必须一致。
   * - 本文的公开测试示例
     - ID ``40``，名称 ``make_microwave_popcorn``；通过 ``--instance-split public_test --instance-id 301`` 选择真实实例 ID，而不是列表下标。首次运行示例最多执行 320 步。

.. _behavior-observation-action:

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - 头部、左腕和右腕三路相机图像，以及机器人关节、夹爪等实测状态，供规划器和动作模型判断当前情况。
   * - 动作
     - 规划器调用 VLA 或机器人控制工具执行动作，再根据新的观测决定下一步。两份模板每次 VLA 调用最多执行 32 步，可通过参数缩短。
   * - 奖励与成功判定
     - Env 从 RLinf 仿真适配器读取原生终止、截断标志和官方 QScore 指标；终止标志本身不等于任务成功。评分结果的文件位置见“查看结果”。
   * - 任务指令
     - 默认每次调用 VLA 都使用所选任务的完整指令。规划器可以选择不同工具，但不把临时生成的子任务指令替换为 VLA 的输入指令。

安装与资源准备
--------------

使用 Linux、Python 3.11，以及满足 `Isaac Sim 5.1 要求 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_ 的 NVIDIA GPU。按工具所需的能力选择 Env 接口。

* 读取观测和执行原生动作可以使用
  `RLinf 的公开 BehaviorEnv 版本 <https://github.com/Forging-ahead/RLinf/tree/b4a7bfabaa54e9acde9aad86369eeb9ad0d6196a>`_。
  环境返回官方 61 维 R1Pro 状态、三路图像、奖励和回合进度。按该版本的 BEHAVIOR 安装说明准备仿真环境，
  再用 Env 的 Python 安装 RPent 的 ``behavior`` extra。启动单实例服务的示例如下：

  .. code-block:: bash

     python -m robots.behavior.env_server --env-api behavior-env \
       --env-config /path/to/radio.yaml --task-id 0 --instance-id 301 \
       --camera-size 128 --output-dir /path/to/run --port 25000

  Env RPC 直接返回 RLinf 的 ``main_images``、按左腕和右腕排列的 ``wrist_images``，
  以及形状为 1 × 61 的 ``states``。Pi0.5 接收这三项和配置的任务指令，工具可读取环境提供的米制深度。

  从 RLinf 的 ``examples/embodiment/config/env/behavior_r1pro.yaml`` 复制出 ``radio.yaml``。
  使用公开测试实例时，将 ``omni_config.task.activity_instance_dir`` 设为该任务公开实例的缓存目录，
  并选择匹配的场景配置。服务使用 RLinf 的 2026 状态格式，通过 Ray 运行一个环境。
  每个确认执行的动作都有对应的返回帧。OmniGibson 在重置和执行首个动作时都可能返回回合索引 0。
* 下文的 CuRobo、相机标定、旧版 256 维模型输入和官方 QScore 示例使用进程内的 ``BehaviorSimulator``。
  `这个 RLinf 版本 <https://github.com/Forging-ahead/RLinf/tree/1edcc27dc0dde23d520da1112545146b4902e846>`_
  提供该接口。可用 ``--env-api simulator`` 显式选择。省略参数时，若安装的 RLinf 包导出
  ``BehaviorSimulator``，RPent 就沿用它；其他情况选择 ``BehaviorEnv``。

使用三个独立的 Python 环境。Planner 和 SAM3 可以共用一个解释器、分别运行进程；Env 和 Pi0.5 各自使用独立解释器，避免仿真、动作模型与分割模型依赖相互替换。

.. list-table::
   :header-rows: 1

   * - 解释器
     - 用途与安装配置
   * - ``ENV_PYTHON``
     - OmniGibson 3.9.3 / Isaac Sim 5.1.0；RLinf 的 ``behavior`` 纯环境配置使用 PyTorch 2.7.0 和 CUDA 12.8 wheel。
   * - ``VLA_PYTHON``
     - 共享 Pi0.5；RLinf 的纯模型 ``--model openpi`` 配置使用 PyTorch 2.7.1 和 CUDA 12.8 wheel，不安装仿真 SDK。
   * - ``PLANNER_PYTHON`` / ``SAM3_PYTHON``
     - RPent 规划器与可选 SAM3 服务；独立于 Env 和 OpenPI 环境安装。

将下方占位路径替换为本地路径，使用新的环境目录，并把新的 ``RPENT_RUN`` 放在源码仓库之外。每个终端都需加载这些变量，也可以保存为本地 shell 文件后在各终端中 source。

.. code-block:: bash

   export RPENT_REPO=/path/to/RPent
   export RLINF_REPO=/path/to/RLinf
   export ENV_VENV=/path/to/envs/behavior-393
   export VLA_VENV=/path/to/envs/behavior-pi05
   export PLANNER_VENV=/path/to/envs/behavior-planner
   export ENV_PYTHON="$ENV_VENV/bin/python"
   export VLA_PYTHON="$VLA_VENV/bin/python"
   export PLANNER_PYTHON="$PLANNER_VENV/bin/python"
   export SAM3_PYTHON="$PLANNER_PYTHON"
   export RPENT_RUN=/path/to/behavior-runs/run-name
   export ENV_GPU=0
   export VLA_GPU=1
   export SAM3_GPU=2
   export OMNIGIBSON_DATA_PATH=/path/to/2026/datasets
   export OMNIGIBSON_DATASET_PATH="$OMNIGIBSON_DATA_PATH/behavior-1k-assets"
   export OMNIGIBSON_ASSET_PATH="$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets"
   export OMNIGIBSON_KEY_PATH="$OMNIGIBSON_DATA_PATH/omnigibson.key"
   export VLA_CHECKPOINT=/path/to/checkpoints/PT50-CS32
   export OPENPI_DATA_HOME=/path/to/cache/openpi
   export SAM3_CHECKPOINT_PATH=/path/to/checkpoints/sam3/sam3.pt
   export PLANNER_MODEL=provider:vision-model-name
   export TASK_ID=40

按 :doc:`../get_started/quickstart` 安装 uv，然后从匹配的 RLinf 仓库依次安装仿真和模型环境。使用原有的 BEHAVIOR 环境配置安装仿真器，使用纯模型 OpenPI 配置安装 Pi0.5。RPent 用两套解释器分别启动两个服务。

.. code-block:: bash

   cd "$RLINF_REPO"
   bash requirements/install.sh embodied --env behavior \
     --python 3.11 --venv "$ENV_VENV" --install-rlinf --no-flash-attn
   bash requirements/install.sh embodied --model openpi \
     --python 3.11 --venv "$VLA_VENV" --install-rlinf --no-flash-attn

Env 安装器使用官方 BEHAVIOR ``v3.9.3-post2`` 源码版本 ``a8247a8cc1633fe1ca0cc66aa07243d46c64f155``，并将其 LeRobot 依赖固定到 ``436812bd8ee39b768c645c248c91f1330834e687``，保持 Python 3.11 兼容。依赖固定操作在单独的 SDK 安装副本中完成，保留原 SDK 源码不变；安装后应保留该副本。只有系统依赖和构建工具已经齐全时才加 ``--no-root``。模型安装会调用 RLinf 现有 OpenPI 资源脚本；下方下载命令将 tokenizer 明确放到 ``OPENPI_DATA_HOME`` 指定的缓存目录，运行时读取其中的 ``big_vision/paligemma_tokenizer.model``。

在三个环境中安装 RPent。``behavior`` extra 提供 RPent 接入所需依赖；``behavior-vla`` 使用单独安装的 RLinf/OpenPI 运行环境。这两个 extra 均不下载仿真资产或模型权重。

.. code-block:: bash

   uv venv --python 3.11 "$PLANNER_VENV"
   cd "$RPENT_REPO"
   uv pip install --python "$PLANNER_PYTHON" -e '.[behavior]'
   uv pip install --python "$ENV_PYTHON" \
     --override "$RLINF_REPO/requirements/embodied/envs/behavior-overrides.txt" \
     -e '.[behavior]'
   uv pip install --python "$VLA_PYTHON" -e '.[behavior-vla]'
   uv pip check --python "$PLANNER_PYTHON"
   "$ENV_PYTHON" -m pip check
   uv pip check --python "$VLA_PYTHON"

Env 使用 RLinf 的 BEHAVIOR 依赖覆盖文件。复用官方 SDK 时，用 Env 解释器运行 pip 检查可以覆盖从 SDK 继承的依赖。已验证的环境报告了 packaging、websockets、wrapt 和 Pillow 四项元数据冲突。具体约束见配套 RLinf 仓库的 BEHAVIOR 安装文档，安装后还需验证服务启动。

加载场景前，先核对适配器导入位置：

.. code-block:: bash

   "$ENV_PYTHON" -c 'import inspect; from rlinf.envs.sim.behavior import BehaviorSimulator; print(inspect.getfile(BehaviorSimulator))'

输出应位于 ``RLINF_REPO`` 内。若导入了其他仓库或安装依赖冲突，先解决再启动服务。这个检查只能确认导入位置，不能证明仿真资产可用或任务能够成功执行。

按照 `BEHAVIOR-1K 官方说明 <https://github.com/StanfordVL/BEHAVIOR-1K>`_ 下载 2026 挑战赛资产和所选任务的公开测试实例。``OMNIGIBSON_DATA_PATH`` 下应包含 ``omnigibson-robot-assets``、``behavior-1k-assets`` 和 ``2026-challenge-task-instances``，数据集密钥位于 ``OMNIGIBSON_KEY_PATH``。安装 Python 包不会提供这些数据。运动工具使用已加载 R1Pro 的关节限位、USD 模型及这些资产中的 CuRobo 配置。

使用模型环境中的 Hugging Face CLI 下载 `公开 PT50-CS32 权重 <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_：

.. code-block:: bash

   "$VLA_VENV/bin/hf" download RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 \
     --local-dir "$VLA_CHECKPOINT"
   test -s "$VLA_CHECKPOINT/model.safetensors"
   test -s "$VLA_CHECKPOINT/config.json"
   test -s "$VLA_CHECKPOINT/assets/behavior-1k/2025-challenge-demos/norm_stats.json"
   "$VLA_VENV/bin/hf" download RLinf/openpi_tokenizer --local-dir "$OPENPI_DATA_HOME"
   test -s "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"

权重与归一化统计必须配套。按照 :doc:`../guides/configure_planner` 设置视觉语言规划器的模型、服务 URL 和凭据；Planner 模型与 VLA 权重分别选择。启动 GPU 服务前，先检查规划器认证：

.. code-block:: bash

   "$PLANNER_VENV/bin/rpent-check-llm" --planner api --model "$PLANNER_MODEL" --json

这个检查不调用图像或工具。若希望不带历史任务经验运行，创建一个空的本地记忆目录即可，不需要编造 ``MEMORY.md``；下方规划器命令会选择这个只读经验库。

.. code-block:: bash

   mkdir -p "$RPENT_RUN/memory-empty"
   cd "$RPENT_REPO"

示例将 GPU 0 分配给 Env、GPU 1 分配给 Pi0.5、GPU 2 分配给可选 SAM3，请按实际分配的设备调整编号。Env 使用 ``OMNIGIBSON_GPU_ID``，启动时移除 ``CUDA_VISIBLE_DEVICES``，避免 CUDA 屏蔽设备后与 Vulkan 使用不同编号。YAML 中保留 ``omni_config.env.device: null``；模型服务使用各自的 ``--cuda-device`` 参数。

运行一个任务
------------

先选择规划器可以调用的工具。两份模板都由视觉语言规划器选择下一步动作，并都保留 VLA；区别在于是否提供额外的感知和本体控制工具。``vla_only.yaml`` 因此不是下文实验中的纯 VLA 基线。

.. list-table::
   :header-rows: 1

   * - 配置
     - 规划器可用的机器人工具
     - 额外依赖
   * - ``vla_only.yaml``
     - 查看观测、调用 VLA、结束会话。
     - 无 SAM3 或 CuRobo 依赖，适合首次运行。
   * - ``agent.yaml``
     - 保留观测、VLA 和结束工具，增加详细机器人状态读取、分割、三维定位，以及夹爪、手臂和底盘控制；共 12 个机器人工具。
     - SAM3、CuRobo；三维定位需要 RGB-D 观测。

下面使用 ``vla_only.yaml``，三个进程各占一个终端。320 步上限用于有界接入检查，不代表完整 benchmark 回合；对比任务效果时需使用一致的任务预算和评测协议。

**1. 启动共享 Pi0.5 服务。** 加载权重和配套归一化统计，等待服务在 ``28011`` 端口就绪。以下统计文件路径属于公开 PT50-CS32 权重；更换权重时需同时选择其配套统计。

.. code-block:: bash

   "$VLA_PYTHON" -m rpent.robots.components.pi05_vla_server \
     --embodiment behavior --model-path "$VLA_CHECKPOINT" \
     --cuda-device "$VLA_GPU" \
     --norm-stats-path "$VLA_CHECKPOINT/assets/behavior-1k/2025-challenge-demos/norm_stats.json" \
     --host 127.0.0.1 --port 28011 --transport socket

**2. 启动仿真 Env 服务。** 服务组合 RLinf YAML 并应用下方参数覆盖，创建一个仿真实例、完成首次 reset 后，在 ``28010`` 端口接收 RPC 动作请求。不再单独启动评测器或 policy WebSocket；等待场景加载完成后再运行规划器。

.. code-block:: bash

   env -u CUDA_VISIBLE_DEVICES OMNIGIBSON_GPU_ID="$ENV_GPU" \
     "$ENV_PYTHON" -m robots.behavior.env_server \
     --env-config "$RLINF_REPO/examples/embodiment/config/env/behavior_r1pro.yaml" \
     --task-id "$TASK_ID" --instance-split public_test --instance-id 301 \
     --max-steps 320 --seed 0 \
     --action-frequency 30 --rendering-frequency 30 --physics-frequency 120 \
     --camera-size 128 --env-wrapper rgbd \
     --host 127.0.0.1 --port 28010 --transport socket \
     --output-dir "$RPENT_RUN/env"

这个有界示例显式设置 30 Hz 控制和渲染、120 Hz 物理频率。不传这些参数时，保留组合 YAML 中的频率；更换权重或评测协议时，应同步核对频率。Env 和 Planner 均通过 ``--task-id`` 确定官方任务名称和指令。

**3. Env 就绪后启动规划器。** 任务 ID 须与 Env 一致。若覆盖官方指令，两条命令需传入相同的 ``--task-language``。动作模型地址指向第 1 步的服务。

.. code-block:: bash

   "$PLANNER_PYTHON" -m rpent.cli.main --robot behavior --planner api \
     --model "$PLANNER_MODEL" \
     --behavior-config robots/behavior/configs/vla_only.yaml \
     --task-id "$TASK_ID" \
     --memory-profile local --memory-dir "$RPENT_RUN/memory-empty" \
     --env-endpoint socket://127.0.0.1:28010 \
     --am-backend pi05 --am-endpoint socket://127.0.0.1:28011 \
     --output-dir "$RPENT_RUN/agent"

规划器退出只关闭自己的客户端连接，借用的服务继续运行。规划器结束后，在 Env 终端按 Ctrl+C 收集最终结果并停止服务，再停止 Pi0.5 终端，以及本轮启动的 SAM3 终端；只停止本轮拥有的服务。官方 SDK 的关闭操作可能立即终止专用 Env 进程，因此 Env 会在调用它之前写出结果。确认 ``env-result.json`` 的 ``server_status`` 为 ``stopped``、查看其中指标，并检查本轮进程均已退出和释放 GPU。下一回合使用新的 Env 进程和输出目录。

连接客户端不会重置 Env。显式调用 ``BehaviorEnvClient.reset()`` 只重置仿真环境及该客户端缓存的观测与反馈。每个 Planner/Toolkit 会话对应一个回合；Env 重置后需要新建会话，不能跨回合复用保留着动作和终止状态缓存的旧 Toolkit。

查看结果
--------

一次运行会生成两类结果：规划器保存工具调用与观测，Env 保存生效配置，并在关闭时写出当前官方指标。运行中的 ``env-result.json`` 记录服务是否就绪；正常停止 Env 后查看最终指标。深度文件仅在观测提供深度时生成。

.. code-block:: text

   $RPENT_RUN/
   ├── agent/
   │   ├── states.json
   │   ├── head_rgb.png/00.png
   │   ├── left_wrist_rgb.png/00.png
   │   ├── right_wrist_rgb.png/00.png
   │   ├── proprio.npy/00.npy
   │   └── head_depth.npy/00.npy
   └── env/
       ├── env_config.yaml
       └── env-result.json

``00`` 为初始观测，随后为 ``01``、``02``……；路径中的 ``head_rgb.png`` 等名称是目录。左、右腕深度按相同方式保存。该编号是保存记录的编号，一次工具调用可执行多个仿真动作。

.. list-table:: 查看哪些字段
   :header-rows: 1

   * - 文件与字段
     - 含义
   * - ``env-result.json``：``server_status``、``session``
     - 服务状态和回合的原生终止、截断信息。正常停止后状态为 ``stopped``，失败时查看 ``error`` 或 ``cleanup_error``。SDK 关闭可能在记录错误后仍以 0 退出，需结合该文件和进程退出情况判断。
   * - ``env-result.json``：``metrics[0]``
     - ``official_metrics`` 保存官方指标，另有缓存的 ``q_score``、``instance_id``、``episode_length``。完成首个动作前 QScore 为 ``null``，不是 0；服务停止也不等于任务成功。
   * - ``env_config.yaml``
     - 应用命令行覆盖后的任务实例、随机种子、频率、步数上限和仿真配置。
   * - ``states.json``：``steps[i].command``、``steps[i].result``
     - 工具调用及执行反馈，包括请求、下发、确认和未确认的动作数；未知值保留为 ``null``。
   * - ``states.json``：``steps[i].state.robot_state``、``steps[i].state.artifacts.views``
     - 该记录的实测机器人状态与对应图像路径，可用于检查动作前后的变化。

Env 启动器不自动编码视频。工具保存的观测是稀疏记录，并非每个仿真步一帧；需要完整轨迹视频时，应通过独立记录程序采集中间帧。判断任务效果应使用包括 QScore 在内的官方任务指标；单独的 ``terminated`` 标志、规划器调用 ``finish`` 或进程正常退出均不证明任务成功。

启用更多工具
------------

运行 ``agent.yaml`` 前，在 Env 环境安装运动依赖。上方纯仿真安装配置不包含 OmniGibson 可选的 ``primitives`` 依赖；使用其固定的公开 CuRobo 版本和 ``warp-lang==1.12.0``，基于 Env 解释器已有的 PyTorch 2.7.0 / CUDA 12.8 编译。仅有 NVIDIA 驱动还不够，需要安装包含 ``nvcc`` 的 CUDA Toolkit 12.8，以及主机 C++ 构建工具。将 ``CUDA_HOME`` 指向实际 Toolkit 目录，``nvcc --version`` 应显示 release 12.8。

.. code-block:: bash

   export CUDA_HOME=/usr/local/cuda-12.8
   export PATH="$CUDA_HOME/bin:$PATH"
   nvcc --version
   uv pip install --python "$ENV_PYTHON" setuptools-scm wheel ninja pybind11
   MAX_JOBS=2 uv pip install --python "$ENV_PYTHON" --no-build-isolation \
     'warp-lang==1.12.0' \
     'nvidia-curobo @ git+https://github.com/StanfordVL/curobo@78612f45cef52c3fa0298de243a54cd7ca614414'
   "$ENV_PYTHON" -m pip check
   "$ENV_PYTHON" -c 'import importlib.metadata as m; import curobo; print({p: m.version(p) for p in ("torch", "warp-lang", "nvidia-curobo")}); print(curobo.__file__)'

保持关闭构建隔离，使 CUDA 扩展使用已安装的 Torch 编译，避免构建时引入另一版本。最后一条命令检查安装版本和导入位置；实际原生运动需另按“开发与测试”验证。

按前述顺序启动新回合时，做三处调整：

1. Env 命令增加 ``--motion-backend curobo --motion-device "cuda:$ENV_GPU"``。
2. Env 保留 ``--env-wrapper rgbd``，提供 RGB、深度、机器人状态，以及三维定位所需的实时相机标定；``rgb`` 不提供深度。
3. 按下方步骤准备并启动 SAM3，再将规划器配置改成 ``robots/behavior/configs/agent.yaml``，增加 ``--sam3-endpoint socket://127.0.0.1:28013``。保留首次运行命令中的任务、Env/VLA 地址、本地记忆和输出选项。

使用运动工具时设置 ``--action-frequency 30``。这些工具的实测速度检查和关节插值目前使用 30 Hz 控制周期；其他控制频率需要单独开展运动验证。

SAM3 在规划器环境中运行，与 Env、OpenPI 分开。通过 RPent 现有 SAM3 extra，安装配套的 PyTorch 2.7.0 / torchvision 0.22.0 CUDA 12.8 wheel：

.. code-block:: bash

   cd "$RPENT_REPO"
   uv pip install --python "$SAM3_PYTHON" --torch-backend cu128 \
     'torch==2.7.0' 'torchvision==0.22.0' -e '.[behavior,sam3]'
   uv pip check --python "$SAM3_PYTHON"

先在 `facebook/sam3 模型页面 <https://huggingface.co/facebook/sam3>`_ 申请访问，再下载 ``sam3.pt``。服务从本地文件加载模型；设置路径本身不会下载权重。

.. code-block:: bash

   "$PLANNER_VENV/bin/hf" auth login
   "$PLANNER_VENV/bin/hf" download facebook/sam3 sam3.pt \
     --local-dir "$(dirname "$SAM3_CHECKPOINT_PATH")"
   test -s "$SAM3_CHECKPOINT_PATH"
   "$SAM3_PYTHON" -m rpent.robots.components.sam3_server \
     --cuda-device "$SAM3_GPU" \
     --host 127.0.0.1 --port 28013 --transport socket

等待 SAM3 完成模型加载并报告 RPC 服务就绪后，再启动规划器。就绪不代表一定能够分割出指定物体，需查看实际返回的分数和 ``found`` 状态。SAM3 请求超时为 120 秒，与 Env 超时分别配置。规划器退出后，借用的 SAM3 服务仍会运行，应与本轮其他服务一起停止。

RGB 图像足以用于 VLA 和二维分割。使用三维坐标时，RGB、深度和相机标定必须来自同一帧。

``agent.yaml`` 默认开放以下 12 个机器人工具，此外还可使用 RPent 通用的文件与记忆工具。

.. list-table:: 工具选择
   :header-rows: 1

   * - 工具
     - 用途与主要限制
   * - ``view_env_state``、``read_robot_state``
     - 查看图像和实测机器人状态，不推进物理仿真；SDK 初始化相机标定时可能触发渲染。
   * - ``segment``、``pixel_to_robot_base``
     - 分割目标、查询像素对应的机器人基座坐标。像素按原图 ``[行, 列]`` 给出；机器人或目标移动后需重新定位。
   * - ``vla_act``
     - 预测并执行动作前缀，每个动作块后获取新观测。
   * - ``set_gripper``
     - ``val=0`` 闭合、``val=1`` 打开。插值与最终保持动作合计最多 5000 步。可额外在 allowlist 中启用 ``release`` 作为打开夹爪的简写。
   * - ``move_to``、``move_eef_delta``
     - 在当前机器人基座坐标系中移动单臂，位置单位为米。前者指定目标，后者指定小幅位移；均需要 CuRobo。
   * - ``rotate_wrist``
     - 经 CuRobo 调整末端绕基座 Z 轴的朝向，接受 −180° 至 180°，保持目标位置；运动中手指仍会扫过周围空间。
   * - ``rotate_wrist_joint``
     - 不经逆运动学，转动选定的 roll、pitch 或 yaw 关节，接受 −90° 至 90°；查看实测转角和限位截断提示。
   * - ``move_base``
     - 执行局部底盘速度命令，最后发送零速度；不保证到达指定位置。
   * - ``finish``
     - 停止规划器的决策循环并记录原因；不关闭借用的 Env 进程，也不将原生任务标记为成功。

Env 根据原生终止信号和配置的步数上限停止接收后续动作，保留最终指标。``finish`` 只停止 RPent 规划器；若在环境结束前调用，保存的运行仍是未完成回合，查看结果后需显式停止借用的 Env 服务。

手臂规划从稳定的实测姿态开始，使用 OmniGibson 的 ``CuRoboMotionGenerator`` 和已加载的 R1Pro 模型，只优化所选机械臂的七个关节。IK 求解时将底盘、躯干、另一臂和手指锁定在实测位置，执行时保留它们的保持命令。IK 求解器检查机器人自碰撞；局部目标求解后采用关节插值，不检查场景碰撞，也不保证整条插值路径无碰撞。请根据图像判断空间是否足够。工具完成需结合动作确认与实测位姿误差判断，夹爪闭合也不证明已抓住物体。

未显式指定 ``quat`` 的 ``move_to`` 和 ``move_eef_delta``，在无法保持当前朝向时允许只约束位置重试，并报告 ``orientation_relaxed``。显式指定 ``quat`` 的 ``move_to`` 和 ``rotate_wrist`` 保留朝向要求，不进行这种放松。IK 失败时不下发规划的手臂轨迹，可返回最近可达位置供调整目标。

常用参数
--------

命令行参数覆盖 ``--behavior-config`` 中的对应设置。通用规划器参数见 :doc:`../guides/cli`；以下是运行 BEHAVIOR 时常用的设置。

.. list-table::
   :header-rows: 1

   * - 设置
     - 用途与默认值
   * - ``--behavior-config``、``--tool-allowlist``
     - 选择模板或覆盖机器人工具列表；列表须保留 ``view_env_state`` 和 ``finish``。共享文件与记忆工具继续可用。
   * - ``--control-mode``、``--instruction-mode``
     - 本文使用 ``agent`` 控制，由规划器选择工具。``--instruction-mode main`` 表示每次 VLA 调用均使用完整任务指令，连接 Env 时会核对该指令。``override`` 允许为单次调用提供不同指令，需所选模型支持这种用法。
   * - ``--execution-horizon``、``--max-chunks-per-call``
     - 两份模板每块最多执行 32 步、每次工具调用最多预测 1 块；减少 horizon 可缩短执行前缀。
   * - ``--env-endpoint``、``--am-endpoint``、``--sam3-endpoint``
     - 接收 ``socket://`` 或 ``http://`` 内部服务地址。动作后端选择 ``--am-backend pi05``；``--vla-endpoint`` 仍作为动作模型地址的别名。
   * - Pi0.5 的 ``--model-path``、``--norm-stats-path``、``--repo-id``
     - 指定权重、配套归一化统计和 RLinf 数据配置的数据集标识。未传权重路径时读取 ``PI05_CHECKPOINT_PATH``；``--cuda-device`` 选择模型 GPU。
   * - Env 的 ``--env-config``、``--instance-split``、``--instance-id``
     - 组合 RLinf 仿真 YAML，并可覆盖实例选择。实例 ID 对应实际资源文件，公开测试实例从 301 开始。
   * - Env 的 ``--max-steps``、``--seed`` 和频率参数
     - 覆盖回合上限、随机种子以及动作、渲染、物理频率；省略时保留组合 YAML 中的设置。
   * - Env 的 ``--camera-size``、``--env-wrapper``、``--torch-num-threads``
     - 可设置统一方形相机分辨率。默认 ``rgbd`` 提供 RGB、机器人状态和线性深度；``rgb`` 不提供深度，``default`` 则保留 RLinf 基础观测配置。PyTorch CPU 线程数默认 4，不控制仿真器或 PhysX 的所有线程。
   * - 规划器的 ``--env-rpc-timeout-s``
     - 完整 Env 请求的超时时间；两份工具模板均设置为 1230 秒。
   * - Env 的 ``--motion-backend``、``--motion-device``
     - 后端默认 ``none``；启用 ``curobo`` 时默认使用 ``cuda:0``。

任务记忆
--------

BEHAVIOR 默认使用 ``--memory-profile local``。``--memory-dir`` 指定本地经验库，默认使用仓库下的 ``memory/behavior/``。规划器先查看初始观测，再读 ``MEMORY.md`` 和最多三篇相关条目；历史物体位置和抓取结果需由当前观测确认。评测期间经验库只读，不从远程同步。索引及 ``global/``、``task-family/`` 的组织方式见 :doc:`../development/memory`。

不带历史经验的运行可以使用空目录，首次运行命令即采用这种方式。保持目录为空即可，无需编造策略，也无需新增只读参数或改变系统文件权限。只读限制作用于记忆工具，运行产物仍写入 ``--output-dir``。

实验结果
--------

以下对比使用团队自训练的同一个 VLA：基线由 VLA 直接执行任务，另一组由 RPent 规划器调用该 VLA 和工具。每个任务、每种方法评测 10 个实例，报告官方评测器给出的平均 QScore。这些结果采集于此前使用官方评测器的接入版本，并非对当前 RLinf 适配器或上述公开权重重新开展的评测。

.. list-table::
   :header-rows: 1

   * - 任务
     - 纯 VLA 基线
     - RPent + 同一 VLA
   * - 打开收音机
     - 0.20
     - 0.80
   * - 制作微波爆米花
     - 0.70
     - 0.90
   * - 捡垃圾
     - 0.33
     - 0.73

实验结果表明，在使用相同自训练 VLA 的条件下，接入 RPent 后，三个任务的平均 QScore 均高于纯 VLA 基线。

开发与测试
----------

配置与注册入口为 ``robots/behavior/robot_spec.py``。``env_server.py`` 加载 RLinf 配置并拥有仿真进程，``rlinf_env.py`` 将本进程接口适配到现有 RPent Env 契约；工具实现仍位于同目录的 ``primitives.py`` 和 ``tools.py``。动作模型复用 ``rpent/robots/components/pi05_vla_client.py`` 和 ``pi05_vla_server.py``。

``motion.py`` 中的 ``R1ProMotionModel`` 接收 Env 进程提供的 ``simulator.get_robot``，读取原生关节限位及 ``robot.curobo_path`` 配置，并使用已加载机器人的 USD 模型求解 IK，无需另行维护一份运动学模型。

原生 Env 的验证配置使用 128×128 相机；其他分辨率需要单独开展运行验证。

另建 Linux CPU 测试环境，保留 SAM3 环境中的 CUDA 依赖。运行 BEHAVIOR 和共享 Pi0.5 离线测试：

.. code-block:: bash

   uv venv --python 3.11 /path/to/envs/behavior-tests
   export TEST_PYTHON=/path/to/envs/behavior-tests/bin/python
   uv pip install --python "$TEST_PYTHON" --torch-backend cpu -e '.[test,behavior]' torch
   "$TEST_PYTHON" -m pytest tests/unit_tests/robots/behavior \
     tests/unit_tests/rpent/robots/components/test_pi05_vla*.py -q

真实服务检查使用同一测试解释器，借用已启动的 GPU 服务。为专用回合完成启动步骤 1–2、等待 Env 就绪后，用下列命令代替步骤 3；无需规划器模型或凭据。串行执行测试，不同时运行其他动作客户端或 ``pytest-xdist``。

.. code-block:: bash

   export RPENT_RUN_BEHAVIOR_INTEGRATION=1
   export RPENT_BEHAVIOR_TASK_ID="$TASK_ID"
   export RPENT_BEHAVIOR_TASK_LANGUAGE="$("$TEST_PYTHON" -c "from robots.behavior.tasks import TASK_LANGUAGE; print(TASK_LANGUAGE[$TASK_ID])")"
   export RPENT_BEHAVIOR_ENV_ENDPOINT=socket://127.0.0.1:28010
   export RPENT_BEHAVIOR_VLA_ENDPOINT=socket://127.0.0.1:28011
   export RPENT_E2E_OUTPUT_DIR="$RPENT_RUN/integration"
   "$TEST_PYTHON" -m pytest tests/e2e_tests/behavior -v -s

未设置 ``RPENT_RUN_BEHAVIOR_INTEGRATION`` 时，测试跳过且不连接服务。``test_components.py`` 的观测与推理检查不下发动作。其中额外的原生运动检查要求 Env 以 ``--motion-backend curobo`` 启动，并设置 ``RPENT_RUN_BEHAVIOR_MOTION_INTEGRATION=1``：它先执行有界机械臂保持动作和 13 个显式稳定动作，再通过 Env RPC 为双臂当前末端位姿请求 IK，检查原生关节限位及观测标识，保存 ``motion-component.json``，不执行返回的 IK 路径。``test_policy_chain.py`` 通过 Toolkit 最多执行并确认 32 步，保存新观测并调用 ``finish(blocked)``。``blocked`` 表示有限步数检查结束，不表示任务失败。

在 ``RPENT_E2E_OUTPUT_DIR`` 查看预测、提交动作、确认计数和保存图像。这些测试验证组件和动作链，不代表自主 Planner 决策或整项任务效果；它们不启动、重置、停止服务，也不通过 ``run_gpu_suite.sh`` 启动。检查结果后，停止借用的服务。
