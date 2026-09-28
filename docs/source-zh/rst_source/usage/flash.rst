Flash Mode
==========

**Flash 计划** 保存一个 LIBERO 任务的动作序列，并标出这些动作依赖的
关键物体或位置。重放时，RPent 从相机画面中找到它们在当前场景里的坐标，更新动作
坐标，然后按顺序执行计划中的动作。

整个过程可以概括为：

1. 计划决定 **做什么**。
2. SAM3 或点定位服务判断在当前场景中 **在哪里做**。
3. LIBERO toolkit 在更新后的坐标上执行动作。

因此，``--planner flash`` 不会调用 LLM 重新规划动作。定位服务在这里只负责视觉
定位：它在相机画面中指出指定的物体或位置，RPent 再将该像素转换成当前场景坐标。

全系列 LIBERO-PRO 性能与执行时间
--------------------------------

在完整的 800-case LIBERO-PRO 矩阵（Spatial、Object、Goal 和 Long；
task/swap；每个任务 10 个 seed）上，使用 Molmo2-8B 的 Flash Mode 成功 581 次（72.63%）。
不使用 reasoning 的 Codex 成功 500 次（62.50%），high reasoning Codex
成功 628 次（78.50%）。两个没有成功源轨迹、因而没有计划的任务
按 0/10 保守计入。

.. image:: https://github.com/RLinf/misc/raw/main/rpent/flash/flash_libero_pro_performance_time.png
   :alt: Flash Mode 与 Codex 在 LIBERO-PRO 全系列上的逐任务成功率和执行时间对比
   :width: 100%
   :align: center

时间统计不包含模型及服务启动时间。Codex 时间是每个任务可用 planner 耗时记录的
均值。Flash Mode 耗时采用每份最终计划对应成功 episode 的
工具执行时间（每份计划一个耗时样本）。所有方法的成功率都使用完整 800-case 矩阵。
两组 Codex baseline 均有完整的 800/800 planner 耗时记录。

重放流程
--------

每份计划包含一组动作和一组锚点（anchor）。锚点表示与任务有关的物体或位置，
例如需要抓取的物体、放置目标等。依赖锚点的动作记录的是相对锚点的偏移，而不只是
某次场景中的绝对坐标。

运行时，RPent 从计划中提取当前计划需要的锚点，并按照计划中记录的方式逐一定位：

* **SAM3** 处理分割类型的锚点，返回物体掩膜及其位置。
* **定位服务** 处理点定位类型的锚点，在相机画面中指出目标物体或位置。

RPent 将实时锚点位置与计划保存的偏移组合成新的路点，再执行对应动作。因此，
即使物体在新布局中换了位置，同一份计划仍能使用当前场景的坐标执行。

计划文件
--------

计划不随 Git 仓库提交，而是通过 Hugging Face 上的 `RLinf/RPent-memory 计划目录
<https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/libero/flash>`_
分发。RPent 在 HF memory 模式下自动下载计划，默认保存到
``memory/libero/flash``。使用 ``--memory-profile local --memory-dir /path/to/memory/libero``
时，从 ``/path/to/memory/libero/flash`` 读取，不下载数据。
80 个任务中有 78 份计划；``goal_swap_t0`` 和 ``10_swap_t9`` 暂无计划。
缺少计划或锚点文件时会报错。

.. code-block:: text

   memory/libero/flash/
     object_swap_t3_anchors.json   运行时需要定位的物体和位置
     object_swap_t3_plan.json      动作及其相对锚点的坐标

任务决定使用哪份计划；seed 只改变环境布局，不改变该任务使用的计划。

生成计划
--------

生成器从一条经模拟器确认成功的 episode 生成一份计划。必需输入只有
episode audit JSON 和与之匹配的 primitive recipe JSONL：

.. code-block:: bash

   python -m robots.libero.flash.generate \
     --audit results/goal_swap_t3_s7.json \
     --recipe results/goal_swap_t3_s7_recipe.jsonl \
     --destination memory/libero/flash

audit 必须包含非空的 ``task_language``（或 ``perturbed_task_language``）以及
``libero_terminated: true``。audit 和 recipe 的文件名，以及 audit 中存在的
suite/task/seed 字段，必须指向同一个 episode。

如果 episode 保存了 ``segment_*.json`` 读数，可通过 ``--segments`` 指定目录；
否则生成器会根据任务指令以及 recipe 中有序的抓取/释放或关节交互 transaction，
生成供点定位服务使用的语义锚点。附近的 ``move_to`` 和 ``move_pose`` 坐标会被保存成
相对锚点的 XY offset，供 Flash replay 直接使用。

关系解析覆盖 LIBERO-PRO 全部 80 个任务：Spatial、Object、Goal、Long（``10``）
各自的 task 和 swap suite。Long 的 transaction 顺序会被保留，例如先打开炉灶再
放置物体，或者先把物体放进设备再关闭设备。

如果只想手动下载计划，可以运行：

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
     --include "libero/flash/**" --local-dir memory

运行计划
--------

Flash Mode 仅用于评测，不能与 ``--explore`` 同用，执行 memory 中准备好的计划。
先启动定位服务，再把服务地址传给 RPent：

.. code-block:: bash

   rpent --robot libero --planner flash \
     --suite libero_object_swap --task 3 --seed 0 \
     --locator-endpoint http://127.0.0.1:20703

计划重放支持 LIBERO-PRO Spatial、Object、Goal、Long（``10``）的 task 和
swap suite，共 80 个 task identity。

VLA 和 SAM3 沿用普通 LIBERO 运行方式。也可以通过 ``--vla-endpoint`` 和
``--sam3-endpoint`` 连接已经启动的服务。

定位服务配置
------------

定位服务接收一张 RGB 图片和查询描述，返回输入原图上的一个像素点 ``(x, y)``；
无法定位目标时返回 ``None``。启动服务时选择后端，Flash 通过同一个接口调用。
全景粗定位、腕部精定位、深度反投影和持物偏移修正对所有后端采用相同流程。

各后端沿用 Molmo 的操作点提示词。API 和 Codex 另外追加 JSON 格式要求：
``{"point": [x, y]}``，坐标归一化到 0–1000；无法定位则返回 ``{"point": null}``。
服务端统一换算为原图像素。JSON 格式错误、非法坐标和服务调用失败会报错。
Molmo 保留原生点标记解析方式，解析不到有效点时仍按未找到目标处理。

Molmo
~~~~~

Molmo 需要的 ``transformers`` 版本比 LIBERO 策略环境更新，因此应在独立 Python
环境中运行：

.. code-block:: bash

   uv venv --python 3.11 /path/to/molmo-venv
   uv pip install --python /path/to/molmo-venv/bin/python -e ".[molmo]"

从 `Hugging Face <https://huggingface.co/allenai/Molmo2-8B>`_ 或
`ModelScope <https://modelscope.cn/models/allenai/Molmo2-8B>`_ 下载
``allenai/Molmo2-8B``，然后启动服务：

.. code-block:: bash

   export MOLMO_CHECKPOINT_PATH=/path/to/Molmo2-8B
   PYTHONPATH=/path/to/RPent /path/to/molmo-venv/bin/python \
     -m rpent.robots.components.locator_server --backend molmo \
     --transport http --host 127.0.0.1 --port 20703

云端 API
~~~~~~~~

API 后端复用 RPent 的 ``provider:model`` 模型名称和对应 provider 的凭据环境变量。
在服务端环境安装 RPent，并选择支持图片输入的模型；无需安装 Molmo extra 或配置 CUDA：

.. code-block:: bash

   python -m rpent.robots.components.locator_server \
     --backend api --model "<provider:model>" \
     --host 127.0.0.1 --port 20703

通过 ``--base-url`` 覆盖 provider 地址。凭据在服务端读取，规则与 API planner 一致，
详见 :doc:`configure_planner`。每次定位只发送一张图片，不携带历史对话。

使用 Claude 时，在定位服务端设置 ``ANTHROPIC_API_KEY``，并选择以 ``anthropic:``
开头的模型名称。将 ``<claude-model-id>`` 替换为 API 账号可用且支持图片输入的
Claude 模型 ID：

.. code-block:: bash

   python -m rpent.robots.components.locator_server \
     --backend api --model "anthropic:<claude-model-id>" \
     --host 127.0.0.1 --port 20703

服务会读取 ``ANTHROPIC_BASE_URL``；指定 ``--base-url`` 时以后者为准。
这里使用 Claude API 凭据，不使用 Claude Code 登录。

Codex 登录
~~~~~~~~~~

使用定位服务所在机器上的 Codex 安装和登录环境。必要时配置 ``CODEX_BIN``，然后启动：

.. code-block:: bash

   codex login
   python -m rpent.robots.components.locator_server \
     --backend codex --model "<model>" --reasoning-effort low \
     --host 127.0.0.1 --port 20703

省略 ``--model`` 时使用 Codex 配置的默认模型。每次定位在临时工作目录里创建独立的
临时会话，结束后关闭对应 Codex 进程。图片直接作为输入，要求返回 JSON 坐标，
不接入 RPent 工具。该后端使用服务端的 Codex 认证环境，包括已有的 ChatGPT 登录，
而非 RPent 客户端的凭据。需要限定临时文件位置时，设置 ``TMPDIR``。

超时、日志与迁移
~~~~~~~~~~~~~~~~

Flash 保留原有的 180 秒客户端 RPC 超时。定位服务不额外设置请求限时，
模型供应商 SDK 的默认超时仍然适用。
服务端启动时记录后端和模型，每次成功请求记录查询提示词、原始回答、图片尺寸、
像素结果和耗时；未找到目标的请求也会记录。API/Codex 会在日志中的操作点提示词后
追加上文所述的 JSON 格式要求。

服务启动模块改为 ``rpent.robots.components.locator_server``，替代 ``molmo_server``。
Python 调用改为 ``locator_client`` 中的 ``LocatorClient.locate(image, query)``，
替代 ``MolmoClient.ground``。RPC 方法改为 ``locator.locate``，客户端和服务端需要
一起升级，并使用 ``--locator-endpoint``。计划中的 ``locator: "point"`` 使用当前选择的
点定位后端，``segment`` 锚点使用 SAM3。已有计划中的 ``locator: "molmo"`` 在加载时
按 ``point`` 处理，使用当前选择的后端，无需修改计划文件。旧端点参数仍不支持。
不同模型的结果应分别报告：
上文性能数据使用 Molmo2-8B，不代表 API 或 Codex 的定位表现。

重放多个布局
------------

使用不同 seed 运行同一计划，即可在不同布局上执行。复用已经启动的 VLA、SAM3
和定位服务，可以避免每次运行都重新加载模型：

.. code-block:: bash

   for seed in $(seq 0 9); do
     rpent --robot libero --planner flash \
       --suite libero_object_swap --task 3 --seed "$seed" \
       --output-dir logs/sweep/swap_t3_s$seed \
       --vla-endpoint http://127.0.0.1:20701 \
       --sam3-endpoint http://127.0.0.1:20702 \
       --locator-endpoint http://127.0.0.1:20703
   done
