机器人 Episode
==============

用同一套四个工具在 LIBERO、RoboTwin、RoboCasa 或 RoboDojo 中运行模型。任务和原生成功判定由仿真器提供。

安装
----

每种仿真器使用独立的 Python 环境。在其中安装 RPent，并按仿真器指南准备运行包和资产。``requirements/libero-rl.txt``\ 、``requirements/rl-robotwin.txt`` 和 ``requirements/rl-robocasa.txt`` 分别提供对应的 episode 依赖。RoboDojo 使用自己的 Isaac 环境和 ``robodojo-sim`` extra。准备 SAM3 权重或已有的 SAM3 服务地址。模型服务使用独立环境。

接入 RLinf 时，从配套的 RPent 代码目录把 ``requirements/episode-planner.txt`` 装入 RL 环境。它提供 planner 和 RPC 客户端。仿真依赖装在运行仿真进程的主机上。

.. code-block:: bash

   uv pip install --python "$RL_PYTHON" -r requirements/episode-planner.txt
   uv pip install --python "$LIBERO_PYTHON" -r requirements/libero-rl.txt
   uv pip install --python "$ROBOTWIN_PYTHON" -r requirements/rl-robotwin.txt
   uv pip install --python "$ROBOCASA_PYTHON" -r requirements/rl-robocasa.txt
   uv pip install --python "$ROBODOJO_PYTHON" -e '.[robodojo-sim]' \
     --extra-index-url https://pypi.nvidia.com \
     --override requirements/robodojo-override.txt

将各变量设为对应虚拟环境中的 Python。仿真依赖与资产在各自后端机器上安装。

.. list-table::
   :header-rows: 1

   * - 仿真器
     - 已验证的运行环境
   * - LIBERO-PRO
     - Python 3.12、MuJoCo 3.3.0、robosuite 1.5.2 和 LIBERO-PRO 资产
   * - RoboTwin
     - Python 3.11、SAPIEN 3.0.0b1、RoboTwin runtime 0.1.3 和匹配的 cuRobo 构建
   * - RoboCasa
     - Python 3.11、MuJoCo 3.3.0、RoboCasa365 1.0.1 和固定版本的 robosuite
   * - RoboDojo
     - Python 3.11、Isaac Sim 5.1、配套 IsaacLab 和 cuRobo 构建，以及原生评测布局

运行评测
--------

激活相应仿真环境，在 RPent 代码目录运行。以下示例连接已有的 SAM3 池和兼容 OpenAI 接口的模型服务。
直跑时如使用模型自带的处理器计算 prompt 长度，请在当前解释器中安装匹配的
Transformers、Torch 和 Torchvision，并用 ``RPENT_TOKENIZER_PATH`` 指向本地 checkpoint。
使用远程 toolkit 时，只有 planner 解释器加载模型处理器。

.. code-block:: bash

   export RPENT_MODEL=openai-chat:Qwen3.8-27B
   export RPENT_BASE_URL=http://127.0.0.1:30000/v1
   export RPENT_SAM3_ENDPOINT=http://127.0.0.1:28424
   bash tools/rl/run.sh --simulator libero --libero-type pro \
     --suite libero_spatial --task 2 --seed 0 --cuda-device 0 \
     --world-cord general --position-unit cm \
     --max-turns 100 --max-tokens 32768 --reasoning-effort xhigh

其它仿真器用 ``--simulator`` 选择 ``robotwin``\ 、``robocasa`` 或 ``robodojo``，并通过 ``--task-name`` 指定任务。RoboDojo 还接受 ``--layout``。批量评测用 ``python tools/rl/run_batch.py run OUTPUT --jobs MANIFEST`` 读取 JSON 清单。``--pilot-jobs`` 可指定先运行的清单子集，结果计入同一批次。收集器逐条写入已审计结果，续跑时从剩余任务与 seed 开始。

接入 RLinf
----------

在仿真环境中启动后端，指定仿真器、GPU 槽位和 SAM3 地址。
RoboTwin 还需在启动管理器前，在后端环境中设置 ``ROBOTWIN_ASSETS_PATH``。

.. code-block:: bash

   python -m robots.rl.remote_manager --simulator libero \
     --gpu-slots 0:1,1:1 --sam3-endpoint http://127.0.0.1:28424 \
     --output-dir /path/to/backend --port 28490

将 RLinf 的 ``tools.rpent.python`` 指向 planner 解释器，将 ``tools.rpent.remote_toolkit_endpoint`` 指向 agent worker 可访问的后端地址。两端使用相同版本的 RPent。后端管理槽位并核验仿真器、工具契约和坐标配置。直接评测与远程调用使用同一适配器。

动作与感知
----------

- ``view_env`` 返回最新 RGB、相机名称和图像尺寸。模型只通过这个工具接收 RGB。
- ``segment`` 接受文字或一个点，用 SAM3 测量可见表面。``back_project`` 接受像素、框或已保存的 mask，查询拍摄时的深度。像素顺序是 ``[row,col]``，框采用半开区间 ``[row_min,col_min,row_max,col_max]``。点和框按对应观察步的实际图像大小校验，mask 绑定原相机和拍摄步。
- 感知回执标明相机、拍摄步、位置单位、拍摄坐标版本和结果坐标版本。历史观察保留拍摄时的 RGB-D 与标定，几何结果使用当前坐标表达。
- ``execute_actions`` 按顺序执行最多三项。RoboTwin 和 RoboDojo 的每个手臂项指定 ``left`` 或 ``right``。RoboCasa 可以把 ``move_base``\ 、``move_lift`` 和手臂动作放在同一批。执行器在每项开始前实测坐标系，并用该坐标系解释模型填入的原始目标。每项返回实测状态和残差，遇到部分完成、错误或终局时停止后续动作。

``--world-cord base`` 使用仿真世界坐标。``general`` 的原点为单臂安装座或双臂安装座中点，+x 向前、+y 向左、+z 向上。``--position-unit m|cm`` 决定线性量的输入与返回单位，角度使用弧度。提示词里的画面方向帮助定位物体，动作目标来自深度工具和实测状态。原生成功或预算耗尽后自动结束，模型提交最终文字答复时根据原生状态记录结果。
