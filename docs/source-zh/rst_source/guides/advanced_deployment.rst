远程服务与并行运行
===========================

默认情况下，RPent 会随每次 LIBERO 运行启动并关闭环境、VLA 与 SAM3 服务。单机运行时建议保留这一默认行为。只有当服务分布在不同主机，或需要跨任务复用 VLA 与 SAM3 模型时，才需要配置外部 endpoint。

三个参数分别设置对应服务的 endpoint：LIBERO 环境用 ``--env-endpoint``， Pi0.5 VLA 用 ``--vla-endpoint``，SAM3 用 ``--sam3-endpoint``。每个都取 ``[protocol://]HOST:PORT``，省略 protocol 时默认 HTTP，也可用 ``socket://`` 改走 socket RPC。

Dashboard Session 不支持 ``--env-endpoint``，因为每个 TaskRun 都需要使用新的环境服务；Dashboard 模式仍可使用 ``--vla-endpoint`` 和 ``--sam3-endpoint``。

LIBERO 环境服务
---------------

一个环境服务固定对应一组 suite、task、seed 和最大 episode 步数，这些参数必须与 RPent 客户端完全一致。在环境主机上运行：

.. code-block:: bash

   export LIBERO_TYPE=pro
   python -m robots.libero.env_server \
     --suite libero_object_swap --task 2 --seed 0 \
     --max-episode-steps 10000 \
     --cuda-device 0 \
     --transport http --host 0.0.0.0 --port ENV_PORT

环境服务与任务绑定。需要修改上述任一参数时，请先停止旧服务再重新启动。

Pi0.5 VLA 服务
--------------

在 VLA 主机上设置 checkpoint 路径并启动 HTTP 服务：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   python -m rpent.robots.components.pi05_vla_server \
     --embodiment libero \
     --cuda-device 0 \
     --transport http --host 0.0.0.0 --port VLA_PORT

VLA 服务只加载一次模型，可以由多个 RPent 运行复用。

.. _apxinf-vla-service-optional:

APXInf VLA 服务（可选）
------------------------------

APXInf 适配器可以替代 Pi0.5 服务。在已安装 RPent 的环境中，按照 `APXinf-robo 安装说明 <https://github.com/RLinf/APXinf-robo#build-apxinf-robo>`_ 安装推理依赖。上游安装说明中的 ``libero`` 和 ``serve`` extras 并非此 RPent RPC endpoint 的必需依赖。规划器和环境可以继续使用原有 LIBERO 安装及其 OpenPI 依赖。

准备兼容的 Pi0.5 LIBERO 模型权重、SentencePiece tokenizer 和归一化统计文件。

使用该环境的 Python 启动服务：

.. code-block:: bash

   python -m rpent.robots.components.apxinf_vla_server \
     --embodiment libero \
     --model-path /path/to/pi05-libero \
     --tokenizer-path /path/to/tokenizer.model \
     --norm-stats-path /path/to/norm_stats.json \
     --precision bf16 \
     --transport http --host 127.0.0.1 --port 8220

在 LIBERO 命令中增加 ``--vla-endpoint http://127.0.0.1:8220``。跨主机访问时，将服务绑定到可访问的网络接口，并使用服务端地址。用完后需自行停止这个外部服务；省略 endpoint 会启动 OpenPI。
APXInf-robo 原生的 OpenPI WebSocket 服务不是 RPent RPC endpoint，请使用上述适配器。

适配器仅支持 eval 模式，模型配置如下：

- ``libero``：两个视角、五次流匹配迭代、十步动作预测；返回前五步动作，使用环境单位和 float32 类型。
- ``dual_franka``：采用本体坐标系增量 TCP rot6d 的模型，输入三个视角；返回 20 步绝对位姿动作，每步 20 维。
- ``yam``：采用关节增量模型，输入三个视角；返回 30 步绝对 qpos14 动作，YAM 工具每次执行五步。

修改 ``--embodiment`` 时，需选择匹配的 APXInf-robo 预设、模型权重和统计文件。双臂 Franka 和 YAM 仍处于实验阶段，权重导出、数值一致性和真机执行尚需验证。适配器需要 APXInf 兼容的权重，不会转换原始 RLinf ``actor/model_state_dict/full_weights.pt``；目前不支持单臂 ``franka``。

SAM3 服务
---------

在 SAM3 主机上设置本地 checkpoint 路径并启动 HTTP 服务：

.. code-block:: bash

   export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt
   python -m rpent.robots.components.sam3_server \
     --cuda-device 0 \
     --transport http --host 0.0.0.0 --port SAM3_PORT

SAM3 服务只加载一次模型，可以由多个 RPent 运行复用。

连接 RPent
----------

在运行 RPent 的机器上，通过三个 endpoint 参数连接上述服务。suite、task、seed 和最大 episode 步数必须与环境服务的启动参数保持一致：

.. code-block:: bash

   rpent \
     --robot libero \
     --suite libero_object_swap --task 2 --seed 0 \
     --libero-type pro --max-episode-steps 10000 \
     --env-endpoint http://ENV_HOST:ENV_PORT \
     --vla-endpoint http://VLA_HOST:VLA_PORT \
     --sam3-endpoint http://SAM3_HOST:SAM3_PORT \
     --planner claude_code --model claude-opus-4-8

请将各 ``*_HOST`` 替换为运行对应服务的机器地址，并确保运行 RPent 的机器能访问该地址；将各 ``*_PORT`` 替换为启动服务时选择的空闲端口。三个 endpoint 参数可以分别省略，某项未指定时，RPent 会在当前机器上启动对应服务并自动选择空闲端口。三个服务省略 protocol 时都默认使用 HTTP，也都可以通过 ``socket://HOST:PORT`` 改用 socket RPC。

自定义 RLinf 源码
-----------------

环境服务和 VLA 服务需要导入 ``rlinf``。手动启动时如果使用开发中的 RLinf 源码而非已安装的包，请先把 ``PYTHONPATH`` 指向该源码目录：

.. code-block:: bash

   export PYTHONPATH=/path/to/rlinf:$PYTHONPATH

由 RPent 自动拉起的 server 无需手动设置：RLinf 源码路径从 ``RPENT_RLINF_ROOT`` （或 ``RLINF_REPO_PATH`` ）解析，默认回退到 RPent 仓库旁边的 ``rlinf`` 目录。解析出的路径不存在也无妨：Python 会忽略无效的 ``PYTHONPATH`` 条目，server 将导入已安装的 ``rlinf`` 包。

.. _libero-parallel-eval:

并行评测
--------

以下以使用 Pi0.5 VLA 和 SAM3 的 LIBERO 评测为例说明如何进行并行评测。其他机器人或评测配置可能使用不同的服务和 endpoint。

要对同一个 LIBERO 任务并行运行多次评测，先按照上文的说明各启动一个 Pi0.5 VLA 服务和一个 SAM3 服务。等待两个服务输出 ``RPC server listening on ...`` 后，为每个并发的 ``rpent`` 进程传入相同的 endpoint： ``http://VLA_HOST:VLA_PORT`` 和 ``http://SAM3_HOST:SAM3_PORT``，其中各占位符替换为对应服务的主机地址和端口。

如果服务与 RPent 在同一台机器上，host 可以使用 ``127.0.0.1``。这样所有进程会共同访问同一组 VLA 和 SAM3 服务。省略 ``--env-endpoint``，则每个进程会单独启动自己的 ``env_server``，评测环境彼此独立； VLA 和 SAM3 模型只需加载一次，无需为每次评测重复启动。

.. code-block:: bash

   pids=()

   for i in $(seq 1 10); do
     rpent --robot libero --libero-type pro \
       --suite libero_object_swap --task 2 --seed 0 \
       --planner claude_code --model claude-opus-4-8 \
       --vla-endpoint http://VLA_HOST:VLA_PORT \
       --sam3-endpoint http://SAM3_HOST:SAM3_PORT \
       --output-dir logs/parallel_object_swap_t2_s0/run_$i &
     pids+=($!)
   done

   wait "${pids[@]}"

.. note::

   通过 SSH 运行长时间评测时，请在 ``nohup`` 或 ``tmux`` / ``screen`` 会话中启动共享服务；直接使用 ``&`` 时，SSH shell 退出可能会结束服务。

远程访问 Dashboard
----------------------

在服务器上启动 Dashboard，并指定端口：

.. code-block:: bash

   rpent --robot libero --dashboard --dashboard-port 8000 \
     --planner claude_code --model claude-opus-4-8

在本机另一个终端建立端口转发，将 ``USER@HOST`` 替换为服务器登录地址：

.. code-block:: bash

   ssh -N -L 8000:127.0.0.1:8000 USER@HOST

在本机浏览器打开 ``http://127.0.0.1:8000``。保持 SSH 连接和服务器上的 RPent 进程运行；任务操作见 :doc:`dashboard`。

.. _code-as-policy-optional:

Code as Policy（可选）
------------------------------

这个模式让模型编写 Python 代码，通过 ``robo.<工具名>()`` 操作机器人。代码可以用循环和条件判断组织多步操作，变量会在同一次会话中保留。

先确认普通任务能正常运行，再在原来的命令后加上 ``--policy-mode code``，例如：

.. code-block:: bash

   rpent --robot libero --libero-type pro \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8 \
     --policy-mode code

每段代码默认最多运行 120 秒、调用 100 次机器人工具，可通过 ``--code-timeout-s`` 和 ``--code-max-calls`` 调整。代码和执行结果保存在运行输出目录下的 ``code_policy/`` 中。

模型通过 ``robo.finish(...)`` 结束任务，finish 成功后同一段代码中后面的语句不会再执行。``request_scene_reset``、``request_operator_verdict`` 等操作员请求仍是 Python 之外的独立工具，因为等待操作员的时间可能超过单段代码的时间限制。

该模式使用单独的提示词，不读取任务记忆，也不生成 recipe；支持 ``api``、``claude_code`` 和 ``codex`` planner，不支持 ``flash``，也不能与 ``--explore`` 一起使用。不加这个参数时，仍使用默认的工具调用模式。
