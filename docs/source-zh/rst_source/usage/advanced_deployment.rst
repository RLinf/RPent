高级部署
========

默认情况下, RPent 会随每次 LIBERO 运行启动并关闭环境、VLA 与 SAM3
服务。单机运行时建议保留这一默认行为。只有在服务分布于不同主机, 或需要
跨任务复用 VLA 与 SAM3 模型时, 才需要配置外部 endpoint。

三个 endpoint 使用不同的传输方式:

.. list-table::
   :header-rows: 1

   * - 服务
     - RPent 参数
     - Endpoint 格式
   * - LIBERO 环境
     - ``--env-endpoint``
     - Socket RPC, ``HOST:PORT``
   * - Pi0.5 VLA
     - ``--vla-endpoint``
     - HTTP, ``http://HOST:PORT``
   * - SAM3
     - ``--sam3-endpoint``
     - HTTP, ``http://HOST:PORT``

LIBERO 环境服务
---------------

一个环境服务固定对应一组 suite、task、seed 和最大 episode 步数。这些参数
必须与 RPent 客户端完全一致。在环境主机上运行:

.. code-block:: bash

   export LIBERO_TYPE=pro
   export CUDA_VISIBLE_DEVICES=0
   python -m robots.libero.env_server \
     --suite libero_object_swap --task 2 --seed 0 \
     --max-episode-steps 10000 \
     --output-dir /path/to/env-output \
     --transport-host 0.0.0.0 --transport-port ENV_PORT

环境服务与任务绑定。需要修改上述任一参数时, 请先停止旧服务并重新启动。

Pi0.5 VLA 服务
--------------

在 VLA 主机上设置 checkpoint 路径并启动 HTTP 服务:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   export CUDA_VISIBLE_DEVICES=0
   python -m robots.libero.vla_server --host 0.0.0.0 --port VLA_PORT

VLA 服务只加载一次模型, 可以由多个 RPent 运行复用。

SAM3 服务
---------

在 SAM3 主机上设置本地 checkpoint 路径并启动 HTTP 服务:

.. code-block:: bash

   export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt
   export CUDA_VISIBLE_DEVICES=0
   python -m robots.libero.sam3_server \
     --host 0.0.0.0 --port SAM3_PORT

SAM3 服务只加载一次模型, 可以由多个 RPent 运行复用。

连接 RPent
----------

在运行 RPent 的机器上, 通过三个 endpoint 参数连接上述服务。suite、task、
seed 和最大 episode 步数必须与环境服务的启动参数保持一致:

.. code-block:: bash

   rpent \
     --suite libero_object_swap --task 2 --seed 0 \
     --libero-type pro --max-episode-steps 10000 \
     --env-endpoint ENV_HOST:ENV_PORT \
     --vla-endpoint http://VLA_HOST:VLA_PORT \
     --sam3-endpoint http://SAM3_HOST:SAM3_PORT \
     --planner claude_code --model claude-opus-4-7

请将各 ``*_HOST`` 替换为运行对应服务的机器地址, 并确保运行 RPent 的机器
可以访问该地址; 将各 ``*_PORT`` 替换为启动服务时选择的空闲端口。三个
endpoint 参数可以分别省略; 某项未指定时, RPent 会在当前机器上启动对应
服务并自动选择空闲端口。
