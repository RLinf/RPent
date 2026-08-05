核心接口
========

给新环境或新 primitive 接入 RPent 时，需要对接的接口如下。具体操作见
:doc:`add_robot`、:doc:`add_primitive`；仓库分层见 :doc:`architecture`。

环境入口
--------

把包放到 ``robots/<env>/`` 后，``main.py`` 会调用 ``__init__.py`` 里的两个函数：

.. code-block:: python

   def get_env_spec() -> EnvSpec: ...
   def get_toolkit(*, primitives_kwargs, video_path=None, dashboard=None): ...

``get_env_spec`` 返回 ``EnvSpec``，其中你需要提供：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段或钩子
     - 你要做什么
   * - ``name``
     - 环境名，对应 ``--env``。
   * - ``prompts``
     - ``PromptBundle``：``system`` 与 ``user`` 两套 prompt 工厂（见
       ``robots/<env>/prompt_bundle.py``）。
   * - ``add_cli_args``
     - 注册本环境的 CLI 参数（如 ``--suite``、``--env-endpoint``）。
   * - ``parse_config``
     - 校验参数并返回 ``RunConfig``；``recipe_tag``、``output_dir``、``prompt_vars``
       三项需由你正确填写（供 prompt 模板插值）。
   * - ``init_runtime``
     - 启动或连接 env 与 VLA 等子进程，构造 ``primitives_kwargs`` 字典
       （env 客户端、模型客户端等），供 toolkit 组装 primitives。

``get_toolkit`` 一般只需把 ``primitives_kwargs`` 传给环境子类；``video_path``、
 ``dashboard`` 由 ``main.py`` 传入，通常不用改。

参考实现：``robots/libero/__init__.py``。

Planner
-------

多数用户用内置 ``api``、``claude_code``、``codex``，见
:doc:`../usage/configure_planner`。自定义 planner 才需实现
``rpent.planner.base.Planner``：

.. code-block:: python

   def solve(
       self,
       *,
       system_prompt: str,
       user_message: str,
       toolkit: Toolkit,
       max_turns: int,
       input_queue=None,
   ) -> PlannerResult: ...

约定：用 ``toolkit.get_tools_spec()`` 把工具交给模型；每次调用 ``toolkit.execute_tool(name, input_dict)``；
把结果喂回模型；在 ``finish`` 工具或轮次用尽时返回 ``PlannerResult``。

工具集
------

在 ``robots/<env>/toolkit.py`` 里继承 ``Toolkit``，用 ``add_tool`` 注册环境工具：

.. code-block:: python

   def add_tool(self, name: str, spec: dict, handler) -> None: ...

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 参数
     - 含义
   * - ``name``
     - LLM 看到的工具名。
   * - ``spec``
     - 工具说明与参数 schema（``name``、``description``、``input_schema``）。
   * - ``handler``
     - 执行逻辑，须返回 ``dict``。任务结束时在dict里设 ``_finish``；
       需要回传相机图时可设 ``_image_bytes`` 等字段。

基类已注册公共文件工具；子类 ``super().__init__()`` 后追加本环境工具即可。逐步状态与
``view_env_state`` 见 :doc:`add_primitive`。

进程间通信
----------

接已有server或写 ``env_server`` / ``vla_server`` 时关注下面两点。

客户端端点（在 ``add_cli_args`` 里暴露，或在 ``init_runtime`` 里解析）：

.. code-block:: text

   [protocol://]host:port    # 未写协议时默认为 http

常见：``--env-endpoint``、``--vla-endpoint``。``http`` 为默认，走 ``POST /call``
传 JSON，其中 NumPy 数组编码为 ``{"__ndarray__": <base64>, "dtype": ..., "shape": ...}``；
观测数据很大、或是多帧堆叠的嵌套 NumPy 字典时可改 ``socket``，用带长度前缀的
pickle 数据帧传输，省掉反复的 JSON 编解码。pickle 不适合不可信输入，socket 只应连接可信端点。

服务端：继承 ``rpent.utils.rpc.RpcFacade``，实现 ``_dispatch`` 分发业务 RPC
（如 ``reset``、``step``、``predict``）。``healthz`` / ``shutdown`` 不必在子类里写。

细节见 :doc:`add_robot` 中的env_server与vla_server章节。
