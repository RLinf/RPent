添加新机器人
============

本指南介绍如何将新的物理机器人或仿真环境接入 RPent 的 LLM-in-the-loop
runner。完整的参考实现见 ``robots/libero/``。

接入步骤概览
------------

RPent 的整体进程划分、服务职责和通信方式见 :doc:`系统设计 <architecture>`。
本页不再重复设计原理，只说明接入新机器人需要实现的扩展点。建议按以下顺序完成：

1. 在 :ref:`入口 <add-robot-entry>` 中注册 ``EnvSpec`` 和 toolkit 工厂。
2. 实现 :ref:`env_client 和 env_server <add-robot-env-rpc>`。如需接入 VLA
   服务和 model client，参见
   :ref:`添加一个 VLA（或其他基于模型的原语）<add-primitive-model-based>`。
3. :ref:`定义 prompt <add-robot-prompts>`。
4. :ref:`实现 toolkit 和 primitives <add-robot-toolkit>`。
5. :ref:`注册环境参数并生成 RunConfig <add-robot-config>`。
6. 实现 :ref:`runtime 钩子 <add-robot-runtime>`：普通 CLI 使用完整 runtime；
   如果环境支持 Dashboard 任务控制，再按 Session/TaskRun 生命周期实现仅供
   Dashboard 使用的拆分钩子。

.. _add-robot-entry:

入口
----

新增名为 ``myenv`` 的环境时，目录结构如下：

.. code-block:: text

   robots/myenv/
       __init__.py            # 入口 —— get_env_spec() / get_toolkit() 工厂
       env_client.py          # MyEnvClient —— agent 侧 RPC client (§1)
       prompt_bundle.py       # system()/user() prompt 工厂              (§2)
       toolkit.py             # MyEnvToolkit + primitives + 工具定义     (§3)
       env_server.py          # 环境侧 facade + RPC 服务                 (§1)
       vla_server.py          # （可选）VLA 模型服务
       spec.py                # （可选）Dashboard 描述

``__init__.py`` 是环境包的入口。``rpent/envs/base.py`` 中的注册表会按需导入
``robots.<name>``，并调用其中的两个工厂函数：

.. code-block:: python

   # robots/myenv/__init__.py
   from rpent.dashboard.events import DashboardEventSink
   from rpent.envs.env_spec import EnvSpec, RunConfig
   from rpent.envs.prompt_bundle import PromptBundle
   from robots.myenv.prompt_bundle import system_prompt, user_prompt
   from robots.myenv.spec import MYENV_DASHBOARD_SPEC

   def get_env_spec() -> EnvSpec:
       return EnvSpec(
           name="myenv",
           prompts=PromptBundle(system=system_prompt, user=user_prompt),
           add_cli_args=_add_cli_args,
           parse_config=_parse_config,
           init_shared_runtime=_init_shared_runtime,
           init_task_runtime=_init_task_runtime,
           init_runtime=_init_runtime,
           dashboard=MYENV_DASHBOARD_SPEC,
       )

   def get_toolkit(*, primitives_kwargs, dashboard_events: DashboardEventSink, video_path=None):
       from robots.myenv.toolkit import MyEnvToolkit
       return MyEnvToolkit(
           primitives_kwargs=primitives_kwargs,
           dashboard_events=dashboard_events,
           video_path=video_path,
       )

   def _add_cli_args(parser, use_dashboard) -> None:
       """向共享 parser 注册环境参数。见第 4 节。"""
       ...

   def _parse_config(args) -> RunConfig:
       """校验最终的 args，返回 RunConfig。见第 4 节。"""
       ...

   def _init_runtime(args, output_dir, dashboard_events: DashboardEventSink):
       """仅普通 CLI 使用：初始化完整 runtime。

       返回 (daemons, primitives_kwargs)。见第 5 节。
       """
       ...

   def _init_shared_runtime(args, output_dir, dashboard_events: DashboardEventSink):
       """仅 Dashboard 使用：初始化由 Session 持有并复用的服务。"""
       ...

   def _init_task_runtime(args, output_dir, dashboard_events: DashboardEventSink):
       """仅 Dashboard 使用：为每个 TaskRun 初始化全新的服务。"""
       ...

``dashboard`` 是可选项；环境不支持 Dashboard 控制时保持为 ``None``。支持时，
在环境包中定义该 spec：其中 ``task`` 描述命令、校验字段、展示模板和输出目录
slug，``runtime_components`` 与 ``frame_channels`` 描述前端展示的环境专用服务行
和相机视图。完整结构参考 ``robots/libero/spec.py``。

``_resolve_env(name)`` 通过 ``importlib.import_module(f"robots.{name}")``
动态加载环境包。因此，只需将环境包放在 ``robots/`` 下，无需维护中央注册列表。

下文依次说明这些模块需要实现的内容。``_add_cli_args`` 和 ``_parse_config``
见第 4 节，三个 runtime 钩子见第 5 节。Dashboard spec 只由 Dashboard runner
使用。

.. _add-robot-env-rpc:

1. ``env_client.py`` + ``env_server.py``
-----------------------------------------

这两个文件连接 agent 进程与 ``env_server``。client 在 agent 进程内将方法调用
转换成 RPC 请求，``env_server`` 负责处理这些请求。

1.1 Env client（agent 侧）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

基础接口包含 ``reset`` 和 ``step`` 两个 Gym 风格的方法；可以根据环境需要
添加其他方法（LIBERO 增加了 ``chunk_step``、``render_camera``、
``get_camera_meta``、``cached_image`` 等）。每个方法通过
``RpcClient.call("<rpc-name>", args=..., kwargs=...)`` 转发，并设置单独的
超时时间。方法名需要保持稳定，因为服务端会按名称分派请求。

.. code-block:: python

   class MyEnvClient:
       def __init__(self, client: RpcClient, *, return_all_frames: bool = False):
           self._client = client
           self.return_all_frames = return_all_frames

       def reset(self):
           return self._client.call("env.reset", timeout_s=120.0)

       def step(self, action):
           return self._client.call("env.step", args=(action,), timeout_s=60.0)
       # ... 根据 env 需要添加其他方法

1.2 Env server（环境侧）
~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``env_server`` 中定义与 client API 对应的 facade 类，例如
``MyEnvFacade``。该类继承 :class:`rpent.utils.rpc.RpcFacade`，实现
``_dispatch(method, args, kwargs)``，将 ``env.*`` 请求分派给对应方法，再通过
``self.serve(...)`` 启动服务。方法接收与 client 一致的位置参数和关键字参数，
返回可 pickle 的值（使用 numpy，不要返回 torch；agent 进程不导入 torch）。

.. code-block:: python

   from rpent.utils.rpc import RpcFacade

   class MyEnvFacade(RpcFacade):
       def __init__(self, env, meta):
           super().__init__()
           self._env = env
           self._meta = meta

       def _dispatch(self, method, args, kwargs):
           if method.startswith("env."):
               return getattr(self, method[len("env."):])(*args, **kwargs)
           raise ValueError(f"unknown RPC method: {method!r}")

       def reset(self): ...
       def step(self, action): ...

   facade = MyEnvFacade(env, meta)
   facade.serve(transport="http", host=host, port=port)

``RpcFacade.serve`` 负责绑定传输方式（HTTP 或 socket）、提供 ``healthz`` 和
``shutdown`` 方法、检测父进程退出并执行资源清理；这里只需实现业务方法。

.. _add-robot-prompts:

2. ``prompt_bundle.py``
-----------------------

定义 ``system_prompt()`` 和 ``user_prompt()`` 两个 prompt 工厂，并在环境的
``__init__.py`` 中构造
``PromptBundle(system=system_prompt, user=user_prompt)``（见上面的“入口”）。
每个工厂返回一个有序的 ``dict[str, PromptNode]``，其中包含带标题的分节；
``PromptBundle.render`` 负责组装和填充。一套 prompt 供 API loop、Claude Code
和 Codex 等 planner 共用。正文使用工具的裸名（如 ``move_to``），并说明 Claude
Code 和 Codex SDK 会将其显示为 ``mcp__rpent__<name>``；无需分别维护 CLI 与
API 版本。

.. code-block:: python

   # robots/myenv/prompt_bundle.py
   from robots.myenv.prompts import system as system_parts
   from robots.myenv.prompts import user as user_parts
   from rpent.context.prompt_utils import PromptNode

   def system_prompt() -> PromptNode:
       return {
           "INTRO": system_parts.PREAMBLE,
           "GOAL": system_parts.GOAL,
           "RULES": system_parts.RULES,
           "WORKFLOW": system_parts.WORKFLOW,
           "ENVIRONMENT": system_parts.ENVIRONMENT,
           "OUTPUT": system_parts.OUTPUT,
       }

   def user_prompt() -> PromptNode:
       return {
           "TASK": user_parts.TASK,
           "BEGIN": user_parts.BEGIN,
       }

将 prompt 内容保存在环境包内，例如 ``robots/myenv/prompts/system.py`` 和
``user.py``。分节内容可以是普通字符串，也可以使用 ``BulletList`` 或
``Numbered``。占位符
``{{suite}}`` / ``{{task}}`` / ``{{seed}}`` / ``{{output_dir}}`` /
``{{recipe_tag}}`` 在渲染时填充。

.. _add-robot-toolkit:

3. ``toolkit.py``
------------------

这个模块持有 LLM 能调用的一切: 工具 schema、primitives、每步状态 dump 以及
MCP allowlist。(LIBERO 中由于历史原因把这些拆到了 ``tools.py`` 和 ``toolkit.py``
两个文件; 新增 env 时全部放在 ``toolkit.py`` 里没问题。)

toolkit 模块通常包含四部分：

**Primitives 类**\ （例如 ``MyEnvPrimitives``）是 toolkit 持有的 Python
对象。它保存 ``EnvClient``、VLA ``model`` client 和单次运行所需的状态。每个
原语工具（``move_to``、``pi0_pick``、``release`` 等）对应一个方法，并返回
日志字典。

**工具定义和处理函数** 包括模块级的 ``TOOLS_SPEC`` 列表（列表元素采用
Anthropic API 的工具定义格式，包含 ``name``、``description`` 和
``input_schema``），以及 toolkit 引用的模块级函数，例如
``view_env_state``、``back_project`` 和 ``finish``。

**每步状态 dump** —— ``dump_state(driver, env_state, log)`` 通过
``env_state.record_step(...)`` 创建由 ``EnvState`` 持有的步骤，并取得分配的
step index；该 ``StepRecord`` 会被立即追加并提交。大型观测通过
``env_state.save(...)`` 保存——在 ``record_step`` 块内可省略 ``step`` 参数
（默认指向刚创建的步骤），传显式 ``step=<int>`` 可指定其它步骤，``step=None``
用于运行级工件。每次保存成功后，``EnvState`` 会自动把基础文件名加入该
``StepRecord`` 的扁平 ``artifacts`` 集合；读取方直接使用规范化的工件文件名。

**Toolkit 类** 继承 ``rpent.tools.toolkit.Toolkit``：

- 在 ``__init__`` 中通过自定义的初始化辅助方法构建 primitives（LIBERO
  中的方法名为 ``init_primitives_clean``；它会调用 ``EnvState.reset()``、构造
  原语并 dump 第 0 步）,
- 用 ``self.add_tool(name, spec, handler)`` 注册每个工具。无状态的读取工具
  （如 ``view_env_state``、``finish``）直接绑定模块级函数；原语工具通过
  ``_step(name, **kwargs)`` 调用。``_step`` 使用
  ``getattr(self._primitives, name)(**kwargs)`` 调用 driver 方法并重新渲染状态；
- 重写 ``close()``，通过 ``EnvState`` 保存 agent 侧剩余工件（例如
  ``state.save("episode.mp4", frames, step=None)``）。

``primitives_kwargs`` 由 ``__init__.py:get_toolkit`` 转发给 toolkit，再原样传入
primitives 的 ``__init__``。其中通常包含
``{"env": MyEnvClient(...), "model": VLAClient(...), ...}``。

建议遵循的约定
--------------

- ``output_dir`` 是 runner 为单次运行创建的工作目录。环境观测由
  ``EnvState`` 管理；调用方只使用逻辑基础文件名，不自行拼接存储路径。
  transcript 等运行管理输出与环境工件共享该目录。
- 工具定义使用 Anthropic API 格式（``name`` / ``description`` /
  ``input_schema``）。
  每个用 ``self.add_tool(...)`` 注册的工具都会暴露给所有 planner。
- 环境侧的返回值必须可 pickle，且不包含 torch 对象。
- 每个原语工具执行后要 dump 一次新的状态快照, 这样下一次
  ``view_env_state`` 看到的是动作后的世界。
- ``dump_state`` 是 Agent 获取环境状态的唯一数据来源；任何新的模态
  （例如触觉、力）都通过它提供。

.. _add-robot-config:

4. ``_add_cli_args`` + ``_parse_config`` (runner 钩子)
------------------------------------------------------

环境特有的 CLI 参数通过两个钩子接入 ``rpent/cli/main.py`` 的解析流程，并参与
最终的 argparse 解析：

**``_add_cli_args(parser, use_dashboard) -> None``。** 将环境参数注册到
main.py 已创建的共享 parser。``use_dashboard`` 决定原本必填的参数是否保持可选。
每个 Dashboard TaskRun 会在 ``parse_config`` 调用前，由环境 Dashboard spec
定义的任务命令提供其声明的字段。main.py 会在 ``parser.parse_args()`` 之前调用
该钩子，因此 argparse 的 usage 和错误信息也会包含环境参数。

**``_parse_config(args) -> RunConfig``。** 普通 CLI 模式下，该钩子在
``parser.parse_args()`` 后调用；Dashboard 模式下，每个 TaskRun 会先把请求字段
写入任务参数，再调用该钩子。该钩子校验这些字段并返回
:class:`~rpent.envs.RunConfig`：

- ``recipe_tag`` —— 单次运行的环境标签，用于 transcript 文件名和 recipe 路径
  （LIBERO 使用 ``f"{suite.replace('libero_', '')}_t{task}_s{seed}"``）。
- ``output_dir`` —— 单次运行的临时目录路径。main.py 随后调用
  ``init_output_dir`` 创建目录并配置日志。
- ``prompt_vars`` —— 传给 ``PromptBundle.render`` 的字典，通常包含运行标识和
  prompt 引用的其他变量。
- ``task_desc`` —— 环境特定的任务标识字典，会原样写入 transcript JSON 记录
  （LIBERO 使用 ``{"suite": ..., "task": ..., "seed": ...}``）。

.. code-block:: python

   def _add_cli_args(parser, use_dashboard) -> None:
       required = not use_dashboard
       parser.add_argument("--suite", default=None, required=required)
       parser.add_argument("--task", type=int, default=None, required=required)
       # ... 其他环境参数 ...

   def _parse_config(args) -> RunConfig:
       if not args.suite: raise ValueError("--suite is required")
       # ... 生成 recipe_tag、output_dir 和 prompt_vars ...
       return RunConfig(
           recipe_tag=recipe_tag,
           output_dir=output_dir,
           prompt_vars=prompt_vars,
           task_desc={"suite": args.suite, "task": args.task, "seed": args.seed},
       )

.. _add-robot-runtime:

5. Runtime 初始化钩子
---------------------

三个 runtime 钩子都返回 ``(owned_daemons, primitives_kwargs)``：

- ``owned_daemons: list[ProcessDaemon]`` 只包含当前进程实际启动的子进程，
  当前 runner 会在清理阶段停止它们。连接外部 endpoint 时，不能把外部服务加入
  该列表。
- ``primitives_kwargs: dict`` 会传给 toolkit 构造器，再由后者传入 primitives
  的 ``__init__``。完整参数通常包含
  ``{"env": MyEnvClient(...), "model": VLAClient(...)}``，以及其他辅助 client。

``init_runtime`` 是普通 CLI 钩子。``parse_config`` 返回后，``main.py`` 调用它
一次，初始化完整 runtime。当前 LIBERO 实现会启动或连接 ``env_server``、
``vla_server`` 和 ``sam3_server``，并一次性返回全部 primitive 参数。

``init_shared_runtime`` 和 ``init_task_runtime`` 是 **仅供 Dashboard 使用** 的
钩子，普通 CLI 不会调用。Dashboard 会在每个 Session 中调用一次
``init_shared_runtime``，初始化 Session 持有的复用服务；随后为每个全新的
TaskRun 调用 ``init_task_runtime``，并合并两者返回的 ``primitives_kwargs``。
具体如何拆分取决于环境自身的生命周期。LIBERO 将 VLA 和 SAM3 放在 Session
范围，将环境放在 TaskRun 范围；其他环境应采用适合自身服务的拆分，不必机械照搬。

endpoint（``--env-endpoint``、``--vla-endpoint``，以及 LIBERO 的
``--sam3-endpoint``）解析、子进程启动和 runtime 状态事件，应放在拥有对应服务的
钩子中，runner 不处理这些环境细节。参考实现见 ``robots/libero/__init__.py``。

冒烟测试
--------

代码可以正常编译后，运行以下最小冒烟测试：

.. code-block:: bash

   PI05_CHECKPOINT_PATH=<path> ANTHROPIC_API_KEY=<key> \
     rpent --env myenv --suite <suite> --task <id> --seed 0 \
     --output-dir /tmp/myenv_smoke --planner api --model anthropic:claude-opus-4-8

.. note::

   共享 CLI parser 将 ``--env`` 限定为 ``libero`` 和 ``robocasa``
   (见 ``rpent/cli/main.py``)。要让上面这条命令在全新的 ``myenv`` 上跑通，
   需要先把新名字加到 ``rpent/cli/main.py`` 中 ``--env`` 的
   ``choices=[...]`` 列表里。

预期结果是 agent 完成 prompt 中指定的任务并调用 ``finish``。运行结束后，
可在 ``<output_dir>/transcript_*.json`` 中查看总结。
