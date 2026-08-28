添加新机器人
============

本指南介绍如何将新的物理机器人或仿真环境接入 RPent 的 LLM-in-the-loop
runner。完整的参考实现见 ``robots/libero/``。

接入步骤概览
------------

RPent 的整体进程划分、服务职责和通信方式见 :doc:`系统设计 <architecture>`。
本页不再重复设计原理，只说明接入新机器人需要实现的扩展点。建议按以下顺序完成：

1. 在 :ref:`入口 <add-robot-entry>` 中注册 ``RobotSpec`` 和 toolkit 工厂。
2. 实现 :ref:`env_client 和 env_server <add-robot-env-rpc>`。如需接入 VLA
   服务和 model client，参见
   :ref:`添加一个 VLA（或其他基于模型的原语）<add-primitive-model-based>`。
3. :ref:`定义 prompt <add-robot-prompts>`。
4. :ref:`实现 toolkit 和 primitives <add-robot-toolkit>`。
5. :ref:`注册环境参数并生成 RunConfig <add-robot-config>`。
6. 实现 :ref:`runtime 钩子 <add-robot-runtime>`：同一个钩子既能为普通 CLI
   初始化完整 runtime，也能为 Dashboard 初始化指定的 component 子集。

.. _add-robot-entry:

入口
----

新增名为 ``myrobot`` 的机器人时，目录结构如下：

.. code-block:: text

   robots/myrobot/
       __init__.py            # 包入口，仅重导出两个工厂
       robot_spec.py          # RobotSpec、工厂、Dashboard 描述和 runtime 钩子
       env_client.py          # MyEnvClient —— agent 侧 RPC client (§1)
       prompt_bundle.py       # system()/user() prompt 工厂              (§2)
       toolkit.py             # MyRobotToolkit + primitives + 工具定义     (§3)
       env_server.py          # 环境侧 facade + RPC 服务                 (§1)
       vla_server.py          # （可选）VLA 模型服务

``__init__.py`` 是机器人包入口，应保持精简，仅重导出 ``robot_spec.py`` 中实现的
工厂。``rpent/robots/base.py`` 中的注册表会按需导入 ``robots.<name>``，并调用
这两个函数：

.. code-block:: python

   # robots/myrobot/__init__.py
   from robots.myrobot.robot_spec import get_robot_spec, get_toolkit

   # robots/myrobot/robot_spec.py
   from rpent.dashboard.events import DashboardEventSink
   from rpent.memory import MemoryManager
   from rpent.robots.robot_spec import RobotSpec, RunConfig
   from rpent.robots.prompt_bundle import PromptBundle
   from rpent.utils.config import get_memory_dir
   from robots.myrobot.prompt_bundle import system_prompt, user_prompt

   MYROBOT_DASHBOARD_SPEC = {...}

   def get_robot_spec() -> RobotSpec:
       return RobotSpec(
           name="myrobot",
           prompts=PromptBundle(system=system_prompt, user=user_prompt),
           add_cli_args=_add_cli_args,
           parse_config=_parse_config,
           init_runtime=_init_runtime,
           dashboard=MYROBOT_DASHBOARD_SPEC,
       )

   def get_toolkit(
       *,
       primitives_kwargs,
       dashboard_events: DashboardEventSink,
       config: RunConfig,
   ):
       from robots.myrobot.toolkit import MyRobotToolkit
       return MyRobotToolkit(
           primitives_kwargs=primitives_kwargs,
           dashboard_events=dashboard_events,
           memory=MemoryManager(
               root=config.prompt_vars.get("memory_dir") or get_memory_dir("myrobot"),
           ),
       )

   def _add_cli_args(parser, use_dashboard) -> None:
       """向共享 parser 注册机器人参数。见第 4 节。"""
       ...

   def _parse_config(args) -> RunConfig:
       """校验最终的 args，返回 RunConfig。见第 4 节。"""
       ...

   def _init_runtime(
       args,
       output_dir,
       dashboard_events: DashboardEventSink,
       components: set[str] | None,
   ):
       """初始化全部 runtime components，或只初始化指定子集。

       返回 (daemons, primitives_kwargs)。见第 5 节。
       """
       ...

``dashboard`` 是可选项；环境不支持 Dashboard 控制时保持为 ``None``。支持时，
在机器人包中定义该 spec：其中 ``task`` 描述命令、校验字段、展示模板和输出目录
slug，``runtime_components`` 与 ``frame_channels`` 描述前端展示的环境专用服务行
和相机视图。完整结构参考 ``robots/libero/robot_spec.py``。

``_resolve_robot(name)`` 通过 ``importlib.import_module(f"robots.{name}")``
动态加载机器人包。因此，只需将机器人包放在 ``robots/`` 下，无需维护中央注册列表。

下文依次说明这些模块需要实现的内容。``_add_cli_args`` 和 ``_parse_config``
见第 4 节，runtime 钩子见第 5 节。Dashboard spec 只由 Dashboard runner 使用。

.. _add-robot-env-rpc:

1. ``env_client.py`` + ``env_server.py``
-----------------------------------------

这两个文件连接 agent 进程与 ``env_server``。client 在 agent 进程内将方法调用
转换成 RPC 请求，``env_server`` 负责处理这些请求。

1.1 Env client（agent 侧）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

继承 :class:`rpent.robots.components.env_client_base.BaseEnvClient`。它已经负责启动时校验
``env.get_env_meta``、执行首次 reset、缓存 ``last_obs``，并实现公共的
``reset``、``step`` 和 ``chunk_step`` RPC。子类只需增加环境专用方法（LIBERO
增加了 ``render_camera``、``get_camera_meta``、``get_task_language`` 等）；扩展方法
需要独立超时时，再扩展超时表。RPC 名称需要保持稳定，因为服务端 facade 会显式
注册每个名称。

.. code-block:: python

   from rpent.robots.components.env_client_base import BaseEnvClient

   class MyEnvClient(BaseEnvClient):
       _TIMEOUT_S = {
           **BaseEnvClient._TIMEOUT_S,
           "env.render_camera": 120.0,
       }

       def render_camera(self, camera_name):
           return self._client.call(
               "env.render_camera",
               args=(camera_name,),
               timeout_s=self._TIMEOUT_S["env.render_camera"],
           )

   env = MyEnvClient(rpc_client, expected_meta=expected_meta)

1.2 Env server（环境侧）
~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``env_server`` 中定义与 client API 对应的 facade 类，例如
``MyEnvFacade``。该类继承
:class:`rpent.robots.components.env_facade_base.BaseEnvFacade`；基类已提供公共 RPC 路由和
读写分派锁。子类实现公共环境方法，并通过 ``_register_rpc`` 增加环境专用路由。
方法接收与 client 一致的位置参数和关键字参数，返回传输层支持的 Python / NumPy
值（不要返回 torch；agent 进程不导入 torch）。

.. code-block:: python

   from rpent.robots.components.env_facade_base import BaseEnvFacade

   class MyEnvFacade(BaseEnvFacade):
       def __init__(self, env, meta):
           self._env = env
           super().__init__()

       def _register_rpc(self):
           super()._register_rpc()
           # 自定义方法需额外注册
           self._rpc["env.custom_method"] = self.custom_method

       # BaseEnvFacade 要求的抽象方法必须实现
       def reset(self): ...
       def step(self, action): ...

       def custom_method(self, arg): ...

   facade = MyEnvFacade(env, meta)
   facade.serve(transport="http", host=host, port=port)

``BaseEnvFacade`` 通过 ``_register_rpc`` 注册公共路由，并使用读写锁串行化会改变
状态的调用。只有确认某个扩展路由可以安全地与其他读操作并发时，才把它加入
``_readonly_methods``。继承的 ``RpcFacade.serve`` 负责绑定传输方式（HTTP 或
socket）、提供 ``healthz`` 和 ``shutdown``、检测父进程退出并执行资源清理。

.. _add-robot-prompts:

2. ``prompt_bundle.py``
-----------------------

定义 ``system_prompt()`` 和 ``user_prompt()`` 两个 prompt 工厂，并在机器人的
``robot_spec.py`` 中构造
``PromptBundle(system=system_prompt, user=user_prompt)``（见上面的“入口”）。
每个工厂返回一个有序的 ``dict[str, PromptNode]``，其中包含带标题的分节；
``PromptBundle.render`` 负责组装和填充。一套 prompt 供 API loop、Claude Code
和 Codex 等 planner 共用。正文使用工具的裸名（如 ``move_to``），并说明 Claude
Code 和 Codex SDK 会将其显示为 ``mcp__rpent__<name>``；无需分别维护 CLI 与
API 版本。

.. code-block:: python

   # robots/myrobot/prompt_bundle.py
   from robots.myrobot.prompts import system as system_parts
   from robots.myrobot.prompts import user as user_parts
   from rpent.prompt.utils import PromptNode

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

将 prompt 内容保存在机器人包内，例如 ``robots/myrobot/prompts/system.py`` 和
``user.py``。分节内容可以是普通字符串，也可以使用 ``BulletList`` 或
``Numbered``。占位符
``{{suite}}`` / ``{{task}}`` / ``{{seed}}`` / ``{{output_dir}}`` /
``{{recipe_tag}}`` 在渲染时填充。

.. _add-robot-toolkit:

3. ``toolkit.py``
------------------

这个模块持有 LLM 能调用的一切: 工具 schema、primitives、每步状态 dump 以及
MCP allowlist。(LIBERO 中由于历史原因把这些拆到了 ``tools.py`` 和 ``toolkit.py``
两个文件; 新增 robot 时全部放在 ``toolkit.py`` 里没问题。)

toolkit 模块通常包含四部分：

**Primitives 类**\ （例如 ``MyRobotPrimitives``）是 toolkit 持有的 Python
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

- 在 ``super().__init__(...)`` 中传入 ``memory``（一个
  :class:`~rpent.memory.MemoryManager`）和 ``state``。``memory_access`` 和
  ``inbox_cell_tag`` 在构造 ``MemoryManager`` 时配置；eval 默认只读。
- 在 ``__init__`` 中通过自定义的初始化辅助方法构建 primitives（LIBERO
  中的方法名为 ``init_primitives``；它会调用 ``EnvState.reset()``、构造
  原语并 dump 第 0 步）,
- 用 ``self.add_tool(name, spec, handler)`` 注册每个工具。无状态的读取工具
  （如 ``view_env_state``、``finish``）直接绑定模块级函数；原语工具通过
  ``_step(name, **kwargs)`` 调用。``_step`` 使用
  ``getattr(self._primitives, name)(**kwargs)`` 调用 driver 方法并重新渲染状态；
- 重写 ``close()``，通过 ``EnvState`` 保存 agent 侧剩余工件（例如
  ``state.save("episode.mp4", frames, step=None)``）。

``primitives_kwargs`` 由 ``robot_spec.py:get_toolkit`` 转发给 toolkit，再原样传入
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

机器人特有的 CLI 参数通过两个钩子接入 ``rpent/cli/main.py`` 的解析流程，并参与
最终的 argparse 解析：

**``_add_cli_args(parser, use_dashboard) -> None``。** 将机器人参数注册到
main.py 已创建的共享 parser。``use_dashboard`` 决定原本必填的参数是否保持可选。
每个 Dashboard TaskRun 会在 ``parse_config`` 调用前，由机器人 Dashboard spec
定义的任务命令提供其声明的字段。main.py 会在 ``parser.parse_args()`` 之前调用
该钩子，因此 argparse 的 usage 和错误信息也会包含机器人参数。

**``_parse_config(args) -> RunConfig``。** 普通 CLI 模式下，该钩子在
``parser.parse_args()`` 后调用；Dashboard 模式下，每个 TaskRun 会先把请求字段
写入任务参数，再调用该钩子。该钩子校验这些字段并返回
:class:`~rpent.robots.RunConfig`：

- ``recipe_tag`` —— 单次运行的机器人标签，用于 transcript 文件名和 recipe 路径
  （LIBERO 使用 ``f"{suite.replace('libero_', '')}_t{task}_s{seed}"``）。
- ``output_dir`` —— 单次运行的临时目录路径。main.py 随后调用
  ``init_output_dir`` 创建目录并配置日志。
- ``prompt_vars`` —— 传给 ``PromptBundle.render`` 的字典，通常包含运行标识和
  prompt 引用的其他变量。
- ``task_desc`` —— 机器人特定的任务标识字典，会原样写入 transcript JSON 记录
  （LIBERO 使用 ``{"suite": ..., "task": ..., "seed": ...}``）。

.. code-block:: python

   def _add_cli_args(parser, use_dashboard) -> None:
       required = not use_dashboard
       parser.add_argument("--suite", default=None, required=required)
       parser.add_argument("--task", type=int, default=None, required=required)
       # ... 其他机器人参数 ...

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

``init_runtime`` 返回 ``(owned_daemons, primitives_kwargs)``：

- ``owned_daemons: list[ProcessDaemon]`` 只包含当前进程实际启动的子进程，
  当前 runner 会在清理阶段停止它们。连接外部 endpoint 时，不能把外部服务加入
  该列表。
- ``primitives_kwargs: dict`` 会传给 toolkit 构造器，再由后者传入 primitives
  的 ``__init__``。完整参数通常包含
  ``{"env": MyEnvClient(...), "model": VLAClient(...)}``，以及其他辅助 client。

第四个参数 ``components`` 指定要初始化的服务名称。``None`` 表示全部服务，普通
CLI 会传入这个值。Dashboard 根据 ``dashboard.runtime_components`` 得到两个子集，
每个 component 都必须显式声明 ``scope: "shared"`` 或 ``scope: "unique"``。Dashboard
先初始化一次 shared components，再为每个新的环境实例初始化 unique
components。两次都调用同一个钩子，最后合并返回的 ``primitives_kwargs``。在
LIBERO 中，这两个子集分别是 ``{"vla", "sam3"}`` 和 ``{"env"}``。

实现应在启动任何服务前拒绝未知 component 名称。如果多个选中的本地服务初始化
较慢，应先全部启动，再依次等待 ready，让初始化过程可以重叠。参考实现见
``robots/libero/robot_spec.py`` 中的有序 component registry。

endpoint（``--env-endpoint``、``--vla-endpoint``，以及 LIBERO 的
``--sam3-endpoint``）解析和环境专用服务命令，应放在拥有对应服务的钩子中。
这些 spawner 应通过 ``rpent.robots.runtime.try_spawn_server`` 和
``try_wait_server`` 组合，使各环境的状态事件、就绪失败和 owned daemon 清理保持
一致；runner 不处理这些环境细节。参考模式见 ``robots/libero/robot_spec.py`` 和
``robots/robocasa/robot_spec.py``。

冒烟测试
--------

代码可以正常编译后，运行以下最小冒烟测试：

.. code-block:: bash

   PI05_CHECKPOINT_PATH=<path> ANTHROPIC_API_KEY=<key> \
     rpent --robot myrobot --suite <suite> --task <id> --seed 0 \
     --output-dir /tmp/myrobot_smoke --planner api --model anthropic:claude-opus-4-8

.. note::

   共享 CLI parser 将 ``--robot`` 限定为 ``libero`` 和 ``robocasa``
   (见 ``rpent/cli/main.py``)。要让上面这条命令在全新的 ``myrobot`` 上跑通，
   需要先把新名字加到 ``rpent/cli/main.py`` 中 ``--robot`` 的
   ``choices=[...]`` 列表里。

预期结果是 agent 完成 prompt 中指定的任务并调用 ``finish``。运行结束后，
可在 ``<output_dir>/transcript_*.json`` 中查看总结。
