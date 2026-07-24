添加新机器人
============

本指南介绍如何将新的物理机器人或仿真环境接入 RPent 的 LLM-in-the-loop
runner。完整的参考实现见 ``robots/libero/``。

RPent 将每个环境的实现分为 Agent 侧和 Driver 侧：

- **Agent 侧**\ （``robots/<env>/``）—— 运行在 Agent 进程内，提供工具 schema、
  primitive driver 逻辑和 prompt。
- **Driver 侧**\ （``robots/<env>/env_server.py``）—— 负责运行仿真器或机器人，
  并通过 :class:`rpent.utils.rpc.RpcFacade` 对外提供环境接口。默认使用 HTTP；
  当观测数据包含多帧历史信息或采用嵌套数据结构时，也可以通过
  ``--transport socket`` 切换到 pickle-framed TCP。

两侧通过 ``EnvClient`` 连接。Agent 侧的每次方法调用都会由 ``EnvClient``
转换成发往 Driver 侧的 RPC 请求。

VLA 模型运行在独立进程中
------------------------

当一个环境使用 VLA 策略（读取相机观测并输出动作的学习模型）时，模型运行在
**第三个独立进程** 中，不在 ``env_server`` 中加载：

- **VLA 侧**\ （``robots/<env>/vla_server.py``）—— 只加载 VLA 策略及其 GPU
  权重，通过独立的 RPC/HTTP 端点提供 ``vla_load``、``vla_infer`` 和
  ``vla_reset``，不导入仿真器依赖。
- toolkit 除了 ``EnvClient``，还接收一个 **model client**\ （LIBERO/Pi0.5
  使用 ``VLAClient``，RoboCasa/RLDX-1 使用 ``RLDXVLAClient``）作为
  ``model`` 参数。两个 client 分别连接不同的 server 进程。

**为什么必须分离：** 模型和仿真器对 CUDA、依赖项及进程生命周期有不同要求。
模型需要加载大规模 GPU 权重及 ``transformers``、``openpi`` 等依赖，仿真器则依赖
MuJoCo/robosuite 和绑定主线程的 EGL 渲染。将二者放在同一进程中会耦合依赖和
生命周期，模型发生 OOM 时也会导致仿真进程退出。分离后，模型与仿真器可以独立
重启、扩容或部署到远程主机（``--vla-endpoint host:port`` 可复用正在运行的模型
server）。每个环境都必须遵守这一划分：``env_server`` 运行仿真，
``vla_server`` 加载模型。

**传输协议可以因环境而异，但架构保持不变。** LIBERO 的 ``env_server`` 和
``vla_server`` 默认都使用 HTTP。当观测数据包含多帧历史信息或嵌套的 numpy
字典时，可以使用 pickle-framed socket（``--transport socket``），避免 JSON
重复编码的开销。两种传输方式都通过 :class:`RpcFacade` 提供相同的
``predict`` 和 ``env.*`` 方法接口。可以根据观测结构选择传输方式，但 env/vla
的进程划分保持不变。

**任何需要仿真 env 对象的逻辑都留在 env_server。** 对 RoboCasa 这样的 env,
抓取检测、动作组装等操作需要活的仿真 env, 因此是 env_server 的 RPC —— **不** 属于
VLA server。于是 agent 侧的 skill 同时持有两个 client: env client 做 render/step,
model client 做推理。

入口
----

新增名为 ``myenv`` 的 env 时, 文件布局如下:

.. code-block:: text

   robots/myenv/
       __init__.py            # 入口 —— get_env_spec() / get_toolkit() 工厂
       env_client.py          # MyEnvClient —— agent 侧 RPC 代理 (§1)
       prompt_bundle.py       # system()/user() prompt 工厂              (§2)
       toolkit.py             # MyEnvToolkit + primitives + tool schemas (§3)
       env_server.py          # driver 侧 facade + RPC server (§1)
       vla_server.py          # (可选) VLA 模型 server (§1)

``__init__.py`` 是环境包的入口。``rpent/envs/base.py`` 中的注册表会按需导入
``robots.<name>``，并调用其中的两个工厂函数：

.. code-block:: python

   # robots/myenv/__init__.py
   from rpent.envs.env_spec import EnvSpec, RunConfig
   from rpent.envs.prompt_bundle import PromptBundle
   from robots.myenv.prompt_bundle import system_prompt, user_prompt

   def get_env_spec() -> EnvSpec:
       return EnvSpec(
           name="myenv",
           prompts=PromptBundle(system=system_prompt, user=user_prompt),
           add_cli_args=_add_cli_args,
           parse_config=_parse_config,
           init_runtime=_init_runtime,
       )

   def get_toolkit(*, primitives_kwargs, video_path=None):
       from robots.myenv.toolkit import MyEnvToolkit
       return MyEnvToolkit(primitives_kwargs=primitives_kwargs, video_path=video_path)

   def _add_cli_args(parser, use_dashboard) -> None:
       """向共享 parser 注册环境参数。见第 4 节。"""
       ...

   def _parse_config(args) -> RunConfig:
       """校验最终的 args，返回 RunConfig。见第 4 节。"""
       ...

   def _init_runtime(args, output_dir):
       """启动 env_server 和 vla_server，构造 primitives_kwargs。

       返回 (daemons, primitives_kwargs)。见第 5 节。
       """
       ...

``_resolve_env(name)`` 通过 ``importlib.import_module(f"robots.{name}")``
动态加载环境包。因此，只需将环境包放在 ``robots/`` 下，无需维护中央注册列表。

下面各章分别说明上面引用的模块各自需要写什么;
``_add_cli_args`` / ``_parse_config`` 在 §4 覆盖, ``_init_runtime`` 在 §5。

1. ``env_client.py`` + ``env_server.py``
-----------------------------------------

这两个文件组成 Agent 侧与 Driver 侧之间的桥梁。client 在 Agent 进程内将方法
调用转换成 RPC 请求，``env_server`` 在 Driver 进程内处理请求。

1.1 Env client (agent 侧)
~~~~~~~~~~~~~~~~~~~~~~~~~

基础接口包含 ``reset`` 和 ``step`` 两个 gym 风格的方法；可以根据环境需要
添加其他方法（LIBERO 增加了 ``chunk_step``、``render_camera``、
``get_camera_meta``、``cached_image`` 等）。每个方法通过
``RpcClient.call("<rpc-name>", args=..., kwargs=...)`` 转发，并设置各自的
timeout。方法名需要保持稳定，因为 Driver 侧的 dispatcher 会按名称匹配。

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

1.2 Env server (driver 侧)
~~~~~~~~~~~~~~~~~~~~~~~~~~

在 driver 侧用 facade 类 (例如 ``MyEnvFacade``) 镜像 client 的 API。继承
:class:`rpent.utils.rpc.RpcFacade`, 实现 ``_dispatch(method, args, kwargs)``
把 ``env.*`` 路由到自己的方法, 用 ``self.serve(...)`` 起服务。方法接收
与 client 发送一致的位置 / 关键字参数, 返回可 pickle 的值 (numpy,
不要 torch —— agent 侧不 import torch)。

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

``RpcFacade.serve`` 负责绑定 transport（http / socket）、提供 ``healthz`` 与
``shutdown`` 方法、检测父进程退出并执行资源清理；这里只需实现业务方法。

2. ``prompt_bundle.py``
-----------------------

定义两个 prompt 工厂 —— ``system_prompt()`` 和 ``user_prompt()`` —— 并在 env 的
``__init__.py`` 中构造 ``PromptBundle(system=system_prompt, user=user_prompt)``
(见上面的入口章节)。每个工厂返回一个有序的 ``dict[str, PromptNode]`` (带标题的
分节), 由 ``PromptBundle.render`` 组装并填充。一份 prompt 服务所有 planner
(API loop、Claude Code、Codex): 用工具的裸名引用 (``move_to``, ...), 并只需说明
一次 Claude Code / Codex SDK 会把它们命名空间化为 ``mcp__rpent__<name>`` ——
不要再维护 CLI/API 两份拷贝。

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

将 prompt 内容放在 env 包内, 例如 ``robots/myenv/prompts/system.py`` 和
``user.py``。分节内容是普通字符串 (或 ``BulletList`` / ``Numbered``), 占位符
``{{suite}}`` / ``{{task}}`` / ``{{seed}}`` / ``{{output_dir}}`` /
``{{recipe_tag}}`` 在渲染时填充。

3. ``toolkit.py``
------------------

这个模块持有 LLM 能调用的一切: 工具 schema、primitive driver、每步状态 dump 以及
MCP allowlist。(LIBERO 中由于历史原因把这些拆到了 ``tools.py`` 和 ``toolkit.py``
两个文件; 新增 env 时全部放在 ``toolkit.py`` 里没问题。)

一个 toolkit 模块通常包含四部分:

**Primitive driver 类** (例如 ``MyEnvPrimitives``) —— toolkit 持有的 Python 对象。
它保存 ``EnvClient``、VLA ``model`` 客户端和任何 per-run 状态; 每个 primitive 工具
(``move_to``、``pi0_pick``、``release``、...) 对应一个方法, 返回一个 ``dict``
形式的日志。

**工具 schema + handler 辅助函数** —— 模块级的 ``TOOLS_SPEC`` 列表
（符合 Anthropic API 格式的 schema 字典，含 ``name``、``description``、
``input_schema``），
以及 toolkit 引用的自由函数 (例如 ``view_driver_state``、``back_project``、
``finish``)。

**每步状态 dump** —— ``dump_state(driver, output_dir, step_idx, log)`` 把 agent
之后会通过 ``view_*`` 工具读回的所有状态 (图像、深度、JSON 状态、camera meta)
序列化到 ``output_dir``。

**Toolkit 类** —— 继承 ``rpent.tools.toolkit.Toolkit``:

- 在 ``__init__`` 中通过 ``init_driver_clean`` 构建 primitive driver (清理过期的
  ``images/`` 等, 构造 primitives, dump 第 0 步),
- 用 ``self.add_tool(name, spec, handler)`` 注册每个工具 —— 无状态读取类
  (``view_driver_state``、``finish`` 等) 直接绑定到模块级函数; primitive 工具走
  ``_step(name, **kwargs)``, 它通过 ``getattr(self._driver, name)(**kwargs)``
  调用 driver 方法并重新渲染状态,
- 重写 ``close()``，将 Agent 侧生成的文件写入磁盘（例如 LIBERO toolkit
  在这里保存 agentview MP4）。

``primitives_kwargs`` (由 ``__init__.py:get_toolkit`` 转发进来) 是 toolkit 原样传给
primitive driver ``__init__`` 的 dict —— 通常是
``{"env": MyEnvClient(...), "model": VLAClient(...), ...}``。

值得遵循的约定
--------------

- ``output_dir`` 是 per-run 的临时目录, 由 runner 创建; 所有工件 (images、
  depths、``states.json``、transcripts、``episode.mp4``) 都写在里面。
- 工具 schema 使用 Anthropic API 格式（``name`` / ``description`` /
  ``input_schema``）。
  每个用 ``self.add_tool(...)`` 注册的工具都会暴露给所有 planner。
- Driver 侧的返回值必须可 pickle, 且不含 torch。
- 每个 primitive 工具执行后要 dump 一次新的状态快照, 这样下一次
  ``view_driver_state`` 看到的是动作后的世界。
- ``dump_state`` 是 Agent 获取环境状态的唯一数据来源；任何新的模态
  （例如触觉、力）都通过它提供。

4. ``_add_cli_args`` + ``_parse_config`` (runner 钩子)
------------------------------------------------------

``rpent/cli/main.py`` 是 env-agnostic 的。env CLI 处理拆成两个钩子, 共享
一次 argparse pass:

**``_add_cli_args(parser, use_dashboard) -> None``。** 把 env 的 flag 注册
到 main.py 已经持有的共享 parser 上。``use_dashboard`` 控制原本必填的 flag
是否保持可选 —— dashboard launcher 之后会填。main.py 在
``parser.parse_args()`` 之前调用, 所以只有一次 argparse pass, 它的 usage /
error 输出已经覆盖 env flag。

**``_parse_config(args) -> RunConfig``。** 在 ``parser.parse_args()`` 和
(如果适用) dashboard launcher 之后调用。强制 dashboard-only 可选字段
现在已经填好, 返回一个 :class:`~rpent.envs.RunConfig`:

- ``recipe_tag`` —— env 的 per-run 标签, 用于 transcript 文件名 / recipe path
  (LIBERO: ``f"{suite.replace('libero_', '')}_t{task}_s{seed}"``)。
- ``output_dir`` —— per-run scratch 目录路径 (main.py 之后调 ``init_output_dir``
  做 mkdir + 装 logging)。
- ``prompt_vars`` —— 喂给 ``PromptBundle.render`` 的 dict (通常包含 run 标识
  加上 prompt 引用的其它变量)。
- ``dashboard_state`` —— ``args.dashboard`` 为真时是一个
  :class:`~rpent.dashboard.state.State`, 否则 ``None``。
- ``task_desc`` —— env 特有的任务标识 dict, 会被原样写进 transcript JSON 的
  record (LIBERO: ``{"suite": ..., "task": ..., "seed": ...}``)。

.. code-block:: python

   def _add_cli_args(parser, use_dashboard) -> None:
       required = not use_dashboard
       parser.add_argument("--suite", default=None, required=required)
       parser.add_argument("--task", type=int, default=None, required=required)
       # ... 其它 env 特定 flag ...

   def _parse_config(args) -> RunConfig:
       if not args.suite: raise ValueError("--suite is required")
       # ... 派生 recipe_tag、output_dir、prompt_vars、dashboard_state ...
       return RunConfig(
           recipe_tag=recipe_tag,
           output_dir=output_dir,
           prompt_vars=prompt_vars,
           dashboard_state=dashboard_state,
           task_desc={"suite": args.suite, "task": args.task, "seed": args.seed},
       )

5. ``_init_runtime`` (runner 钩子)
----------------------------------

``parse_config`` 之后, main.py 调用 ``env_spec.init_runtime(args, output_dir)``
把 env / VLA 进程拉起来, 并构造 toolkit 需要的 kwargs。env 自己决定要 spawn
几个子进程 —— LIBERO 起一个 ``env_server`` + 一个 ``vla_server`` —— 只要
最终返回 ``(daemons, primitives_kwargs)``:

- ``daemons: list[ProcessDaemon]`` —— 本次 run 拥有的子进程; main.py 在
  ``finally`` 里逐个 ``.stop()``。
- ``primitives_kwargs: dict`` —— 原样传给 toolkit 构造器, 后者再传给
  primitive driver 的 ``__init__``。通常是
  ``{"env": MyEnvClient(...), "model": VLAClient(...)}``。

Endpoint 解析 (``--env-endpoint``、``--vla-endpoint``) 和子进程环境组装
(``CUDA_VISIBLE_DEVICES``、``MUJOCO_GL`` 等) 都在这里 —— main.py 完全
不知道这些细节。参考实现见 ``robots/libero/__init__.py``。

冒烟测试
--------

代码可以正常编译后，运行以下最小冒烟测试：

.. code-block:: bash

   PI05_CHECKPOINT_PATH=<path> ANTHROPIC_API_KEY=<key> \
     rpent --env myenv --suite <suite> --task <id> --seed 0 \
     --output-dir /tmp/myenv_smoke --planner api --model anthropic:claude-opus-4-8

预期结果是 Agent 完成 prompt 指定的任务并调用 ``finish``。运行结束后，
可在 ``<output_dir>/transcript_*.json`` 中查看总结。
