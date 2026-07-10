# 新增 environment 指南

本指南说明如何把一个新的物理 / 仿真环境接入 RPent 的 LLM-in-the-loop
runner。请把 `robots/libero/` 当作完整参考实例。

RPent 把一个 env 拆成两个进程:

- **Agent 侧** (`rpent/envs/<env>/`) — 跑在 agent 进程内, 提供工具
  schema、primitive driver 逻辑和 prompt。
- **Driver 侧** (`robots/<env>/env_server.py`) — 持有重量级的仿真器 /
  机器人; 通过 pickle-framed TCP RPC server
  (`rpent.rpc_driver.socket.SocketRpcServer`) 对外暴露 env。

两侧通过一个 `EnvClient` 类相连: 每个 agent 侧方法调用对应一次到 driver 的 RPC。

### VLA 模型跑在自己独立的进程里(env / vla 分离)

当一个 env 使用 VLA 策略(读取相机观测、输出动作的学习模型)时, 该模型跑在
**第三个独立进程**里 —— 绝不塞进 env_server:

- **VLA 侧** (`robots/<env>/vla_server.py`) — 只持有 VLA 策略（GPU
  模型), 通过自己的 RPC/HTTP 端点暴露 `vla_load` / `vla_infer` / `vla_reset`,
  不 import 任何仿真器。
- toolkit 除了 `EnvClient` 之外, 还接收一个 **model client**(LIBERO/Pi0.5 用
  `VLAClient`, RoboCasa/RLDX-1 用 `RLDXVLAClient`)作为 `model` 参数。两个
  client 指向两个不同的 server 进程。

**为什么这个分离是强制的(而非可选):** 模型(大 GPU 权重、自己的 CUDA 上下文、
`transformers`/`openpi` 等重依赖)和仿真器(MuJoCo/robosuite、绑定主线程的 EGL
渲染)在进程层面的需求相互冲突。把它们放进同一进程会耦合生命周期、逼一个解释器同时
满足两套依赖树, 且模型 OOM 会连带拖垮仿真。分开后, 任一侧都能独立重启、扩容或指向
远程主机(`--vla_endpoint host:port` 可复用已在运行的模型 server)。每个 env 都
**必须**遵守: env_server 持有仿真, vla_server 持有模型。

**传输协议可因 env 而异, 但架构不可变。** LIBERO 的 `vla_server.py` 走 HTTP
`/predict`(扁平的 image+state 载荷); RoboCasa 的 `vla_server.py` 走和它
env_server 相同的 pickle-framed socket RPC —— 因为 RLDX 观测是历史堆叠的嵌套
numpy dict(3 路相机 video 张量 `(1,T,H,W,3)` + `state.*` + annotation +
session/reset_memory), 用 socket 天然承载, 走 HTTP 则要额外设计 wire 格式。按
观测形态选编解码, 但保持 env/vla 进程分离一致。

**任何需要仿真 env 对象的逻辑都留在 env_server。** 对 RoboCasa, `check_grasp`
和 `assemble_action`(eval 的 `unmap_action` + composite-controller
split-index 组装)需要活的 robosuite env, 因此是 env_server 的 RPC —— **不**属于
VLA server。于是 agent 侧的 skill(`RLDXSkill`)同时持有两个 client: env client
做 render/step/grasp/assemble, model client 做推理。

## 入口

新增名为 `myenv` 的 env 时, 文件布局如下:

```
rpent/envs/myenv/
    __init__.py            # 入口 — get_env_spec() / get_toolkit() 工厂
    myenv_env_client.py    # MyEnvClient — agent 侧 RPC 代理 (§1)
    prompt_bundle.py       # system()/user() prompt 工厂              (§2)
    toolkit.py             # MyEnvToolkit + primitives + tool schemas (§3)

robots/<env>/env_server.py    # driver 侧 facade + RPC server (§1)
```

`__init__.py` 是这个包的入口。`rpent/envs/base.py` 中的注册表会按需
lazily import `rpent.envs.<name>`, 并调用其两个工厂函数:

```python
# rpent/envs/myenv/__init__.py
from rpent.envs.env_spec import EnvSpec
from rpent.envs.prompt_bundle import PromptBundle
from rpent.envs.myenv.prompt_bundle import system_prompt, user_prompt

def get_env_spec() -> EnvSpec:
    return EnvSpec(name="myenv", prompts=PromptBundle(system=system_prompt, user=user_prompt))

def get_toolkit(*, primitives_kwargs: dict[str, Any], video_path: str | None = None):
    from rpent.envs.myenv.toolkit import MyEnvToolkit
    return MyEnvToolkit(primitives_kwargs=primitives_kwargs, video_path=video_path)
```

整个注册流程就是这样 — `_resolve_env(name)` 通过
`importlib.import_module(f"rpent.envs.{name}")` 动态加载, 所以
把包放在磁盘上就够了, 没有中央列表需要维护。

下面三章分别说明上面引用的三个模块各自需要写什么。

---

## 1. `myenv_env_client.py` + `robots/<env>/env_server.py`

这两个文件构成 agent ↔ driver 的桥梁: client 跑在 agent 进程内, 把方法调用转成
RPC; env_server 跑在 driver 进程内, 应答这些调用。

### 1.1 Env client (agent 侧)

类约定了两个 gym 风格的方法 (`reset`、`step`); 根据 env 需要增加其他方法 (LIBERO 增加了
`chunk_step`、`render_agentview`、`get_camera_meta`、`cached_image` 等)。每个方法通过
`RpcClient.call("<rpc-name>", args=..., kwargs=...)` 转发, 并设置各自的 timeout。
方法名要稳定 — driver 侧 dispatcher 按名字匹配。

```python
class MyEnvClient:
    def __init__(self, client: RpcClient, *, return_all_frames: bool = False):
        self._client = client
        self.return_all_frames = return_all_frames

    def reset(self):
        return self._client.call("env.reset", timeout_s=120.0)

    def step(self, action):
        return self._client.call("env.step", args=(action,), timeout_s=60.0)
    # ... 根据 env 需要添加其他方法
```

### 1.2 Env server (driver 侧)

在 driver 侧用 facade 类 (例如 `MyEnvFacade`) 镜像 client 的 API。方法接收与 client
发送方一致的位置 / 关键字参数, 返回可 pickle 的值 (numpy, 不要 torch — agent
侧不 import torch)。

把 facade 包在 dispatcher 中, 用 `SocketRpcServer` 提供服务:

```python
def dispatch(method, args, kwargs):
    if method.startswith("env."):
        return getattr(facade, method[len("env."):])(*args, **kwargs)
    if method == "shutdown":
        shutdown_event.set()
        return {"ok": True}
    raise ValueError(f"unknown RPC method: {method!r}")

server = SocketRpcServer((host, port), dispatch)
print(json.dumps({"event": "transport_ready", "kind": "socket",
                  "host": host, "port": bound_port}), flush=True)
```

stdout 上的 `transport_ready` 事件是必须的 — `cli.main.start_env_server`
会阻塞直到看到它。

当前的 `cli/main.py` 直接 import 了 `LiberoEnvClient` 和 LIBERO 的 env_server
脚本路径。新增 env 时, 要么在 `args.env_name` 上分支选择 client 类和 driver
脚本, 要么把这两处调用点抽到每个 env 的小型 helper 后面。

---

## 2. `prompt_bundle.py`

定义两个 prompt 工厂 — `system_prompt()` 和 `user_prompt()` —
并在 env 的 `__init__.py` 中构造 `PromptBundle(system=system_prompt, user=user_prompt)`
(见上面的入口章节)。每个工厂返回一个有序的 `dict[str, PromptNode]` (带标题的
分节), 由 `PromptBundle.render` 组装并填充。一份 prompt 服务所有 cerebrum
(API loop、Claude Code、Codex): 用工具的裸名引用 (`move_to`, ...), 并只需说明
一次 Claude Code / Codex SDK 会把它们命名空间化为
`mcp__rpent__<name>` — 不要再维护 CLI/API 两份拷贝。

```python
# rpent/envs/myenv/prompt_bundle.py
from rpent.context.prompt_utils import PromptNode
from rpent.context.prompts import prompt as base_prompt
from rpent.envs.myenv import prompts as myenv_prompt

def system_prompt() -> dict[str, PromptNode]:
    return {
        "Intro": myenv_prompt.PREAMBLE,
        "Goal": myenv_prompt.GOAL,
        "Rules": myenv_prompt.RULES,
        "Workflow": myenv_prompt.WORKFLOW,
        "Environment": myenv_prompt.ENVIRONMENT,
        "Output": base_prompt.OUTPUT,
    }

def user_prompt() -> dict[str, PromptNode]:
    return dict(base_prompt.USER)
```

可以复用 `rpent.context.prompts.prompt` 中的共享分节 (`OUTPUT`、
`USER`), 也可以自己写。分节内容是普通字符串 (或 `BulletList` / `Numbered`),
占位符 `{{suite}}` / `{{task}}` / `{{seed}}` / `{{output_dir}}` /
`{{recipe_tag}}` 在渲染时填充。

---

## 3. `toolkit.py`

这个模块持有 LLM 能调用的一切: 工具 schema、primitive driver、每步状态 dump
以及 MCP allowlist。(LIBERO 中由于历史原因把这些拆到了 `tools.py` 和
`toolkit.py` 两个文件; 新增 env 时全部放在 `toolkit.py` 里没问题。)

一个 toolkit 模块通常包含四部分:

**Primitive driver 类** (例如 `MyEnvPrimitives`) — toolkit 持有的 Python 对象。
它保存 `EnvClient`、VLA `model` 客户端和任何 per-run 状态; 每个 primitive 工具
(`move_to`、`pi0_pick`、`release`、...) 对应一个方法, 返回一个 `dict` 形式
的日志。

**工具 schema + handler 辅助函数** — 模块级的 `TOOLS_SPEC` 列表 (Anthropic 形状
的 schema dict, 含 `name`、`description`、`input_schema`), 以及 toolkit 引用
的自由函数 (例如 `view_driver_state`、`back_project`、`finish`)。

**每步状态 dump** — `dump_state(driver, output_dir, step_idx, log)` 把 agent
之后会通过 `view_*` 工具读回的所有状态 (图像、深度、JSON 状态、camera meta)
序列化到 `output_dir`。

**Toolkit 类** — 继承 `rpent.tools.toolkit.Toolkit`:

- 在 `__init__` 中通过 `init_driver_clean` 构建 primitive driver (清理过期的
  `images/` 等, 构造 primitives, dump 第 0 步),
- 用 `self.add_tool(name, spec, handler)` 注册每个工具 — 无状态读取类
  (`view_driver_state`、`finish` 等) 直接绑定到模块级函数; primitive 工具走
  `_step(name, **kwargs)`, 它通过 `getattr(self._driver, name)(**kwargs)`
  调用 driver 方法并重新渲染状态,
- override `close()` 来 flush agent 侧的工件 (例如 LIBERO toolkit 在这里
  保存 agentview MP4)。

`primitives_kwargs` (由 `__init__.py:get_toolkit` 转发进来) 是 toolkit
原样传给 primitive driver `__init__` 的 dict — 通常是
`{"env": MyEnvClient(...), "model": VLAClient(...), ...}`。

---

## 值得遵循的约定

- `output_dir` 是 per-run 的临时目录, 由 runner 创建; 所有工件 (images、
  depths、`states.json`、transcripts、`episode.mp4`) 都写在里面。
- 工具 schema 是 Anthropic 形状 (`name` / `description` / `input_schema`)。
  每个用 `self.add_tool(...)` 注册的工具都会暴露给所有 cerebrum。
- Driver 侧的返回值必须可 pickle, 且不含 torch。
- 每个 primitive 工具执行后要 dump 一次新的状态快照, 这样下一次
  `view_driver_state` 看到的是动作后的世界。
- 把 `dump_state` 当作 agent 视角的 "事实源" — 任何新的模态 (例如触觉、力)
  都从它走。

## 冒烟测试

代码可以编译之后, 最小的冒烟回路如下:

```
PI05_CHECKPOINT_PATH=<path> ANTHROPIC_API_KEY=<key> \
  python -m cli.main --env myenv --suite <suite> --task <id> --seed 0 \
  --output-dir /tmp/myenv_smoke --cerebrum api --model anthropic:claude-opus-4-8
```

期望: driver 输出 `transport_ready`, agent 完成 prompt 的任务, 并调用 `finish`。
查看 `<output_dir>/transcript_*.json` 获取运行结束的总结。
