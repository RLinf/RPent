# Adding a new environment

This guide walks through what you need to write to plug a new physical /
simulated environment into PhysicalAgent's LLM-in-the-loop runner. Use
`physical_agent/envs/libero/` as the worked reference.

PhysicalAgent splits an env into two processes:

- **Agent side** (`physical_agent/envs/<env>/`) — runs in the agent process;
  contributes the tool schemas, primitive-driver logic, and prompts.
- **Driver side** (`deployment/<backend>/env_server.py`) — owns the
  heavyweight simulator / robot; exposes its env over a pickle-framed TCP
  RPC server (`physical_agent.rpc_driver.socket.SocketRpcServer`).

The two are connected by an `EnvClient` class that turns each agent-side
method call into one RPC against the driver.

## Entry point

For a new env `myenv`, the file layout is:

```
physical_agent/envs/myenv/
    __init__.py            # entry point — get_env_spec() / get_toolkit() factories
    myenv_env_client.py    # MyEnvClient — agent-side RPC stub (§1)
    prompt_bundle.py       # PROMPTS = PromptBundle(...)              (§2)
    toolkit.py             # MyEnvToolkit + primitives + tool schemas (§3)

deployment/<backend>/env_server.py    # driver-side facade + RPC server (§1)
```

`__init__.py` is the package's entry point. The registry in
`physical_agent/envs/base.py` lazily imports `physical_agent.envs.<name>`
on demand and calls its two factories:

```python
# physical_agent/envs/myenv/__init__.py
from physical_agent.envs.env_spec import EnvSpec
from physical_agent.envs.myenv.prompt_bundle import PROMPTS

def get_env_spec() -> EnvSpec:
    return EnvSpec(name="myenv", prompts=PROMPTS)

def get_toolkit(*, primitives_kwargs: dict[str, Any], video_path: str | None = None):
    from physical_agent.envs.myenv.toolkit import MyEnvToolkit
    return MyEnvToolkit(primitives_kwargs=primitives_kwargs, video_path=video_path)
```

That's the entire registration step — `_resolve_env(name)` does an
`importlib.import_module(f"physical_agent.envs.{name}")`, so dropping the
package on disk is enough. No central list to update.

The three sections below describe what each of the three referenced
modules must contain.

---

## 1. `myenv_env_client.py` + `deployment/<backend>/env_server.py`

These two files form the agent ↔ driver bridge. The client lives in the
agent process and turns method calls into RPCs; the env_server lives in the
driver process and answers them.

### 1.1 Env client (agent side)

The base contract is two gym-style methods (`reset`, `step`); 
add whatever your env needs on top (LIBERO has `chunk_step`, `render_agentview`,
`get_camera_meta`, `cached_image`, …). Each method forwards through
`RpcClient.call("<rpc-name>", args=..., kwargs=...)` with a per-method
timeout. Keep names stable — the driver-side dispatcher matches by name.

```python
class MyEnvClient:
    def __init__(self, client: RpcClient, *, return_all_frames: bool = False):
        self._client = client
        self.return_all_frames = return_all_frames

    def reset(self):
        return self._client.call("env.reset", timeout_s=120.0)

    def step(self, action):
        return self._client.call("env.step", args=(action,), timeout_s=60.0)
    # ... add other env-specific methods
```

### 1.2 Env server (driver side)

Mirror the client's API in a facade class on the driver side
(e.g. `MyEnvFacade`). Methods take the same positional / keyword arguments
the client sends and return pickleable values (numpy, not torch — the agent
side does not import torch).

Wrap the facade in a dispatcher and serve over `SocketRpcServer`:

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

The `transport_ready` event on stdout is required — `cli.main.start_env_server`
blocks until it sees it.

`cli/main.py` currently imports `LiberoEnvClient` and the LIBERO env_server
script path directly. Adding a new env means either branching on
`args.env_name` to pick the client class + driver script, or factoring those
two callsites out behind a per-env helper.

---

## 2. `prompt_bundle.py`

Export a single module-level `PROMPTS = PromptBundle(...)` instance with all
seven fields populated. The bundle carries the LLM-facing strings the runner
renders before the loop starts:

```python
PROMPTS = PromptBundle(
    system_prompt=SYSTEM_PROMPT,
    initial_user_template=INITIAL_USER_TEMPLATE,
    perception_prefix=PERCEPTION_PREFIX,
    perception_user_template=PERCEPTION_USER_TEMPLATE,
    claude_code_prompt_template=CLAUDE_CODE_PROMPT_TEMPLATE,
    claude_code_perception_prompt_template=CLAUDE_CODE_PERCEPTION_PROMPT_TEMPLATE,
    format_claude_code_prompt=format_claude_code_prompt,
)
```

Either reuse the shared strings in `physical_agent.context.prompt_base`, or
write your own — they're plain `str.format`-style templates that take
`suite` / `task` / `seed` / `output_dir` / `recipe_tag`. The bundle is
referenced from the env's `__init__.py` (see entry point above), and
`EnvSpec.prompts` carries it to the cerebrum.

---

## 3. `toolkit.py`

This module owns everything the LLM can call: the tool schemas, the
primitive driver, the per-step state dump, and the MCP allowlist. (In the
LIBERO env these are split between `tools.py` and `toolkit.py` for
historical reasons; for a new env it is fine to keep them all in
`toolkit.py`.)

A toolkit module typically contains four pieces:

**Primitive driver class** (e.g. `MyEnvPrimitives`) — a Python object the
toolkit owns. It holds the `EnvClient`, the VLA `model` client, and any
per-run state. It exposes one method per primitive tool (`move_to`,
`pi0_pick`, `release`, …) returning a `dict` log.

**Tool schemas + handler helpers** — a module-level `TOOLS_SPEC` list of
Anthropic-shaped schema dicts (`name`, `description`, `input_schema`), plus
any free functions referenced by the toolkit (e.g. `view_driver_state`,
`back_project`, `finish`).

**Per-step state dump** — `dump_state(driver, output_dir, step_idx, log)`
serializes whatever state the agent will read back via the `view_*` tools
(images, depths, JSON state, camera meta) into `output_dir`.

**Toolkit class** — subclass `physical_agent.tools.toolkit.Toolkit`:

- declare `allowed_mcp_tool_names` (the namespaced `mcp__physical_agent__*`
  list, used by Claude Code / MCP-style cerebrums),
- build the primitive driver in `__init__` via `init_driver_clean`
  (wipes stale `images/` etc., constructs the primitives, dumps step 0),
- register each tool with `self.add_tool(name, spec, handler)` — stateless
  readers (`view_driver_state`, `finish`, …) bind directly to module-level
  functions; primitive tools route through `_step(name, **kwargs)` which
  calls `getattr(self._driver, name)(**kwargs)` and re-renders state,
- override `close()` to flush any agent-side artifacts (e.g. the LIBERO
  toolkit saves the agentview MP4 there).

`primitives_kwargs` (forwarded from `__init__.py:get_toolkit`) is the dict
the toolkit passes verbatim to your primitive driver's `__init__` —
typically `{"env": MyEnvClient(...), "model": VLAClient(...), ...}`.

---

## Conventions worth keeping

- `output_dir` is the per-run scratch directory and is created by the runner;
  every artifact (images, depths, `states.json`, transcripts, `episode.mp4`)
  goes there.
- Tool schemas are Anthropic-shaped (`name` / `description` / `input_schema`)
  and the toolkit prepends `mcp__physical_agent__` for the MCP allowlist.
- Driver-side return values must be picklable and torch-free.
- Each primitive tool dumps a fresh state snapshot after running so the next
  `view_driver_state` call reflects the post-action world.
- Treat `dump_state` as the source of truth for what the agent sees — any
  new modality (e.g. tactile, force) goes through it.

## Smoke test

Once everything compiles, the minimal smoke loop is:

```
PI05_CHECKPOINT_PATH=<path> ANTHROPIC_API_KEY=<key> \
  python -m cli.main --env myenv --suite <suite> --task <id> --seed 0 \
  --output_dir /tmp/myenv_smoke --model claude-opus-4-7 --cerebrum anthropic
```

Expect the driver to emit `transport_ready`, the agent to complete the
prompted task, and `finish` to be invoked. Check
`<output_dir>/transcript_*.json` for the post-run summary.
