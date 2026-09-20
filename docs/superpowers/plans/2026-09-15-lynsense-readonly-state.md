# Lynsense Read-Only State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本计划默认主 agent 实施，仅委派独立只读评审，不自动提交。

**Goal:** 在 `robots/lynsense/` 增加只读右臂状态后端，离线验证 RPent 到 ROS 适配器的完整调用和清理链路。

**Architecture:** 沿用 RobotSpec、Toolkit；不引入 py_trees、RPC、运动原语或技能注册表。运行时返回未连接 adapter，Toolkit 获得所有权后启动两个只读订阅；状态转换和新鲜度判断单独进行纯 Python 测试。

共享代码仅补充可选的 API 图像读取器：默认开启保持原有后端行为，Lynsense 关闭，避免额外 read_image 工具和无意义的 EnvState 依赖。

**Tech Stack:** RPent 现有依赖、Python 标准库、pytest；实机订阅使用 ROS 2 Humble 的 rclpy、sensor_msgs、xarm_msgs，由获准的机器人环境提供。

**Spec:** [只读状态设计](../../architecture/specs/2026-09-15-lynsense-readonly-state-design.md)

## Global Constraints

- 第一轮只验证连接、状态快照和错误反馈，不发送任何运动、夹爪、使能或状态切换命令。
- 本轮执行门禁是本地假 ROS 测试与源码审查；不调用真实模型，不连接机器人。
- 不在机器人上安装依赖、修改文件或启动额外 ROS 2 进程。用户另行批准了项目本地 `.venv` 和基础/测试依赖；不修改 `pyproject.toml` 或全局环境。
- 不使用 py_trees，不创建运动工具、`skills.py`、感知模块或空的扩展接口。
- ROS 2 Humble，期望 `ROS_DOMAIN_ID=3`，只校验进程环境，不自动改写。
- 目标订阅：`/right_xarm/robot_states` / `xarm_msgs/msg/RobotMsg`；`/right_xarm/joint_states` / `sensor_msgs/msg/JointState`。
- 等待超时 5 秒、新鲜度阈值 2 秒、关闭预算 2 秒，均为正有限配置值；只读不是“可安全运动”的证明。
- 保留 `is_real_robot=True` 的现有 TTY 限制。第一轮只支持 `--planner api --memory-profile local`；无 Dashboard、interactive、exploration、task-card 或模型 shell 执行。
- 不修改 `pyproject.toml`、共享 runner、机器人权限或 SSH 配置，不借用同级项目环境。

## 基线与前提

2026-09-15：工作区根不是 Git 仓库，但 `RPent/` 是，HEAD 为 `43f32aa08cba07bd4d49a4bfa5eba4ef633e9b92`。计划开始前已有本任务创建的未跟踪 `docs/architecture/`；不能漏审这些文件。`robots/lynsense/` 尚不存在。

当前 `/usr/bin/python3` 为 3.14.4，不符合 RPent `>=3.10,<3.13`，且没有 pytest、numpy、pydantic_ai 或 rclpy。执行前先检查项目已获准环境；以下命令约定从 `RPent/` 执行，使用 `.venv/bin/python`。这是一项待满足的环境前提，不是已有环境声明或创建/安装授权。缺失时停止代码测试阶段，报告缺口并请求单独的环境配置授权，不输出虚假的 red/green 证据。

机器人侧 `zxh` 无法读取 `/home/rpp/rpp_ws`，消息类型只从 ROS 图查询到，尚无 `xarm_msgs` 导入或实际样本证据。本轮不试图修权限、复制工作空间或远程安装。

实施记录（2026-09-15）：经用户批准，使用现有 uv 建立 Python 3.10.20 的 `.venv` 并安装 `.[test]`，本地环境前提已满足。分支为 `feat/lynsense-readonly-state`，未提交或推送。Task 1/2 的缺模块 red 已确认，初次 green 合计 75 项。Task 3 的入口缺失和图像工具 red 已确认；修正测试自身的日志目录/假模型工厂配置后，完整门禁为 146 项。补充单一 adapter 所有权、中断及线程启动失败回归后，完整门禁为 151 项，退出码 0。独立代码评审待完成。

评审修复记录：首轮独立评审发现 3 个 Important：executor 故障未自动清理、复制参数字典能共享 adapter、默认 context shutdown 失败后无法重试。三项均补了实际失败的测试并修复：adapter 层原子认领；spin 异常后自动清理并保留故障；显式关闭自己持有的 context，成功后才清除匹配的默认槽位。新增自动清理和外部关闭竞争、清理失败重试、替代 context 保护及新实例重连测试。扩展门禁（原门禁加 `tests/unit_tests/rpent/tools/test_toolkit_contracts.py`）175 passed，exit 0；等待同一 reviewer 复审。`uv pip check` 检查 80 包通过；`rpent --robot lynsense --help` 通过；ruff/pre-commit 未安装，未运行。

最终复审（2026-09-15）：同一独立 reviewer 给出 Yes，关闭全部 3 项 Important，无新增阻塞项。Reviewer 独立运行状态/适配器测试 81 passed，并重复验证所有权 50 次、关闭竞争 25 次，无遗留 worker；主 agent 另在独立进程重复关闭竞争 20 次全部通过。上述仅为离线证据。代码保留在本地未提交分支，不合并、不推送、不部署。默认 `prompt_vars` 为空；指定 `--memory-dir` 时沿用现有工厂约定传递本地路径，固定提示词不依赖它。

## 文件范围

| 文件（相对 RPent） | 责任 |
| --- | --- |
| `robots/lynsense/__init__.py` | 包入口，最终导出两个工厂 |
| `robots/lynsense/state.py` | JointState 校验、两条流的缓存及新鲜度 |
| `robots/lynsense/ros2_adapter.py` | 惰性 ROS 导入、连接、回调线程、关闭 |
| `robots/lynsense/toolkit.py` | 唯一的 read_robot_state 工具 |
| `robots/lynsense/robot_spec.py` | RobotSpec、提示词、参数与工厂 |
| `robots/lynsense/README.md` | 只读用法、环境要求、验证限制 |
| `tests/unit_tests/robots/lynsense/test_state.py` | 纯数据测试 |
| `tests/unit_tests/robots/lynsense/test_ros2_adapter.py` | 假 ROS 生命周期及禁止控制测试 |
| `tests/unit_tests/robots/lynsense/test_extension.py` | 工厂、注册表、Toolkit、假 Planner 集成 |
| `tests/unit_tests/rpent/robots/test_registry_contracts.py` | 增加机器人名称与提示词样例，保留其他断言 |
| `rpent/tools/toolkit.py` | 仅增加 include_image_reader 默认能力声明 |
| `rpent/planner/api_loop.py` | 条件构造额外的 read_image 工具 |
| `tests/unit_tests/rpent/planner/test_api_contracts.py` | 原有默认行为及禁用图像后的最终工具表回归 |

## Task 1: 状态快照

**Files:** 新增 `state.py`、包 `__init__.py` 和 `test_state.py`；此时包入口不导入未实现的工厂。

**Interfaces:** `StateCache(stale_after_s: float)`；`update_joints(message, received_mono: float, observed_at: str) -> None`；`update_driver(received_mono: float) -> None`；`snapshot(now_mono: float) -> dict`。message 按属性读取，仅使用标准 JointState 的 `name/position/velocity/effort`；不导入 ROS。adapter 负责锁，测试直接用简单对象。

- [x] **1. 写失败测试。** 首条 joint 和 driver 均收到且新鲜才返回 `ok`；测试时间由调用方传入，不 sleep。

```python
from types import SimpleNamespace
from robots.lynsense.state import StateCache

def test_requires_both_streams_and_expires_each():
    cache = StateCache(stale_after_s=2.0)
    msg = SimpleNamespace(name=['j1'], position=[0.1], velocity=[], effort=[])
    cache.update_joints(msg, 10.0, '2026-09-15T10:00:00+00:00')
    assert cache.snapshot(10.0)['status'] == 'unavailable'
    cache.update_driver(10.0)
    assert cache.snapshot(10.5)['positions'] == [0.1]
    assert cache.snapshot(10.5)['status'] == 'ok'
    cache.update_joints(msg, 12.1, '2026-09-15T10:00:02+00:00')
    assert cache.snapshot(12.1)['status'] == 'stale'
```

- [x] **2. 运行 red。** `.venv/bin/python -m pytest tests/unit_tests/robots/lynsense/test_state.py -q`；预期新增模块/符号缺失，不把环境缺 pytest 当成产品失败。
- [x] **3. 实现最少缓存逻辑。** 内部保存最近合法 joint、两条独立接收时间、最新 joint 校验错误。每次更新复制数组；joint 无效时立刻标记 invalid，不能继续输出旧值为 ok；合法更新允许恢复。可选数组为空则省略，长度错误或非有限则省略并返回 warnings。

```python
import math

def valid_joint_fields(names, positions):
    return (
        bool(names) and len(names) == len(positions)
        and all(isinstance(n, str) and bool(n.strip()) for n in names)
        and len(set(names)) == len(names)
        and all(not isinstance(v, bool) and math.isfinite(float(v)) for v in positions)
    )
```

转换时捕获缺少属性、TypeError、ValueError、OverflowError，形成 invalid 的 reason；不让坏消息静默保留原 ok。snapshot 优先级为 invalid、unavailable、stale、ok，分别在最新 joint 无效、任一流未收到、任一 age 超阈值时选取。非有限接收时间拒绝，时钟倒退报告 failed。age 使用接收 monotonic 差，observed_at 仅为 UTC 接收时间，不作数据源同步或健康证明。`ok` 时返回设计里的所有字段；无有效数据时使用 null，旧合法数据可保留但必须附非 ok 状态。快照深拷贝，调用者修改不会污染缓存。

- [x] **4. 补参数化用例并运行 green。** 覆盖空/重复/空白名称、长度不符、NaN/Inf、非数字、非法可选数组、两条流各自过期、无消息、坏消息后恢复、快照隔离；运行同一命令并记录实际计数。

## Task 2: 只读 ROS 适配器

**Files:** 新增 `ros2_adapter.py`、`test_ros2_adapter.py`；消费 Task 1 的 StateCache。

**Interfaces:** `Ros2StateAdapter(*, expected_domain_id=3, joint_topic='/right_xarm/joint_states', robot_topic='/right_xarm/robot_states', connect_timeout_s=5.0, stale_after_s=2.0)`；`connect() -> dict`（status 为 ok/failed）；`get_snapshot() -> dict`；`close() -> None`。构造只校验配置、创建本地缓存和 Condition，不能初始化 ROS。close 后对象不可重新连接；已连接时重复 connect 返回当前状态，不重复建线程。

- [x] **1. 写失败测试。** 使用 pytest monkeypatch 构造测试文件内的假 `rclpy`、Node、executor、sensor_msgs、xarm_msgs 模块。Node 只允许 create_subscription；测试任何机器人 publisher/client/Action 创建即失败。通过假 executor 队列投递两种消息，并记录 init/shutdown/销毁次数。

```python
def test_construction_is_inert_and_close_is_idempotent():
    from robots.lynsense.ros2_adapter import Ros2StateAdapter
    adapter = Ros2StateAdapter()
    assert adapter.get_snapshot()['status'] == 'unavailable'
    adapter.close()
    adapter.close()
    assert adapter.get_snapshot()['status'] == 'closed'
```

- [x] **2. 运行 red。** `.venv/bin/python -m pytest tests/unit_tests/robots/lynsense/test_ros2_adapter.py -q`。
- [x] **3. 实现 connect。** 懒加载 ROS 包，仅检查显式 ROS_DOMAIN_ID 与期望值相等；非法/未设置时不初始化。拒绝已初始化默认 context，且不 shutdown 外来 context。自身 init 成功后设置 ownership 标志；使用 `args=[]` 避免解析 RPent 参数。后续失败均调用自己的 close，再返回 failed/reason，工厂把失败升级为 RuntimeError。

```python
node = rclpy.create_node(
    'rpent_lynsense_readonly',
    enable_rosout=False,
    start_parameter_services=False,
    use_global_arguments=False,
)
```

仅订阅两个已指定 Topic，BEST_EFFORT / VOLATILE / KEEP_LAST(10)。收到相应类型的消息即证明端点发现和类型匹配，不用 ROS 图中的“名称存在”作为收到消息的替代。订阅 RobotMsg 只更新 driver 接收时间，不读取未知诊断字段。消息回调在 Condition 内更新缓存并 notify；转换不能进行 I/O。

- [x] **4. 实现有界回调和快照。** 独立 SingleThreadedExecutor 线程每次 `spin_once(timeout_sec=0.1)`；connect 在 Condition 上按 monotonic 截止时间等待 cache.snapshot 为 ok。超时说明哪条流缺失/过期；收到坏 joint 可等待窗口内后续合法值，否则失败。executor 异常保存 failure 并唤醒等待者；get_snapshot 不再 spin 或等待消息，只返回加锁副本，失败/closed 优先于缓存状态。订阅消失表现为相应流 stale，不谎称已诊断不存在或 QoS 错误。

```python
while not stop_event.is_set():
    executor.spin_once(timeout_sec=0.1)
```

- [x] **5. 实现关闭和错误测试。** 停止标志、Condition 通知、executor.wake、join（剩余 2 秒预算）、executor.shutdown、销毁 Node、仅 shutdown 自己拥有的 context。线程仍活着则报告 RuntimeError，不销毁正在使用的 Node，不谎报关闭；保留资源供下一次 close 重试，失败门禁不得通过。connect/close 竞态串行化，等待 connect 的关闭不能无限阻塞。测试首条消息缺失、坏 joint、独立 stale、executor 故障、init/Node/第二订阅创建失败、默认 context 外部占有、重复关闭、关闭超时和中途取消连接。
- [x] **6. 运行 green。** 运行 Task 1 和 Task 2 两个测试文件；所有 ROS 使用假模块，禁止连接 DDS/机器人。构造/import 在缺 ROS 的宿主必须成功，connect 给出明确的缺包诊断。

## Task 3: RPent 原生扩展和端到端离线验收

**Files:** 新增 `robot_spec.py`、`toolkit.py`、`README.md`、`test_extension.py`；更新 `__init__.py` 和现有 `test_registry_contracts.py`；仅在共享 `rpent/tools/toolkit.py`、`rpent/planner/api_loop.py`、`test_api_contracts.py` 加入图像读取器的兼容开关和测试。不修改其他机器人或共享 runner。

**Interfaces:** 包导出 `get_robot_spec() -> RobotSpec`、`get_toolkit(*, runtime_kwargs: dict, dashboard_events, config: RunConfig) -> LynsenseToolkit`；工厂消费 Task 2 adapter，Toolkit.close 转交其 close。

- [x] **1. 写失败测试。** 机器人枚举新增 lynsense；spec 导入不得加载 rclpy；验证只有一个只读工具、输入不接受额外参数、无默认文件工具、异常时无资源泄漏。

```python
def assert_readonly_toolkit(toolkit):
    from rpent.planner.api_loop import _build_tools
    specs = toolkit.get_tools_spec()
    assert [s['name'] for s in specs] == ['read_robot_state']
    assert specs[0]['input_schema']['additionalProperties'] is False
    assert 'error' in toolkit.execute_tool('set_position', {}).result
    assert 'error' in toolkit.execute_tool('read_robot_state', {'move': True}).result
    assert [tool.name for tool in _build_tools(toolkit)] == ['read_robot_state']
    assert [tool.name for tool in _build_tools(toolkit, no_images=True)] == ['read_robot_state']
```

- [x] **2. 运行 red。** `.venv/bin/python -m pytest tests/unit_tests/robots/lynsense/test_extension.py tests/unit_tests/rpent/robots/test_registry_contracts.py -q`。
- [x] **3. 实现 Toolkit。** 沿用父类初始化和执行锁，覆盖通用注册为空，handler 无参数。schema 是 `{"type":"object","properties":{},"additionalProperties":false}`；闭包或方法须正确保留 readonly 标记。工厂先建 Toolkit（仅本地注册），随后 connect，失败关闭并重新抛出；正常返回后由 runner 调用 close。

```python
class LynsenseToolkit(Toolkit):
    include_image_reader = False

    def _register_common_tools(self):
        pass

    @readonly
    def read_robot_state(self):
        return self._adapter.get_snapshot()

    def close(self):
        self._adapter.close()
```

基础 Toolkit 声明 `include_image_reader: ClassVar[bool] = True`；API `_build_tools` 改为以下前缀，随后原有 toolkit schema 转换循环保持原样。getattr 的默认 True 保留既有 duck-typed FakeToolkit 的行为；False 时不得访问 toolkit.state。回归测试保留既有 `['read_image', 'finish']` 断言，并新增关闭能力时 state 属性一旦访问就抛异常的用例。

```python
tools: list[Tool] = []
if getattr(toolkit, 'include_image_reader', True):
    image_reader = _make_image_reader(toolkit.state, no_images=no_images)
    tools.append(Tool(image_reader, name='read_image', sequential=True))
```

Lynsense 的 `get_env_state()` 只返回同一结构化快照，不新增图像 EnvState 或 StepRecord；只读 handler 不走自动抓图分支，API 图像入口已显式关闭。`solved()` 返回 False，未定义运动目标或实机任务成功条件；`write_recipe()` 保持父类 no-op。MemoryManager 只按已有构造方式引用 local 目录，不 sync、不向模型提供文件工具。API Planner 普通文本可结束本轮；不添加 finish，不把文本自述当成读取成功，离线验收以真实 handler 的 ok 快照为准。

- [x] **4. 实现 RobotSpec 与提示词。** `is_real_robot=True`、`supports_exploration=False`、不设置 finalize/replay。`add_cli_args` 仅增加 `--ros-domain-id`（期望）、`--joint-state-topic`、`--robot-state-topic`、`--state-timeout`、`--state-max-age`，默认值来自规格。`parse_config` 拒绝非 api 或非 local memory，早于共享 CLI 的 HF sync；校验参数，不做 ROS 导入/初始化。返回 RunConfig(recipe_tag='lynsense_readonly', output_dir=用户路径或本地 logs 下唯一带日期目录, prompt_vars={}, task_desc={'operation':'read_robot_state','arm':'right'})。提示词工厂返回固定字符串：仅调用 read_robot_state，按返回状态汇报，不调用或推断运动，非 ok 不可报成功。

`init_runtime` 根据 components 校验选择，返回未连接 adapter，无外部 daemon；无选择时返回空。`get_toolkit` 使用原有签名，调用失败保证清理；重复 Toolkit 要新建 adapter，不共享已连接实例。DashboardSpec 仅为注册表元数据提供 `/rpent-task`、空 fields、静态 display/output_slug、env unique 组件、空 frame_channels 和 primitives，不启用 Dashboard 运行。

- [x] **5. 更新已有注册表测试。** `EXPECTED_ROBOTS` 按排序加入 lynsense，`PROMPT_VARIABLES['lynsense']={}`；现有其他机器人断言保留。增加 parse_config 对 api/local、负超时、非有限值、空/非法 Topic、非法 Domain 的测试。Domain 为整数 0..232；两个 Topic 必须为不同的绝对 ROS Topic，拒绝空名及非法名称。
- [x] **6. 离线完整链路测试和用法。** 只替换 ROS 模块、provider/model 网络行为及 TTY 输入，不 mock 整个 Planner，不替换真实 `_build_tools` 或 build_api_model 的空值校验。走 `get_robot_spec -> init_runtime -> get_toolkit -> API Planner 构造工具 -> execute_tool -> close`；检查构造失败、模型异常、未调用工具即结束和成功读取后的关闭。另模拟共享 CLI 的普通 TTY 路径，使用非空 provider 前缀模型名但以假模型替换底层连接，日志写 tmp_path，断言 local memory 没有 sync、无外部调用。增加缺失 `--model` 仍报错且不连接 ROS 的用例。全部离线成功后在 README 给出未来环境就绪且授权后的入口示例（本轮不执行）：

```bash
ROS_DOMAIN_ID=3 rpent --robot lynsense --planner api \
  --model "${RPENT_MODEL:?先设置已获准的 provider:模型标识}" \
  --memory-profile local --ros-domain-id 3 --max-turns 3
```

README 区分本地离线通过与真机尚未验证；列明 ROS 消息包/Python ABI/工作空间权限、已获准的非空模型标识和对应 provider 环境前提，说明此示例会调用模型和写 RPent 日志，未获部署及 API 调用授权不得执行。不输出或保存密钥。解释每条流的接收新鲜度、状态不是安全诊断、关闭方法及未支持的 Planner。不要提供或尝试运动命令、sudo、自动安装或绕过权限步骤。

- [x] **7. 运行 green 与回归。**

```bash
.venv/bin/python -m pytest tests/unit_tests/robots/lynsense tests/unit_tests/rpent/robots/test_registry_contracts.py tests/unit_tests/rpent/cli/test_main_contracts.py tests/unit_tests/rpent/planner/test_api_contracts.py -q
```

不执行硬件/GPU/模型测试。每阶段记录退出码及测试数量；缺环境报告 NOT RUN，不更换到不支持的 Python 或下载包凑门禁。

## 文档与最终验收

- [x] 从 `git -C RPent status --short`、暂存和未暂存 diff、未跟踪文件字节建立评审包；不能只审 commit diff。代码评审包含整个 lynsense 目录、测试、README、规格和本计划。
- [x] 独立只读评审检查生命周期、禁止控制、工具表和注册兼容；发现 important/blocker 必须修复并交原 reviewer 复审。
- [x] 向用户展示一次离线 `read_robot_state` 调用顺序和实际测试结果。明确没有部署，没有读取真机消息，没有招手能力。
- [ ] 单独授权后才进入真机验证：先解决消息包可读性/导入，再验收双 Topic 首条消息、stale、关闭。当前计划完成不等于真机验收通过。

## 计划自检

规格覆盖：Task 1 对应字段与新鲜度；Task 2 对应 Domain、消息、QoS、生命周期及不控制；Task 3 对应 RobotSpec、唯一只读工具、Planner 和测试门禁。声明的函数均在其拥有任务定义；环境缺失与实机权限是明确前提，不作为已完成项。
