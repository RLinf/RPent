# RPent 公司机器人只读状态接入设计

日期：2026-09-15

状态：已完成本地实现、假 ROS 测试及独立代码复审（2026-09-15）。扩展回归门禁 175 项通过；尚未部署或进行真机订阅验收。

## 目标

在 RPent 中增加一个 `lynsense` 真机扩展，使 RPent 运行在机器人一号主机上时，能够通过 ROS 2 读取右臂状态。第一轮只验证连接、状态快照和错误反馈，不发送任何运动、夹爪、使能或状态切换命令。

机器人一号当前验证到的运行条件：ROS 2 Humble，ROS Domain ID 为 `3`，工作空间为 `/home/rpp/rpp_ws`。右臂状态接口为 `/right_xarm/robot_states`（`xarm_msgs/msg/RobotMsg`）和 `/right_xarm/joint_states`（`sensor_msgs/msg/JointState`）。

证据限制：2026-09-15 的 ROS 图检查验证了 Domain 3 中的名称和类型，并观察到驱动进程引用该工作空间；`zxh` 未能读取该工作空间，未成功验证 source 或导入 `xarm_msgs`，也未读取实际消息。图中可见不等于本账号能够订阅或设备可安全运动。不得用 sudo、修改权限或其他账户绕过此限制。

## 不在本轮范围内

- 不调用 `/right_xarm/set_position`、`/right_xarm/set_servo_angle*` 或任何其他运动接口。
- 不调用夹爪 Action、`motion_enable`、`set_state`、清错或参数写入服务。
- 不接入视觉、手眼标定、导航、py_trees 或 LLM 生成运动轨迹。
- 不在机器人上安装依赖、修改文件或启动额外 ROS 2 进程；本轮实现允许 RPent 进程内部创建自己的只读 ROS 2 Node。用户另行批准了本地 `.venv` 和 RPent 基础/测试依赖，已使用 Python 3.10.20 配置；这不构成远程部署或 ROS 环境修改授权。

## 模块与接口

新增 RPent 机器人扩展 `RPent/robots/lynsense/`，沿用现有 `RobotSpec` 和 `Toolkit` 接口。

### RobotSpec

`robot_spec.py` 注册机器人名称、提示词、命令行配置和运行时初始化函数，包入口导出 `get_robot_spec()` 和 `get_toolkit(...)`。`init_runtime(args, output_dir, dashboard_events, components)` 返回 `([], {"adapter": adapter})`，其中 adapter 此时只是未连接的对象，不持有 ROS context、Node 或线程。`components=None` 或 `{"env"}` 选择本轮运行时，空集返回 `([], {})`，其他组件报配置错误。

Toolkit 构造完成工具注册后调用 adapter 的 `connect()`；构造或连接失败由 `get_toolkit` 关闭 adapter 后重新抛出异常，正常路径由 `Toolkit.close()` 关闭。这样 runner 在 Toolkit 构造前退出也不会泄漏 ROS 资源，无须修改 RPent 共享 runner。每个 Toolkit 使用独立 adapter，不支持共享已连接实例。

实施评审补充：adapter 提供 `claim_toolkit()` 原子认领，防止复制运行时参数或并发调用工厂造成重复所有权。认领失败不关闭已有拥有者的资源；工厂仅在认领成功后负责构造/连接失败的清理。

运行时配置至少包含：

- `ROS_DOMAIN_ID` 期望值，默认值为 `3`；它只用于校验，不能在 `rclpy.init()` 后由扩展修改进程环境；
- 必要的 ROS Python 包是否可导入；不设置“已 source”布尔开关，不以路径存在代替导入检查；
- 右臂状态 Topic 名称，默认使用上述两个实际名称；
- 状态读取超时和状态新鲜度阈值。

初始化和连接不发送控制请求。本轮仅支持 API Planner、local memory、普通终端；保留 RPent 的真机 TTY 要求，不开启 Dashboard、interactive、exploration、task-card 或带通用 shell 工具的 Planner。日志仍使用 RPent 原有机制；本轮不在机器人上运行该程序或写日志。

### ROS 2 适配器

`ros2_adapter.py` 持有一个 `rclpy` Node 和两个订阅。适配器负责把 ROS 消息转换成 RPent 内部的稳定状态结构，隐藏 ROS 消息类型、QoS、回调和线程细节。适配器启动一个由自己拥有的 `SingleThreadedExecutor` 线程驱动回调；`connect()` 在发现窗口内等待两个 Topic 都可见且各收到首条消息，超时后返回 `failed`。并发读取快照使用锁或不可变快照，`close()` 先停止 executor、等待线程退出，再销毁 Node。`starting` 只作为连接过程中的内部状态，不从 `connect()` 返回。

第一轮要求独立 RPent 进程，由 adapter 的 `connect()` 懒加载 ROS 包、检查默认 context 未被初始化，再初始化并独占该默认 context。检测到已初始化的默认 context 时拒绝连接，不关闭它；不宣称能够枚举其他模块创建的所有私有 context。adapter 只在自己成功初始化后获得 shutdown 所有权，失败路径也按该所有权清理。

清理显式传入持有的 context，成功后才摘除仍指向它的默认槽位。若 shutdown 抛错，保留该 context 供重试；默认槽位若已被替换，不能关闭或摘除替代 context。

Domain 环境值必须在连接前显式设置，且与配置期望值相同；未设置不自动使用 Domain 0，扩展不修改环境。只创建两个目标订阅，不创建机器人控制 publisher、Service client 或 Action client；Node 禁用参数服务和 rosout，并忽略进程级 ROS 重映射。ROS/DDS 自身发现或参数事件设施不等于运动指令，不能声称整个进程绝无发布器。

适配器对调用方提供小接口：

```text
connect() -> dict  # status 为 ok 或 failed，失败包含 reason
get_snapshot() -> dict  # 结构见下文快照字段
close() -> None
```

状态不足、Topic 不存在、消息超时或消息字段无效时返回明确的非成功状态，不伪造有效状态。Topic 发现等待窗口为配置项；两个 Topic 中任一必需 Topic 在窗口内不可见，或任一 Topic 在窗口内没有首条消息，则连接失败。发现并收到消息后，无新消息超过新鲜度阈值则为 `stale`。

默认等待 5 秒、新鲜度阈值 2 秒，均须为正有限值。以本地 monotonic 接收时间分别判断两条流是否过期，不把它们伪装成同步快照，也不声称接收时间证明传感器采样时间。两个订阅使用 BEST_EFFORT、VOLATILE、KEEP_LAST(10)。executor 每次 `spin_once` 最多等待 0.1 秒；回调仅转换和缓存，异常唤醒等待者并报告失败。关闭等待总预算 2 秒，超时必须显式报错，不能假报资源已释放。

### Toolkit

第一轮只通过 `add_tool("read_robot_state", spec, handler)` 注册一个工具，handler 使用现有 `readonly` 标记。子类覆盖 `_register_common_tools()`，不注册父类默认的文件读写或 `finish` 工具；API Planner 通过普通文本结束，不伪造 `_finish` 成功结果。工具的参数 schema 是不接受任何字段的空对象。

源码核查补充：API Planner 的 `_build_tools` 还会独立注入 `read_image` 并访问 `toolkit.state`，仅清空通用工具不足以限制最终模型工具表。因此在基础 Toolkit 增加 `include_image_reader=True` 的默认能力声明；Lynsense 设为 False，API Planner 仅在该声明为 True 时构造图像读取器并访问 EnvState。其他后端保持默认行为，Lynsense 不创建无用途的图像状态对象。回归测试必须检查真实 `_build_tools` 生成的最终工具表，而不只检查 Toolkit 注册表。这是本轮唯一需要调整的共享执行代码，不修改 runner 生命周期。

快照始终返回 `status`、`reason`、`observed_at`、`joint_names`、`positions`、`robot_state_received` 和两条流的 `age_s`。`ok` 时关节数组非空、名称非空且唯一、长度相等、数值有限；`observed_at` 是关节消息的 UTC 接收时间。未收到合法关节消息时数据字段为 null，不能用零数组伪造；可选 `velocities`、`efforts` 无效时省略并提供 warning。状态取 `ok`、`unavailable`、`invalid`、`stale`、`failed`、`closed`；`ok` 仅表示数据可读，不代表驱动器健康、机械臂空闲或允许运动。第一轮对 `RobotMsg` 只记录收到消息及接收时间，不解析其诊断字段。

第一轮不把运动原语注册给 Planner。后续运动原语和 `wave_hello` 技能必须在单独设计、离线测试和现场监督验证后加入。

## 数据流

```text
RPent 启动
  -> RobotSpec.init_runtime
  -> 创建未连接 adapter（无 ROS 资源）
  -> get_toolkit / adapter.connect
  -> rclpy.init / 创建 Node 和 executor
  -> 订阅右臂状态 Topic
  -> Toolkit 读取状态快照
  -> Planner 获得结构化状态
```

RPent 不执行 `source`；操作人员须提供可导入 `rclpy`、`sensor_msgs`、`xarm_msgs` 的兼容 Python 环境。`/home/rpp/rpp_ws` 仅是已观察到的部署路径，访问权限和 Python 类型支持仍需另行验证。普通 RPent 安装不新增 `rclpy` 依赖；枚举机器人、导入规格及离线测试不得要求 ROS 可用。

## 错误与生命周期

- ROS 包缺失、默认 context 已被初始化或 Domain 不匹配：连接失败并说明原因；未初始化的 context 是正常启动条件，不是错误。
- Topic 在发现窗口内不可见或没有首条消息：`connect()` 返回 `failed`；等待期间只保留内部 `starting` 状态，不能把空值作为机器人状态返回。
- 状态超过新鲜度阈值：返回 `stale`，Planner 不得据此发起后续动作。
- `close()` 幂等：设置停止标志、唤醒等待者和 executor、等线程退出、关闭 executor 并销毁 Node，最后只 shutdown 自己成功初始化的 context。关闭后快照为 `closed`，再次连接同一对象报错；executor 故障或 Toolkit 构造失败执行同样清理，不更改其他 ROS 资源。
- 本轮 Toolkit 只暴露 `read_robot_state`，测试其注册表及 ROS 工厂调用中均无机器人控制能力。
- executor 故障在 `spin_once` 完全退出后，由该线程自动调用清理；不等待自身，外部关闭者等待该线程时必须释放生命周期锁，避免互相等待。资源释放后仍保留 `failed` 及原始故障原因；普通主动关闭返回 `closed`。自动清理失败会记录错误，并保留资源供拥有者重试。
- 取消和停止接口在本轮不实现；它们属于下一轮运动控制设计的前置约束。

## 验证

离线测试使用假的 ROS 适配器或消息转换输入，覆盖：正常快照、缺失字段、空关节数组、长度不一致、NaN/Inf、重复关节名、超时、过期状态、Topic 发现失败、Domain 配置失败、已有 context、executor 关闭生命周期和只读工具注册。测试不连接外部服务。

本轮执行门禁是本地假 ROS 测试与源码审查；不调用真实模型，不连接机器人。本机 Python/pytest 环境缺失时报告测试未运行，不自动安装。真机上的 RPent 订阅验收保留为单独授权的后续门禁，先解决工作空间可读性和消息包导入；离线通过不能称为真机接入成功。
