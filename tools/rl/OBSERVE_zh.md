# Episode 感知列表

[English](OBSERVE.md) · [环境安装](INSTALL_zh.md) · [Workflow](WORKFLOWS_zh.md)

`observe` 是本地 `episode` 可选的感知接口，将 `view_env`、`segment`、
`back_project` 放入一个平铺的 `calls` 列表。不移动机器人，也不在工具内部额外调用模型。

## 启动与配置

按[首次运行指南](README_zh.md)配置 `.rl-env.sh` 并启动模型与 SAM3。在 checkout 根目录：

```bash
source .venv/bin/activate
source .rl-env.sh
source tools/rl/env.sh
bash tools/rl/run.sh --simulator libero --workflow episode \
  --perception-mode batch --max-observe-calls 8 --observe-concurrency 8 \
  --suite libero_object --task 0 --seed 1 --cuda-device 0
```

| CLI 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--perception-mode` | `single` | `batch` 将三个独立感知工具替换为 observe |
| `--max-observe-calls` | `8` | 列表子操作总数上限，包含 view，须大于 0 |
| `--observe-concurrency` | `8` | 独立子操作并发上限，须大于 0；不改变 SAM worker 限制 |

三个参数也适用于 `python -m tools.rl.run_batch run`，写入 manifest，恢复时固定。
`run.sh` 支持对应的 `RPENT_PERCEPTION_MODE`、`RPENT_MAX_OBSERVE_CALLS`、
`RPENT_OBSERVE_CONCURRENCY` 环境变量。非默认上限必须配合 batch。
`episode_image`、全部 `zap_*` 和远程 toolkit 会在启动前拒绝此选项。
本地 episode 使用各仿真器真实相机名称。动作工具和 `finish` 保留；
动作批处理 `--action-mode batch` 与历史裁剪可独立选择。每条模型回复仍只有一个顶层工具。

## source 与执行

每项包含唯一局部 `id` 和 `op`。`@` 名称只在本次调用有效；普通数据 ID 在当前 episode 持续有效。

| 子操作 | source | 其他参数 |
| --- | --- | --- |
| `view_env` | 不需要，也不允许传 | `camera`，默认 all，读取最新观测 |
| `segment` | 图片 ID 或 `@view_id.camera` | 非空 prompt／point=[row,col] 严格二选一 |
| `back_project` | 图片 ID 或 `@view_id.camera` | pixel=[row,col]／box=[r0,c0,r1,c1] 严格二选一 |
| `back_project` | mask ID 或 `@segment_id` | 不传 pixel/box，可传 z_range_m |

`z_range_m=[low,high]` 是世界坐标 z 的闭区间过滤，单位米，仅用于 box/mask。
box 左闭右开，宽高必须大于零。像素使用对应的模型可见图片尺寸，内部映射到原始深度。

初始状态和动作反馈都有 `image_sources`（相机→图片 ID），即使没有附 RGB。
每个 `image_id` 固定绑定帧、相机和像素映射，必须复制，不能编造。
旧图片／mask ID 有意测量旧帧，不能跨 episode 使用。mask 已固定帧和相机。
因此测量子操作没有 step、camera、frame_from、mask_from、mask_id 参数，统一通过 source 选择。

整个列表先校验，再执行任何子项。非法参数、未知／重复 ID、越界、循环依赖拒绝整个调用；
允许向后引用。代码从 source 推导依赖，独立节点并行，依赖节点只等自己的来源完成。
同次调用的 view 使用入口时冻结的最新观测，动作不能插入。

未找到目标返回简短说明，服务异常返回不同错误。只跳过失败来源的依赖项，保留独立结果。
不隐藏重试、不改 query。如果需要先看新图才能决定描述，应先返回 view，下一轮再决定。

## 示例

只有 view，不传 source：

```json
{"calls":[{"id":"v","op":"view_env","camera":"all"}]}
```

view 后独立分割两个目标，再测量一个 mask。surface 只等 cup，不等 plate。
每项 segment 选择一个 mask，两项成功即返回两个 mask ID。

```json
{"calls":[
  {"id":"v","op":"view_env","camera":"all"},
  {"id":"cup","op":"segment","source":"@v.agentview","prompt":"red cup"},
  {"id":"plate","op":"segment","source":"@v.wrist","prompt":"plate"},
  {"id":"surface","op":"back_project","source":"@cup","z_range_m":[0.8,1.2]}
]}
```

无需 view／segment，直接对先前图片、mask 反投影。示例 ID 必须替换为真实返回值，
像素来自对应图片。两项独立并行：

```json
{"calls":[
  {"id":"point","op":"back_project","source":"img_EXAMPLE_0_agentview","pixel":[512,512]},
  {"id":"mask","op":"back_project","source":"m_EXAMPLE","z_range_m":[0.8,1.2]}
]}
```

也可以只做 segment，source 复制 image_sources.agentview，不附图：

```json
{"calls":[{"id":"cup","op":"segment","source":"img_EXAMPLE_0_agentview","prompt":"red cup"}]}
```

## 返回与审计

返回 `{"results":[{"id":"...","op":"...","result":...}]}`，按输入顺序排列。
每个 result 是成功 dict 或说明文本，必须逐项阅读。

- view：step 和 images；每张包含 image_id、camera、size=[height,width]、从 0 开始的
  image_index，对应所附 RGB。同一图片只附一次，只有 view 返回图片。
- segment：image_id、mask_id、step、box 和可见表面几何。world_xyz 是有效表面点中位数；
  top_z_m 是 z 的第 95 百分位；size_xyz_m 是各轴第 5–95 百分位范围；
  axis_yaw 是 xy 主轴方向 [-pi/2, pi/2)，不明显细长时为 null，不是夹爪指令。这些不是隐藏物体中心或保证成功的抓取位姿，不显示 score／found。
- back_project：image_id、来源 step、深度测得的 world_xyz。mask/box 可能受背景或遮挡影响。
  segment 已包含表面几何，只有另选点／区域或使用 z 过滤时才需额外反投影。

observe/sources.json 记录图片／mask 来源；observe/NNNN.json 记录节点、依赖、状态、
调度／线程等待与起止时间。并行项时间重叠，不能相加当作 wall 时间。
原生帧、原始 SAM 审计、transcript、api_trace 保留。返回的 step 是审计元数据，不是第二套来源参数。
