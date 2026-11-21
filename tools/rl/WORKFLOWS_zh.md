# 对比 LIBERO 工作流

[English](WORKFLOWS.md) · [安装](INSTALL_zh.md) · [首个任务](README_zh.md)

各模式使用同一个 API planner，LIBERO 坐标系通过 `--world-cord` 选择。
区别是感知工具和对话保留范围。成功由原生环境标志判定，todo 状态或
`finish` 的声明不能代替环境判定。公开工作流名称如下。

## 工具和上下文边界

| 工作流 | 工具 | 对话保留范围 |
| --- | --- | --- |
| `episode` | `move_to`、`set_gripper`、`view_env`、`segment`、`back_project`、`finish` | 整条任务，支持压缩 |
| `episode_image` | `move_to`、`set_gripper`、`segment`、`back_project`、`finish` | 整条任务，动作后返回原始图片 |
| `pear` | `move_to`、`set_gripper` | 整条任务，每次动作后返回 RGB 和状态，自动结束 |
| `zap_turn` | `move_to`、`set_gripper`、`change_query`、`update_todo`、`finish` | 每次请求独立 |
| `zap_episode` | `move_to`、`set_gripper`、`change_query`、`finish` | 整条任务，支持压缩 |
| `zap_subtask` | `move_to`、`set_gripper`、`change_query`、`update_todo`、`finish` | 当前子任务，支持压缩 |

每次回复严格一个工具调用，唯一例外是 `zap_subtask` 初始化时同时输出
`change_query` 和 `update_todo`，各一次。整条回复全部校验后才执行工具；
违规回复不执行任何动作，返回反馈并计入模型请求预算。

`episode` 从任务和实测状态开始，通过 `view_env` 获取图像，通过显式感知工具
获得世界坐标。`zap_turn` 首轮调用 `change_query`，次轮调用 `update_todo`；
后续输入只有当前观察、query、文字 todo 和上次原始工具结果，没有过去的对话，
无需压缩。

`zap_episode` 首轮调用 `change_query`，随后累积对话，没有 todo。
`zap_episode` 和 `zap_subtask` 的 `move_to`、`set_gripper`、`change_query`
结果自动附带当前标注图像和深度几何。感知取自当前 RGB/深度帧，不来自期望动作；
动作没有推进物理步时，最新已记录帧仍是当前观察。

`zap_subtask` 首轮同时创建 query 和完整结构化 todo。每项只有 `text` 与
`status`（`pending`、`in_progress`、`done`）。每次有效 `update_todo` 完整替换
清单，并在下一次请求前清空上下文。新上下文包含同一份 system prompt、任务、
当前 query、观察和更新后的清单，不回放旧 reasoning 或工具历史。
已有 todo 就代表初始化完成。模型依据观察确认子任务完成，控制器没有任务脚本或
oracle；无效更新不会触发重置。

累积模式的启动器默认采用 70% 上下文 / 80→10 消息窗口压缩；`zap_subtask` 只压缩当前
子任务。重置不会重置环境、全局请求次数或截止时间。默认上限为 80 次模型请求、
10000 环境步、7200 秒 planner 时间和 14400 秒外层 batch 时间。

## VLM 图片尺寸

使用 `--image_size 1024`、`512` 或 `256`，默认 `1024`；`run.sh` 也可通过
`RPENT_IMAGE_SIZE` 配置。原始 RGB、深度和 SAM3 输入始终是 1024，只改变发送给
VLM 的图片。框坐标对应显示图，像素/矩形工具参数会映射回原始深度坐标，
世界坐标仍以米计。标注直接按显示尺寸绘制。

```bash
RPENT_OUTPUT_DIR=logs/libero-zap-subtask-512 bash tools/rl/run.sh \
  --simulator libero --workflow zap_subtask --image_size 512 \
  --suite libero_object --task 0 --seed 1 --cuda-device 1
```

原始图像和世界坐标图保持不变。`api_trace/` 保存实际模型图片字节，
`workflow_requests/` 记录 context ID、必需工具和重置边界，
`workflow_responses/` 记录回复是否通过校验。`rollout.json` 包含图片尺寸、
上下文重置次数和计时。

## 共享 SAM3 池

下面在两个环境 GPU 上各部署四个独立 SAM3 副本。激活公共 RPent 环境并加载
`.rl-env.sh` 后，每条服务命令分别在一个终端运行，由该终端管理服务退出。
确认八个服务均就绪，再在另一终端启动调度器。

```bash
mkdir -p logs
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8114 --metrics logs/sam3-0.jsonl
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8115 --metrics logs/sam3-1.jsonl
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8116 --metrics logs/sam3-2.jsonl
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8117 --metrics logs/sam3-3.jsonl
python -m rpent.robots.components.sam3_server --cuda-device 1 --port 8118 --metrics logs/sam3-4.jsonl
python -m rpent.robots.components.sam3_server --cuda-device 1 --port 8119 --metrics logs/sam3-5.jsonl
python -m rpent.robots.components.sam3_server --cuda-device 1 --port 8120 --metrics logs/sam3-6.jsonl
python -m rpent.robots.components.sam3_server --cuda-device 1 --port 8121 --metrics logs/sam3-7.jsonl
```

```bash
python -m rpent.robots.components.sam3_pool \
  --backends http://127.0.0.1:8114 http://127.0.0.1:8115 http://127.0.0.1:8116 http://127.0.0.1:8117 http://127.0.0.1:8118 http://127.0.0.1:8119 http://127.0.0.1:8120 http://127.0.0.1:8121 \
  --port 8124 --per-backend-concurrency 1 --metrics logs/sam3-pool.jsonl
```

所有 planner 使用 `RPENT_SAM3_ENDPOINT=http://127.0.0.1:8124`。
ZAP 每次观察会并行请求两路相机，仍按原来的相机和 query 顺序组装结果，
两路数据来自同一个已记录观察步。仅修改 todo 时继续复用对应帧的感知缓存。

一个图像的完整 query 组交给同一副本，特征锁保护该图像的全部查询。
`--per-backend-concurrency` 控制单副本放行数，默认 1，可以测试 2；
它不会取消特征锁。调高前应同时测量完整观察耗时和副本内部锁等待。
调度器记录排队、后端往返耗时与错误；后端时间包含模型处理、mask 序列化和传输，
不能当作纯 GPU 时间。服务的 `--metrics` 分别记录 `lock_wait_seconds`
与 `model_lock_seconds`，`sam3.stats` 返回累计指标。
mask 使用快速无损 PNG 压缩，解码后的像素不变。

容量验证需要让全部环境同时驻留。八个副本本身不能证明显存足够，也不能保证
80 个双相机请求同时到达时完全没有排队。

## 模型前缀缓存诊断

`serve.sh` 启用 `--enable-cache-report`，让 OpenAI 兼容接口的
`prompt_tokens_details.cached_tokens` 返回实际复用量。未上报缓存量不等于
没有命中。每条 episode 应固定连接一个模型 endpoint。结合 SGLang prefill 日志
或逐请求缓存明细，区分长 KV 被淘汰与历史压缩、上下文重置导致的前缀变化；
发生变化的历史仍按原工作流保留。

长多模态历史可以测试下面的服务配置：

```bash
export SGLANG_VLM_CACHE_SIZE_MB=2048
SERVE_GPUS=2,3 SERVE_PORT=30180 SERVE_CONCURRENCY=32 \
  SERVE_MEM_FRACTION=0.86 bash tools/rl/serve.sh \
  --mamba-full-memory-ratio 0.5 --enable-metrics
```

这些设置只调整缓存分配和上报，不改变权重、上下文与输出上限、采样或对话内容。
需要兼容的 SGLang 版本和足够显存；换硬件前应重放同一组请求验证。
因果前缀已经变化的 token 无法靠增加缓存容量直接复用。
性能报告需保留模型与 SAM3 配置，并分别记录排队和生成时间。

## 固定任务对比

启动器默认输出32768 tokens、最多80次模型请求、压缩后保留10条尾部消息。
下方历史固定面板示例采用独立的32768 tokens／100次请求配置；复现该配置时，
须在这些比较命令前显式导出 `RPENT_MAX_TOKENS=32768`、`RPENT_MAX_TURNS=100`、
`RPENT_COMPACTION_KEEP_MESSAGES=40`。

在模型服务环境中，使用原始 Qwen3.8-27B，分别在 GPU 2–3、4–5、6–7 上部署
TP2 副本，端口 30180–30182。以下三条命令加载模型配置后，分别在三个终端执行：

```bash
SERVE_GPUS=2,3 SERVE_PORT=30180 SERVE_CONCURRENCY=32 bash tools/rl/serve.sh
SERVE_GPUS=4,5 SERVE_PORT=30181 SERVE_CONCURRENCY=32 bash tools/rl/serve.sh
SERVE_GPUS=6,7 SERVE_PORT=30182 SERVE_CONCURRENCY=32 bash tools/rl/serve.sh
```

切回公共环境，在 checkout 根目录使用新的输出目录，依次验证容量、真实工具和
模型请求，然后启动正式对比：

```bash
export RPENT_SAM3_ENDPOINT=http://127.0.0.1:8124
export MODEL_ENDPOINTS=http://127.0.0.1:30180/v1,http://127.0.0.1:30181/v1,http://127.0.0.1:30182/v1
python -m tools.rl.qualify_compare --output logs/capacity40 \
  --sam3-endpoint "$RPENT_SAM3_ENDPOINT" --gpus 0,1 --hold-seconds 300
python -m tools.rl.preflight_compare --output logs/preflight \
  --capacity logs/capacity40/verified.json --sam3-endpoint "$RPENT_SAM3_ENDPOINT" \
  --endpoints "$MODEL_ENDPOINTS" --model-path "$MODEL_PATH"
python -m tools.rl.compare_workflows --output logs/comparison \
  --qualification logs/preflight/verified.json \
  --sam3-endpoint "$RPENT_SAM3_ENDPOINT" --endpoints "$MODEL_ENDPOINTS" \
  --gpus 0,1 --per-gpu 20 --sam3-metrics logs/sam3-pool.jsonl
```

容量验证持有 40 个真实环境，执行动作和 SAM3 请求，并核实清理。预检默认运行
CPU 检查，也可提供匹配当前源码的 `--cpu-proof`；随后验证四种模式的脚本化真实
工具链和六条 seed 1 的短 VLM 链路。这些都不计入正式得分。
运行检查前需安装仓库测试依赖。

控制器先将四种模式分别在 1024 下跑完整 40 tasks、seed 0；按原生成功数选优，
同分按整组总耗时、再按模式声明顺序决定。只对胜出模式补跑 512 和 256，
1024 直接复用，共 240 条正式轨迹。任务与 GPU/模型副本的分配固定，40 个环境
驻留后才统一放行模型请求；每卡最多同时冷启动两个环境。该布局必须在实际硬件上
单独通过容量验证。

组间配置冻结：context 262144、output 32768、thinking+xhigh、temperature 1、
top-p .95、top-k 20、显式 presence penalty 0、repetition penalty 1。
每组完成后清理模型 prefix cache，服务保持预热；源码必须匹配资格验证哈希。

查看 `report-<workflow>-<size>.json`、`winner.json`、`final-report.json`。
报告保留成功、正常失败和基础设施错误，并比较相同且共同成功任务的完成耗时。
正常失败和模型预算耗尽不重跑；基础设施错误最多额外尝试一次，得分分母始终为
40。对比目录中的 `STOP_REQUESTED.json` 会停止新任务和自有 collector。
保留全部输出，只有配置完全相同时才能恢复中断的运行。

## 图片 episode、完整位姿、动作列表与历史保留

LIBERO `episode_image` 在初始及物理动作后返回当前 agentview、wrist 双相机 RGB。
它有 `segment`、`back_project`、`finish`，没有 `view_env`；感知仍由模型显式调用，
不自动执行 SAM。此模式的感知参数不含 `step`，固定使用当前观测；动作前的旧 mask
必须重新分割。原 `episode` 保留历史帧的显式 `step`。

LIBERO 的 `move_to(xyz?, orientation_rpy?)` 至少提供一个目标。
`xyz` 为 MuJoCo grip site 的绝对世界坐标，单位米；`orientation_rpy=[roll,pitch,yaw]`
为同一参考帧相对世界的绝对姿态，单位弧度，固定轴 XYZ，满足
`R=Rz(yaw)@Ry(pitch)@Rx(roll)`。`[pi,0,yaw]` 表示接近轴朝下，夹爪沿 site 的
x 轴闭合。省略项保持该动作开始时的实测值。反馈包含实际 `pose.xyz`、
`pose.orientation_rpy`、开合度及残差；`partial` 不表示到达。
RoboCasa 和 RoboTwin 的 base 坐标模式沿用原有动作接口。

`--action-mode single` 为默认；选择 `batch` 后，独立 `move_to`、`set_gripper`
替换成 `execute_actions`，感知工具和 `finish` 继续可用。
`--max-actions-per-call K` 默认 3，最小 1：

```json
{"actions":[
  {"action":"move_to","xyz":[0.1,0,0.3],"orientation_rpy":[3.14159265,0,0]},
  {"action":"set_gripper","opening":0}
]}
```

开始前校验整个列表。逐项执行并测量，成功、截断、error 或 partial 立即停止，
剩余项不执行。返回各项实测位姿、`executed_count`、从 0 开始的 `stopped_at`、
`stop_reason`、`final_state`。`episode_image` 只发送批末／暂停时的最新双相机图，
中间帧完整留档。每次回复仍只有一个顶层 tool call；一批计一次模型请求，
各原子动作共用既有环境步数与时间预算。

`--reasoning-history N` 按之前的模型回复计数，一次回复的 thinking 片段共同一组。
`--image-history N` 按图片回传计数，双相机共同保留，初始观测也算一组；
单相机 view_env 的回传同样是一组。两个默认值都是 -1：

| 值 | 含义 |
|---|---|
| -1 | 不额外裁剪，仍受现有历史压缩约束 |
| 0 | 移除该类全部内容；图片连最新观测也移除 |
| N > 0 | 只保留最近 N 个回复／观测组 |

先裁剪再压缩，不删除工具调用、普通 assistant 文本与实测文本结果。
原始回复和观测留档。删除历史 reasoning 不关闭当前轮 thinking，不插入虚构 ignore
文本；Qwen 原生模板可能产生空 thinking 标记。裁剪旧历史可能降低前缀缓存复用，
需结合实际延迟和缓存指标判断。新开关目前支持本机 LIBERO 的 `episode`、
`episode_image`，其他组合在启动仿真前报错。

完成[安装](INSTALL_zh.md)、激活环境、加载复制后的配置并启动模型／SAM 服务后，
将 `RPENT_OUTPUT_ROOT` 设为可写目录，运行：

```bash
bash tools/rl/run.sh --simulator libero --workflow episode_image   --suite libero_object --task 0 --seed 0 --cuda-device 0   --action-mode batch --max-actions-per-call 3   --reasoning-history 0 --image-history 1

python -m tools.rl.run_batch run "$RPENT_OUTPUT_ROOT/image-batch"   --jobs tools/rl/jobs/panel40.txt --seeds 0 --gpus 0,1 --per-gpu 20   --start-barrier --workflow episode_image --action-mode batch   --max-actions-per-call 3 --reasoning-history 3 --image-history 3
```

manifest 固定这些配置，断点恢复不允许混用参数。独立八组实验先跑两个 workflow
与两种动作格式的四组 -1/-1，再对各 workflow 的胜出动作格式测试 3/3、0/1。
不调用 `compare_workflows.py`，后者属于另一套 ZAP／图片尺寸比较。

分割成功返回 dict，不含 `score`、`found`；查询不合法与未检测到目标分别返回
修复提示，服务异常单独报告。`world_xyz` 为可见深度点逐轴中位数，
`top_z_m` 为 world-z 的 95 分位数，`size_xyz_m` 为各轴 5～95 分位跨度，
`axis_yaw` 为细长可见表面的 xy 主轴方向或 null；它们不是完整尺寸或抓取目标。

### 复现八组比较

按下表串行运行，每次只启动一组。各组使用相同 LIBERO-PRO 40-task 清单、
seed 0、image_size 1024、100 次模型请求和 10000 环境步，模型、采样及时间预算
保持一致。

| 组 | workflow | 动作模式 | reasoning-history | image-history |
|---|---|---|---:|---:|
| 1 | episode | single | -1 | -1 |
| 2 | episode | batch，K=3 | -1 | -1 |
| 3 | episode_image | single | -1 | -1 |
| 4 | episode_image | batch，K=3 | -1 | -1 |
| 5 | episode | 组1/2胜者 | 3 | 3 |
| 6 | episode | 组1/2胜者 | 0 | 1 |
| 7 | episode_image | 组3/4胜者 | 3 | 3 |
| 8 | episode_image | 组3/4胜者 | 0 | 1 |

每个 workflow 独立选择动作模式：原生成功数最高者胜出；同分比较整组 wall，
仍相同选 single。把所选动作模式代入上面的批处理命令。正式共320条，
接口预检与容量资格验证另记，不计入得分。

8卡布局中，GPU2-7 放三个 TP2 模型；GPU0/1 各驻留20个环境和4个 SAM3，
共享一个 pool endpoint。用最终源码验证40环境加8个 SAM 至少保持300秒，
40环境全部 ready 后才统一放行首批模型请求。各组采用相同预热及缓存清理口径，
记录清理结果；KV flush 不保证清空 VIT 特征缓存。
保留所有尝试，只有基础设施异常最多额外重试一次。除整组 wall 外，
还应比较相同且共同成功任务的耗时，不能把提前失败算作加速。

Qwen 通过 SGLang 运行且存在无参数工具时，使用带有界空白的原生 XML 结构化工具格式。
实际请求使用 tool_choice=auto，并显式提供携带同一份工具 schema、至少调用一次工具的格式；
允许的工具调用要求保持不变。语法空白最多 16 个字符，避免 finish() 的空参数区域循环。
不添加占位参数，不修改依赖、提示词或采样；整条回复仍须先通过严格校验。
上游说明：https://github.com/mlc-ai/xgrammar/issues/802。

## Episode 感知列表

`--workflow episode --perception-mode batch` 将独立感知替换为 observe，
由 source 推导依赖，独立节点并行。[CLI、示例、ID 与返回说明](OBSERVE_zh.md)。

## RoboTwin 双臂共享坐标的 episode

RoboTwin 的 episode 可以启用双臂共享坐标系：

```bash
bash tools/rl/run.sh --simulator robotwin --task-name click_alarmclock --seed 100000 \
  --workflow episode --world-cord general --action-mode batch \
  --max-actions-per-call 3 --perception-mode single --prompt simple
```

原点在两个机械臂安装基座的中点，X 向前、Y 向机器人左侧、Z 向上。控制目标和实测
TCP 均以指垫中点为基准，位置用米，绝对姿态用固定轴 XYZ 弧度。每个动作通过 `arm`
指定左臂或右臂。可使用 `other_xyz`、`other_orientation_rpy` 或 `other_opening`
在同一动作内同步控制另一臂。
`view_env`、`segment` 和 `back_project` 仍按相机和观测步数查询，返回的三维
几何量使用共享坐标系。

## 基座坐标与纯图像控制

`--world-cord base` 默认保留仿真世界坐标和现有 `grip_site`。所有 LIBERO RL workflows 都支持 `--world-cord general`：原点在安装基座中心，x 向前、y 向左、z 向上，TCP 为两指指垫中心的实测中点。初始反馈给出桌面相对基座的高度，控制器负责补偿 TCP 偏移。位置单位为米，姿态是固定轴 XYZ 欧拉角，单位为弧度。

几何结果统一遵循所选坐标系，覆盖批量 observe 和 ZAP 的 `world_xyz`、`top_z_m`、`size_xyz_m`、`axis_yaw` 及输入 `z_range_m`。原始 `*_world.npz` 保留仿真世界坐标，`coordinates.json` 记录变换。有效点先转换，再按高度筛选和计算范围。图像及相机、观察步数的绑定不变。

`pear` 沿用 episode_image 的会话机制，初始和动作后自动提供固定相机、腕部相机 RGB 及实测状态，不使用 SAM3、深度感知、动作批处理或 `finish`。每轮恰好调用一个 `move_to` 或 `set_gripper`。`--reasoning` 开启 thinking；`--no-reasoning` 关闭 thinking，并要求直接 tool call、不输出文字，分别等价于原有的 thinking 开关。原生成功或预算耗尽后自动结束；无效回复立即按零奖励任务失败结束，不请求纠正，动作 partial 本身不会结束任务。

```bash
bash tools/rl/run.sh --simulator libero --suite libero_goal --task 0 --seed 0 \
  --workflow pear --world-cord general --action-mode single --no-reasoning
```

只运行 pear 的远程 manager 可省略 `--sam3-endpoint`，现有感知工作流仍需 SAM3 端点。RPC 两端使用相同 RPent 版本；general 必须从更新后的环境服务读取基座、桌面和指垫标定。
