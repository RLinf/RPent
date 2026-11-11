# 对比 LIBERO 工作流

[English](WORKFLOWS.md) · [安装](INSTALL_zh.md) · [首个任务](README_zh.md)

四种模式使用同一个 API planner，并共用机器人、坐标和夹爪说明。
区别是感知工具和对话保留范围。成功由原生环境标志判定，todo 状态或
`finish` 的声明不能代替环境判定。公开工作流名称为下面四种。

## 工具和上下文边界

| 工作流 | 工具 | 对话保留范围 |
| --- | --- | --- |
| `episode` | `move_to`、`set_gripper`、`view_env`、`segment`、`back_project`、`finish` | 整条任务，支持压缩 |
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

累积模式沿用 70% 上下文 / 80→40 消息窗口压缩；`zap_subtask` 只压缩当前
子任务。重置不会重置环境、全局请求次数或截止时间。默认上限为 100 次模型请求、
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
