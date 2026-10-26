# LIBERO 规划器工具批量执行实验

[English](README.md)

本实验入口比较普通工具调用与有长度上限的 `execute_sequence`。规划器一次提交多个原有工具及已知参数，程序同步依次执行，复用原工具的状态记录和取消检查，最后返回各动作结果与最后观测。它不合并 VLA 请求、不改变 action chunk，也不让 Pi0.5 接管脚本移动控制。

## 运行

在具备 [LIBERO-Pro 环境](../../../docs/source-zh/rst_source/simulators/libero.rst)的 Linux GPU 主机上，从干净的 RPent checkout 运行。Python 环境需要支持本地模拟器和 Codex 规划器，并已配置 Codex 认证。先分别启动 OpenPI、APXInf 的 RPent RPC 服务和 SAM3，部署方式见上述指南中的 APXInf 部分。两种 VLA 后端应使用相同 checkpoint、归一化参数、观测路由与动作 chunk 设置。实验结束后，共享服务继续运行。

准备冻结的本地评测记忆目录和 provenance JSON。该 JSON 是操作者提供的证据，脚本不会自动验证其真实性：应记录数据集和初始状态版本／哈希、记忆来源、权重和归一化哈希、RPent／RLinf／OpenPI／APXInf-robo／引擎版本、dtype、chunk 设置、GPU、CUDA 和相关环境变量；不要包含凭证。运行前应比较后端元数据并核验这些输入。

```bash
python -m robots.libero.experiments.benchmark_batching \
  --backend openpi=http://127.0.0.1:9012 \
  --backend apxinf=http://127.0.0.1:9014 \
  --sam3-endpoint http://127.0.0.1:9018 \
  --memory-dir /path/to/frozen-memory \
  --provenance /path/to/provenance.json \
  --model gpt-6-astra --reasoning-effort low \
  --tasks 0 1 2 3 4 5 6 7 8 9 --seeds 0 1 \
  --batch-sizes 0 4 \
  --output-dir logs/batching-object-01
```

各组使用同一个可用的 Codex 模型。当前支持 `libero_object_task`，包含将 10 种指定物品分别放入篮子的任务。上述命令运行 80 回合：10 个任务 × 2 个 seed × 2 个后端 × 2 种批量设置。原环境服务将 seed 对初始状态数量取模，选择对应状态；这不是完整的 LIBERO-Pro 评测。不同任务／seed 配对之间轮换各组顺序，并检查配对初始相机图像的哈希一致。该检查不证明后续轨迹一致或后端数值等价。

`--batch-sizes` 同时控制参数校验、工具描述和追加的规划器提示词：

- `0`：普通工具组，不提供序列工具；应保留此基线。
- `1`：提供序列工具，但每次仅允许一个子工具，作为接口／提示词对照。
- `4`、`8` 等：每次最多执行的子工具数量，不是目标值或下限。

规划器仍可调用普通工具。提高上限不保证实际批量长度增加，应统计每批实际动作数和规划器回复数。相同任务与后端下，`--batch-sizes 0 1 4 8` 会运行 160 回合。一个 `pi0_pick` 内部可多次调用 VLA，因此批量上限不是 VLA chunk 数。工具内部预算和成功阈值保持不变。实验同时改变工具可用性和提示词，不能分离两者的贡献。

默认上限为 100 次规划器回复、900 秒规划器时间和 10,000 个环境步，可通过 `--max-turns`、`--planner-timeout-s`、`--max-episode-steps` 对所有组统一修改。工作进程看门狗额外允许 900 秒用于准备与清理。每回合启动独立本地环境服务，不支持真机或 Flash 模式。

## 序列边界

任何动作执行前，先校验全部子工具名及参数。允许 `move_to`、`move_pose`、`rotate_wrist`、`set_gripper`、`pi0_pick`、`pi0_doubled`、`release`；不允许嵌套序列、reset 或感知工具。未知坐标需要先单独感知和规划。

遇到以下任一条件就返回：

- 工具取消或报错；
- 环境原生任务成功或被截断；
- `pi0_pick` 未明确报告 `success=True`；
- 工具报告未收敛或位置保持失败；
- `move_to`／`move_pose` 未提供最终位置距离，或距离超过容差加 0.0001 米（兼容结果舍入）；
- `pi0_doubled` 执行完，交回规划器判断接触结果；
- 已提交列表执行完。

`pi0_doubled.success` 表示完整任务终止，不表示接触技能成功，因此将它设为观察边界，而非判为抓取失败。这些是读取原有反馈的实验性继续执行规则，没有改变模拟器成功条件，也不能证明每个子目标都已达成。例如 `move_pose` 没有报告最终 yaw 误差，本实验未额外增加姿态验证器。原工具仍负责内部闭环控制。序列提前返回通常只是交回规划器，不等于整个任务失败。

## 证据与续跑

manifest 记录命令设置、提供的 provenance、冻结记忆哈希、实验源码哈希和 RPent revision。每回合保留独立记忆副本、渲染后的 system prompt、规划器文本／原始事件、工具调度日志、RPC 日志、环境日志、状态／图片／视频和结果。文件可能占用大量空间，需为完整矩阵准备存储；脚本不会删除实验文件。

`result.json` 读取环境原生终止标志，不使用规划器 `finish` 中的成功声明。规划器预算耗尽作为任务结果；服务／传输失败或工作进程异常退出作为基础设施失败，并停止后续实验。未知规划器错误保守归为基础设施失败，供人工检查。保留每次尝试，不自动重试或替换失败结果。

使用完全相同的命令加 `--resume`，可跳过已完成的有效回合；manifest 输入必须一致。基础设施失败或中断回合不会自动重试，需先检查证据，再使用新输出目录进行单独披露的补跑。不能将成功补跑混入原结果而不说明选择规则。

`summary.json` 给出有效成功数／总数、全样本和成功样本平均耗时、实际批量使用情况、每次规划器回复对应的 VLA 调用数、VLA 时间占比与配对成功变化。配对加速只统计两边都成功的回合，并报告更快的配对数；提前失败不算提速。同一任务的多个 seed 有相关性，这些描述性统计不能证明成功率等价。

整回合计时从环境启动前到规划器返回后，不包含最终清理和共享服务启动／就绪检查；包含工具初始化、记忆复制、规划、动作执行和状态采集。VLA RPC 计时包含 RPC 调用范围内的客户端编码／传输，不是纯 GPU 推理时间，也不包含模拟器执行。工具与 RPC 计时相互包含，不能相加；剩余时间也不能当作 LLM 延迟。轨迹差异和 RPC 延迟可能独立于推理速度改变整回合耗时。

## 离线检查

```bash
pytest tests/unit_tests/robots/libero/test_batching_experiment.py -v
```

测试使用替身验证调度、停止、取消和统计行为，不代表 GPU 任务成功，也不验证 benchmark 依赖已就绪。
