# RoboDojo ↔ RPent 接口规范（M0）

本文档记录把 RoboDojo（Isaac Sim）接入 RPent 所需的接口事实，
来自 2026-08-20 的 M0 摸底（官方 debug gate PASS）与源码阅读。

> **环境安装**：完整复现环境（RPent / sim / Pi_05 三个运行时 + 资产 +
> checkpoint）的搭建步骤见
> [`docs/ROBODOJO_INSTALLATION.md`](../../../docs/ROBODOJO_INSTALLATION.md)。

## 1. 启动契约

RoboDojo 仿真栈运行在独立 conda 环境（与策略环境隔离）：

```bash
ROBODOJO_WORKSPACE=/home/admin/robodojo_pro6000_ws
ROBODOJO_SOURCE_ROOT=$ROBODOJO_WORKSPACE/src/RoboDojo
ROBODOJO_SIM_ENV=/home/admin/robodojo_runtime/envs/robodojo-sim
ROBODOJO_XPOLICYLAB_ROOT=$ROBODOJO_SOURCE_ROOT/XPolicyLab

PYTHONPATH=$ROBODOJO_SOURCE_ROOT:$ROBODOJO_XPOLICYLAB_ROOT
$ROBODOJO_SIM_ENV/bin/python -u <server.py> --headless ...
```

其余缓存/显存路径环境变量见 `config/runtime.env`（OMNI_CACHE_PATH、
XDG_*、CUDA_CACHE_PATH 等），启动前需全部导入。Isaac Sim 的
`AppLauncher` 必须在任何 isaac/isaaclab import 之前、进程最早期调用，
且每进程只能有一个实例 → env 服务必须独立进程。

## 2. 环境 API（src/collect_client/collect_env.py）

- `create_collect_env(config, app)` → `CollectEnv`
- `env.reset(seed=layout_id)` — seed 即布局 id（`env.seed_manager` 维护）
- `env.get_obs(env_idx=0)` → 观测 dict
- `env.apply_target(control_info, env_idx=0)` — 推入控制序列并步进到完成，
  随后 `reward_manager.step`
- `env.is_success(env_idx=0)` → reward 最终判定 > 0.999
- `env.take_action_cnt[0]` / `env.step_lim` → 当前步 / 步上限（终止）
- `env.traj_recorder.record_obs(env_idx, obs)` — 视频/轨迹记录
- `env.reward_manager` / `env.robot_manager` / `env.scene_manager` / `env.seed_manager`

## 3. 观测格式（obs_manager.get_obs 真实输出）

```python
obs = {
  "vision": {
    "cam_head":        {"color": np.uint8[H,W,3], "shape": [H,W,3], ...},
    "cam_left_wrist":  {"color": ..., "shape": ...},
    "cam_right_wrist": {"color": ..., "shape": ...},
  },
  "state": {
    "left_arm_joint_state":  np.float32[6],
    "right_arm_joint_state": np.float32[6],
    "left_ee_joint_state":   np.float32[1],
    "right_ee_joint_state":  np.float32[1],
    "left_ee_pose":          np.float32[7],   # world ee pose (xyzw quat)
    "right_ee_pose":         np.float32[7],
    ...  # 取决于 obs_config.robot（joint_states / world_ee_state）
  },
  "action": {},
  "data_format_version": "v1.0",
  "additional_info": {"frequency": 25},
  "instruction": "Pick up the bottles and throw them into the dustbin, ...",
  "env_idx": 0,
}
```

默认相机配置只有 RGB（`env_cfg/camera/camera_config.yml` 中 depth /
intrinsics / extrinsics 被注释）。要支持像素→世界反投影（back_project），
需要启用 `distance_to_image_plane_capture` 与相机标定输出（M2 改动点）。

## 4. 动作格式

机器人 `dual_x5`（双臂）：`arm_dim=[6,6]`，`ee_dim=[1,1]`，总 14 维
（`env_cfg/robot/_robot_info.json`）。

策略返回的 action dict（XPolicyLab `unpack_robot_state`）：

```python
# action_type=joint
{
    "left_arm_joint_state": [6],
    "right_arm_joint_state": [6],
    "left_ee_joint_state": [1],
    "right_ee_joint_state": [1],
}
# action_type=ee
{
    "left_ee_pose": [7],
    "right_ee_pose": [7],
    "left_ee_joint_state": [1],
    "right_ee_joint_state": [1],
}
```

eval client 转成 control_info 再喂 `apply_target`：

```python
# joint 路径
control_info = {
    "left_arm": {"position": joint_pos[6]},
    "right_arm": {"position": joint_pos[6]},
    "left_gripper": {"position": [val, val * mimic + shift]},  # ee 归一化→物理
    "right_gripper": {"position": [...]},
}
# ee 路径：对每个臂 robot_manager.solve_ik(target_pose=ee_pose)（CuRobo）
#   → {"status":"Success","joint_value":[...]} → control_info
```

`apply_target` 内部按 obs 频率做 8:2 插值后逐步执行（`_expand_control_info`）。

## 5. 终止与成功

- 步上限：任务类 `step_lim`（put_bottles_into_dustbin = 700）
- 成功：`reward_manager` 最终判定（`is_success` = reward > 0.999）
- put_bottles_into_dustbin：4 个瓶子（bottle0..3）+ 1 个 dustbin；
  成功谓词 `is_A_on_B_bottom(bottle, dustbin)`；gripper 需回到 open；
  指令："Pick up the bottles and throw them into the dustbin, using
  handover when needed."

## 6. 策略服务（Pi_05，ws 协议）

- 服务端：`XPolicyLab/setup_policy_server.py` 加载
  `XPolicyLab.policy.Pi_05.model.Model`，ws 端口由 deploy.yml 决定
- checkpoint：`RoboDojo-sim-arx_x5-joint-0`（seed-0 step 59999），
  symlink `XPolicyLab/policy/Pi_05/checkpoints → robodojo_runtime/checkpoints/Pi_05`
- 帧：`message_type / message_id / evaluation_id / action_case_id /
  trial_id / repeat_index / step / payload`（二进制 msgpack）
- 调用：`hello` → `prepare_case` → `reset` → `update_obs(obs)` →
  `get_action`（返回 action chunk list）→ ...
- 环境侧适配：`client_server/ws/model_client.py` 的 `WsModelClient`

## 7. M0 验证证据

- `scripts/run_pi05_official_eval_gate.sh debug` → **PASS**
  （Pi_05 服务加载、ws 连接、20 轮 fake-obs 推理）
- 官方 real gate（Isaac Sim + stack_bowls）在 2026-08-18 干净树状态下
  PASS（记录于 docs/），当前因用户 WIP 未提交改动未重跑
- M1 env_server 实测（2026-08-20）：
  - Isaac Sim headless 启动 ~30s 后 RPC 就绪
  - `reset / get_obs / get_status / is_success / apply_action / apply_target`
    全部可用；三相机 480×640 RGB、关节/末端状态、指令正确回传
  - joint 动作真实驱动右臂运动（joint0 -0.0 → 0.0441 rad），步数计数正常

## 8. 关键实现教训

**Isaac Sim 的渲染/捕获不是线程安全的**：`env.get_obs()` 必须在主线程调用，
在 RPC handler 线程里调用会永久阻塞（render/capture 死锁）。env_server 采用
请求队列：handler 线程只解析入队，主线程循环串行执行 env 操作并回填结果。

相机捕获还需要 `--enable_cameras` + `--kit_args "--enable
isaacsim.replicator.behavior --enable isaacsim.sensors.camera"`（Kit 透传），
否则 `capture_manager.step()` 同样阻塞。

## 9. 待办/风险

- 相机 depth/intrinsics 需启用（back_project 依赖）
- RoboDojo 源仓库当前有用户未提交改动（gates 会拒绝，不影响新增代码）
- Isaac Sim 启动慢（数十秒）、单进程单实例；RPent 侧需保活/复用策略

## 10. M5 端到端运行记录（2026-08-20）

`rpent --env robodojo --task put_bottles_into_dustbin --layout 1 --planner codex
--model deepseek-v4-flash --max-turns 30`：

- 全程 31 分钟、184 次工具调用、51 轮；Isaac Sim + Pi_05 + SAM3 三服务正常
- 感知：SAM3 找到瓶子、深度反投影定位（发现 back_project 的 -Z 深度符号
  问题，已在 tools.py 修复并在 memory 记录）
- 操作：右臂 pi0_pick 成功抓起黄色瓶子；右臂受工作空间限制够不到地面上的
  dustbin（x≈-0.63），交接失败；左臂策略把剩余瓶子打落桌面
- 结果：1/4 瓶子进桶，3 个丢失，700 步上限触发；arms 归位、夹爪打开后
  success 仍为 false → 诚实判定 failure，audit + finish 完成
- 产物：`logs/20260820-20:51:31_robodojo_put_bottles_into_dustbin_l1/`

### M5 暴露的真实问题（后续改进点）

1. **工作空间限制**：右臂从 home 到 x≈-0.63 的落点不可达，move_to 迭代
   也到不了 → 需要给 agent 更明确的"可达区域"提示（或换臂规划）。
2. **交接/双臂协作**：单臂策略在长距离搬运上不稳，任务本身要求 handover，
   但 agent 没有可靠的交接原语。
3. **策略后效**：pi0_pick 返回后策略仍继续动作，导致物体被扫落 → 需要
   更短的动作预算或显式的停止机制。
4. **文本模型感知开销**：纯文本 LLM 每步都要做 HSV/ASCII 分析，轮次消耗
   大 → 多模态模型会显著改善。

## 11. 视频录制（2026-08-20 补）

`env_server` 支持 `--video-dir`，每个相机输出一个 mp4：

```text
<video-dir>/episode_<ROBODOJO_RUN_ID>_cam_head.mp4
<video-dir>/episode_<ROBODOJO_RUN_ID>_cam_left_wrist.mp4
<video-dir>/episode_<ROBODOJO_RUN_ID>_cam_right_wrist.mp4
```

- 每次 `get_obs`（reset / 每个动作后）写入一帧；25 fps、640×480、mp4v。
- `close` / SIGTERM / SIGINT 时 flush 并关闭 writer。
- spec.py 已把 `--video-dir` 指到运行输出目录的 `videos/`。
- 实测：10 帧 / 3 相机 / 可解码。

## 12. v2 rollout（2026-08-20，含视频）

- 结果：failure（0/4 进桶，step 648/700）；三路 mp4 各 951 帧已产出
  （`<output_dir>/videos/`）。
- agent 发现并绕过的工具 bug：ee 动作分支忽略夹爪控制 → 已在 env_server
  修复（对齐 eval_env.take_action_batch）。
- 复现性结论：右臂硬边界 x≈0 够不到 x=-0.63 的 bin；`pi0_pick` 会选任意臂，
  交接过程易打落物体。

## 13. Reward/Score 明细（2026-08-20 补）

- 新增 RPC `get_reward_details`：返回逐项判定——每个 bottle 的
  `is_A_on_B_bottom`（底部贴合 dustbin 底部平面）、`grippers_open`、
  `arms_home`、当前 `score` 分档（10/25/40/100）、`reward` 与 `success`。
- env_server 在 reset 后注册任务 `get_score()` 分档（对齐官方 eval 流程），
  RPC reset 时重新注册。
- agent 侧新增只读工具 `get_reward_details`，audit 前应调用并记录 score。
- 背景：v2 rollout 的 audit 无法给出分数（agent 拿不到 reward 状态），
  此改进让每次运行都有客观 score。

## 14. move_to IK 发散修复（2026-08-21）

- 现象：固定姿态全位姿 IK 在低 z 侧向目标上发散（v2: y=-0.83；v3:
  z=1.35/y=-0.53），导致"无意义摆动"并吃掉步数预算。
- 修复：env_server 新增 `solve_ik_position` RPC（位置约束 + 多候选姿态，
  取关节位移最小解）；`move_to` 优先走该路径，失败回退旧 ee 路径。
- 实测：v3 发散的三个目标全部 <1cm 收敛、无摆动。

## 15. 安全监测与桶口原语（2026-08-21）

- **安全监测（env_server 内置）**：每次 apply 后检查各 bottle 位姿
  （env 内部状态，非 LLM 定位提示），`rolling`（速度 > 0.2 m/s）或
  `off_table`（超出桌面范围 / 低于桌面 0.15m）时产生告警；通过
  `get_safety_status` / 每个工具结果的 `safety` 块暴露。prompt 要求
  agent 见到告警先 `stabilize`（把最近的臂开到瓶前挡住）再继续任务。
- **桶口原语 `place_in_bin`**：携带物到桶口中心 → 下降到 rim 以下 →
  释放 → 回缩。解决 v5 中"在桶口高度释放、瓶子卡在桶后沿"的问题。
- **已知限制**：多候选姿态 IK 对 z≈0.9+ 的搬运路径与桶口（x=-0.63）
  可靠；对桌面中心低 z（z≈0.80）目标仍可能不可达/发散——该区域的操作
  应交给 Pi_05 策略（pi0_pick），而非脚本 move_to。
