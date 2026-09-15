RPent × RoboDojo 集成全流程记录
================================

记录周期：2026-08-20 ～ 2026-08-21。工作区 ``/home/admin/rpent-robodojo-ws``
（原仓库 ``/home/admin/RPent`` 保持干净）。

背景与目标
----------

把 RoboDojo（Isaac Sim / IsaacLab、双臂 ARX-X5、Pi_05 策略）接入 RPent
作为可插拔环境后端，让 LLM 编排的 harness 在更丰富、更难的 benchmark 上
与裸策略做同布局 A/B 对照。

工作区原则
----------

* 第三方源码（RoboDojo / XPolicyLab）只读，改动全部落在 rpent 工作区；
* 每个里程碑留下证据（视频 / reward_details / transcript）。

里程碑（M0–M5 + 补丁）
----------------------

* M0 摸底：观测 / 动作 / 终止 / 启动契约、Pi_05 WebSocket 协议；
* M1–M3：观测、脚本运动、真实 Pi_05 抓取闭环通过；
* M4：三相机逐相机 mp4 录制；夹爪归一化映射修复；
* M5：reward 明细（逐物体谓词 + 官方分数档）接入 agent 工具；
* 补丁：``move_to`` 改为位置约束多候选姿态 IK（消除固定姿态 IK 发散）、
  ee 动作回退分支修复、安全监测（rolling / off-table + ``stabilize``）、
  ``place_in_bin`` 桶口放置原语、``--random`` 官方口径随机布局。

关键实现教训
------------

#. Isaac Sim 渲染必须在主线程（RPC 层用请求队列）；
#. 相机捕获需要 replicator / sensor 扩展与 ``--enable_cameras``；
#. ``back_project`` 深度符号按 Isaac 相机 -Z 约定；
#. 夹爪标度 1.0=开 / 0.0=合（与 reward 约定一致）；
#. 固定姿态全位姿 IK 在低 z 侧向目标上发散，位置约束多候选姿态收敛；
#. 安全监测把进桶物体与滚出桌面区分开，避免误报。

Rollout 记录
------------

* v3（layout 1/2/3 + 随机布局）：``put_bottles_into_dustbin`` 均 100/100
  （单局抽检）；
* v6–v8：安全机制实战拦截滚动瓶子、随机场景 6 次 reset 得到 6 个不同布局；
* A/B（同 layout，裸 Pi_05 官方 eval client vs harness）：

  * ``stack_bowls``（generalization）：裸策略失败布局 2/3 被 harness 翻转为
    100；
  * ``fill_pen_holder``（long-horizon）：三对持平（10/40/25），定位真实瓶颈
    为垂直插入精度而非时序；
  * ``swap_blocks``（memory）：裸跑 10 集全 0，harness 首轮受夹爪语义 bug
    干扰未完成（已修复）。

以上为指示性小样本结果，非官方 50 集完整评测。

提交与证据
----------

* 提交记录：基于上游 ``3fc7586`` 的 29 个提交（``feat/robodojo-integration``）；
* 证据索引：``robots/robodojo/guides/interface.md`` 记录接口规范；
  rollout 产物与 audit 见工作区 ``evidence/``。

已知问题与下一步
----------------

* ``handover`` 未实现（当前靠策略原生投掷绕开）；
* 低 z 桌面级脚本 IK 仍可能不可靠，交策略处理；
* 纯文本 LLM 感知开销大（多模态 planner 预期降低感知轮次）；
* swap_blocks 重试、Open 类任务与大样本统计评测留作后续工作。
