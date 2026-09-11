RoboDojo A/B 对照协议：裸 Pi_05 vs RPent Harness
=================================================

目的
----

量化 RPent harness（agent 循环 + 工具链 + 安全监测 + 记忆）在 RoboDojo
任务上相对裸 Pi_05 策略的增量。裸 Pi_05 是官方评估方式：Isaac Sim 把观测
经 WebSocket 发给 Pi_05 策略服务，策略直接返回动作 chunk，闭环到 episode
结束，无 LLM、无工具、无安全监测。

场景对齐（关键机制）
--------------------

两边使用同一批官方 layout 文件，保证场景逐 episode 一致：

* 官方 layout 文件：``<robodojo-modelscope>/Assets/Eval_Layout/RoboDojo/
  <env_cfg>/<seed>/<task>_<N>.json``（N 即 layout id，0..54）；
* 裸跑侧：``eval_env.py`` 支持 ``ROBODOJO_LAYOUT_IDS="N1,N2,..."`` 白名单，
  只评估这些 layout；
* Harness 侧：``env_server.py --layout N`` → ``env.reset(seed=N)`` →
  ``seed_manager.get_seed_scene_info(N)`` → 同一个 layout 文件。

因此 ``ROBODOJO_LAYOUT_IDS="0,1,2"`` 裸跑等价于 harness 依次
``--layout 0/1/2``。

标准命令
--------

裸 Pi_05 基线（官方 eval client）：

.. code-block:: bash

   cd <RoboDojo 源码目录>
   ROBODOJO_LAYOUT_IDS="0,1,2,3,4,5,6,7,8,9" \
     bash scripts/robodojo.sh eval \
       --policy-dir XPolicyLab/policy/Pi_05 \
       --task <TASK> --ckpt RoboDojo-sim-arx_x5-joint-0 \
       --policy-env uv --eval-env <sim env> \
       --action-type joint --seed 0 --policy-gpu 0 --env-gpu 0 --eval-num <N>

结果在 ``eval_result/RoboDojo/<TASK>/Pi_05/arx_x5/.../_result.json``
（success_rate / score / 逐 episode layout_id + score），另有每 episode
三路相机 mp4。

Harness（RPent agent）：

.. code-block:: bash

   cd <rpent checkout>
   export PYTHONPATH="$PWD" PATH="$PWD/.venv/bin:$PATH" \
     SAM3_CHECKPOINT_PATH=$PWD/checkpoints/sam3/sam3.pt \
     HF_HUB_DISABLE_XET=1 CELL_TIMEOUT_S=3600
   rpent --env robodojo --task <TASK> --layout <N> \
     --sim-device 0 --planner codex --model deepseek-v4-flash --max-turns 30 \
     --output-dir <ws>/evidence/<run_tag>

结果在 ``--output-dir`` 下：``audit*.json``（含 reward_details / score）、
``videos/*.mp4``、各服务日志。

报告口径
--------

每个任务出一张表（至少 10 裸 episode 对照，harness 至少覆盖同一批
layout 的子集并注明）：

.. code-block:: text

   | layout | 裸 Pi_05 score | harness score | 裸 Pi_05 失败模式 | harness 增量 |

判定规则：

* score 逐 episode 对比（官方 reward 分档，如 put_bottles 0/25/40/100）；
* success 判定以官方 reward（``is_success`` / score>=1.0）为准，不采用
  agent 自述；
* 小样本（<=10）只作指示性结论；正式结论需 50-episode 官方协议。

已知基线（本地官方裸跑记录）
----------------------------

``put_bottles_into_dustbin``（standard，seed 0）：

.. list-table::
   :header-rows: 1

   * - 规模
     - SR
     - score
   * - 45
     - 64.4%
     - 75.1
   * - 16
     - 43.8%
     - 64.4
   * - 9
     - 66.7%
     - 76.7

``stack_bowls_random``：

.. list-table::
   :header-rows: 1

   * - 规模
     - SR
     - score
   * - 300
     - 6.7%
     - 15.9
   * - 30
     - 16.7%
     - 24.2

注：裸跑小样本方差极大（10 集可 0 或 100），对比必须锁同一批 layout。

当前进度
--------

见 :doc:`integration_log`（A/B 验证记录）。
