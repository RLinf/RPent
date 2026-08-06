# 从零重跑 SteamInMicrowave s3 实验

删掉当前 venv → 重新 setup → 跑通 `SteamInMicrowave` seed3 的完整命令流程。

---

## 第 1 步:删掉当前 venv + 重新 setup(15~30 分钟,后台跑)

```bash
cd /mnt/public/zhuchunyang_rl/pa_robocasa_work/PhysicalAgent_omo

# 显式删掉当前 venv(setup_robocasa.sh 内部也会 rm -rf 重建, 这里先手动删干净)
rm -rf /mnt/public/zhuchunyang_rl/pa_robocasa_work/.venv_pa_robocasa

# 后台跑 setup + 长超时, 日志留档
bash scripts/setup_robocasa.sh | tee /mnt/public/zhuchunyang_rl/pa_robocasa_work/logs/setup_$(date +%Y%m%d-%H%M%S).log
```

看到结尾出现 `DEPLOYMENT SUCCESSFUL!` 即完成。

第 2 步:确认 GPU 空闲(关键 —— 之前 OOM 就栽在这)

nvidia-smi

VLA 权重常驻 ~15.4G,24G 卡上必须几乎整卡空闲才不 OOM。若看到 vla_server.py 之类占用进程,需先停掉它再跑(上次停的是 PhysicalAgent 的 vla_server.py,它至今没重启)。

## 第 3 步:跑 SteamInMicrowave seed3(2000s,含视频+图片)

```bash
cd /mnt/public/zhuchunyang_rl/pa_robocasa_work/PhysicalAgent_omo

# 时间戳只算一次, 保证日志目录名和 nohup 主日志用的是同一个时间
TS=$(date +%Y%m%d-%H%M%S)
RUN_SUBDIR="${TS}_run_SteamInMicrowave_s3"
RUN_DIR="/mnt/public/zhuchunyang_rl/pa_robocasa_work/logs/$RUN_SUBDIR"
mkdir -p "$RUN_DIR"   # 先建好, 否则 nohup 重定向会因目录不存在而失败

RUN_SUBDIR="$RUN_SUBDIR" \
OUT_BASE=/mnt/public/zhuchunyang_rl/pa_robocasa_work/logs \
  bash scripts/run_robocasa.sh SteamInMicrowave 0 3 2000 \
  | tee "$RUN_DIR/run.log" 2>&1
```

`run_robocasa.sh <TASK> <GPU> <SEED> <TIMEOUT>` — 参数:任务 `SteamInMicrowave`、GPU `0`、
seed `3`、超时 `2000`s。视频/图片开关(`RLDX_VIDEO=1`)是 run_robocasa.sh 的默认行为;
`RUN_SUBDIR` 显式覆盖成 `<时间>_run_SteamInMicrowave_s3` 格式,
所有产物(含 nohup 主日志 `run.log`)都落在这一个目录下。

产物落在(`TAG = <TASK>_s<SEED> = SteamInMicrowave_s3`):

```
/mnt/public/zhuchunyang_rl/pa_robocasa_work/logs/<时间>_run_SteamInMicrowave_s3/
├── run.log                          ← nohup 主日志(run_robocasa.sh + explore 进度)
├── SteamInMicrowave_s3.json         ← audit
└── run_logs/SteamInMicrowave_s3/
    ├── videos/cmd_NN.mp4            ← rollout 视频
    └── image_cam_*.png              ← 各命令快照
```
