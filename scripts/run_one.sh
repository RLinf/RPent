#!/bin/bash
# Run ONE task end-to-end with the Chinese dashboard, record it (Playwright
# webm -> ffmpeg mp4) at 1080p, and leave a single dashboard.mp4 in the task's
# output dir. The agent tree is launched under `setsid` so it can be killed
# whole (agent + env_server + claude subprocess) once the recording is done.
#
#   run_one.sh <task> <seed> <out_dir> [rec_dur_s] [display_tag]
set -u
TASK="$1"; SEED="$2"; OUT="$3"; REC_DUR="${4:-300}"
PYBIN=/mnt/public/zhuchunyang_rl/.venv_agenticvla/bin/python
REPO=/mnt/public/zhuchunyang_rl/PhysicalAgent
cd "$REPO"
export PYTHONPATH=/mnt/public/zhuchunyang_rl/rlinf_libero_camera_meta:"$REPO":/mnt/public/zhuchunyang_rl/.venv_agenticvla/libero:/mnt/public/zhuchunyang_rl/.venv_agenticvla/libero_pro:/mnt/public/zhuchunyang_rl/.venv_agenticvla/libero_plus:
export ANTHROPIC_BASE_URL=https://cloud.infini-ai.com/maas
export ANTHROPIC_AUTH_TOKEN=sk-df5f6pi4wpctb2ft
mkdir -p "$OUT"

setsid nohup "$PYBIN" "$REPO/cli/main.py" \
  --dashboard --dashboard_language zh-cn --dashboard-host 127.0.0.1 \
  --vla-endpoint http://127.0.0.1:50008 \
  --suite libero_object_task --task "$TASK" --seed "$SEED" \
  --cerebrum claude_code --model deepseek-v4-pro \
  --max-turns 100 --max-tokens 4096 --output-dir "$OUT" \
  >"$OUT/run.log" 2>&1 &
AGENT_PID=$!

# Wait for the dashboard URL.
URL=""
for _ in $(seq 1 120); do
  URL=$(grep -oE 'Dashboard: http://127\.0\.0\.1:[0-9]+' "$OUT/run.log" 2>/dev/null | head -1)
  [ -n "$URL" ] && break
  sleep 0.5
done
if [ -z "$URL" ]; then
  echo "[one] task=$TASK seed=$SEED: no dashboard URL" >> "$OUT/record.log"
  kill -9 "-$AGENT_PID" 2>/dev/null
  exit 1
fi
URL="${URL#Dashboard: }"
echo "[one] task=$TASK seed=$SEED dashboard=$URL" >> "$OUT/record.log"

# Drive + record at 1080p.
"$PYBIN" -u "$REPO/scripts/record_dashboard.py" "$URL" "$OUT" "$REC_DUR" 1080 \
  >>"$OUT/record.log" 2>&1

# Convert webm -> mp4 (every frame kept), drop the webm so only one file stays.
if [ -f "$OUT/dashboard.webm" ]; then
  ffmpeg -y -hide_banner -loglevel warning -i "$OUT/dashboard.webm" \
    -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p "$OUT/dashboard.mp4" \
    >>"$OUT/record.log" 2>&1
  rm -f "$OUT/dashboard.webm"
fi

# Kill the whole agent tree (session leader = agent process group id).
kill -9 "-$AGENT_PID" 2>/dev/null
OUTPAT="${OUT//\//\\/}"
pkill -9 -f "env_server.py.*${OUTPAT}" 2>/dev/null
echo "[one] task=$TASK seed=$SEED FINISHED mp4_bytes=$(stat -c%s "$OUT/dashboard.mp4" 2>/dev/null)" >> "$OUT/record.log"
