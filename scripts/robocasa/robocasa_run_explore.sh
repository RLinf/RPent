#!/usr/bin/env bash
# Drive ONE RoboCasa task with a claude -p explore agent (reset-based multi-attempt).
# Usage: robocasa_run_explore.sh <TASK> <GPU> [MAX_TURNS]
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(cd "$REPO_ROOT/.." && pwd)"
cd "$REPO_ROOT"
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl} NO_ALBUMENTATIONS_UPDATE=1 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SIM_PY="$PARENT/.venv_pa_robocasa/bin/python"

TASK=${1:?usage: robocasa_run_explore.sh <TASK> <GPU> <SEED> [AGENT_TIMEOUT_S]}
GPU=${2:?usage: robocasa_run_explore.sh <TASK> <GPU> <SEED> [AGENT_TIMEOUT_S]}
SEED=${3:-0}
AGENT_TIMEOUT=${4:-1800}   # wall-clock budget for the explore agent (seconds)
TAG="${TASK}_s${SEED}"
# PROMPT_FILE / OUT_BASE overridable -> e.g. no-reset variant writing to a separate dir
PROMPT_FILE="${PROMPT_FILE:-$REPO_ROOT/resources/robocasa/agent_task_prompt_robocasa_explore.md}"
OUT_BASE="${OUT_BASE:-$PARENT/logs}"
# reset (episode restart) is allowed ONLY in EXPLORE mode (the *_explore prompt). The
# no-reset / matched launchers override PROMPT_FILE to the non-explore prompt -> reset
# stays disabled (driver default RLDX_ALLOW_RESET=0). Tie the switch to the prompt so it
# can never accidentally be on during a multi-seed/matched eval.
case "$PROMPT_FILE" in *explore*) export RLDX_ALLOW_RESET=1 ;; *) export RLDX_ALLOW_RESET=0 ;; esac
WD="/tmp/explore_${TASK}_s${SEED}"
OUT="$OUT_BASE/${TASK}"
# RUN_SUBDIR overrides the per-run output layout. When set (e.g. "20260705-1612-SteamInMicrowave"),
# ALL durable artifacts (audit json, run_logs, images, videos) go under $OUT_BASE/$RUN_SUBDIR/
# instead of $OUT_BASE/$TASK/. Lets run_robocasa.sh request a "date-task" folder per user preference.
if [ -n "${RUN_SUBDIR:-}" ]; then
  OUT="$OUT_BASE/$RUN_SUBDIR"
fi
rm -rf "$WD"; mkdir -p "$WD" "$OUT/attempts/$TAG"

# OPTIONAL video capture: when RLDX_VIDEO=1 (or RLDX_VIDEO_DIR set), tell the VLA skill
# to write one mp4 per rldx_skill call into the workdir. Exported so the driver child
# (and rldx_skill inside it) inherit it. Videos are archived to run_logs alongside images.
if [ "${RLDX_VIDEO:-0}" = "1" ] && [ -z "${RLDX_VIDEO_DIR:-}" ]; then
  export RLDX_VIDEO_DIR="$WD/videos"
fi

echo "[explore:$TASK s$SEED] starting driver on GPU$GPU (workdir=$WD)"
CUDA_VISIBLE_DEVICES=$GPU "$SIM_PY" "$REPO_ROOT/cli/main_robocasa.py" \
  --env "$TASK" --split target --seed "$SEED" --workdir "$WD" > "$WD/driver.log" 2>&1 &
DRIVER_PID=$!
# Kill the driver on normal exit or on TERM/INT. (SIGKILL of THIS script can't run a
# trap, but the driver also self-installs PR_SET_PDEATHSIG so the kernel reaps it the
# moment this parent dies — belt and suspenders against GPU-hogging orphans.)
trap "kill $DRIVER_PID 2>/dev/null" EXIT TERM INT

# wait for state_00 (driver loaded env + dumped initial state)
for i in $(seq 1 60); do
  [ -f "$WD/done_00.flag" ] && break
  kill -0 $DRIVER_PID 2>/dev/null || { echo "[explore:$TASK] driver died"; tail -20 "$WD/driver.log"; exit 1; }
  sleep 3
done
[ -f "$WD/done_00.flag" ] || { echo "[explore:$TASK] driver never produced state_00"; exit 1; }
echo "[explore:$TASK] driver ready; launching claude -p"

# fill the prompt template
PROMPT="$(sed -e "s#{TASK}#$TASK#g" -e "s#{WORKDIR}#$WD#g" -e "s#{OUTPUT_DIR}#$OUT#g" \
              -e "s#{TAG}#$TAG#g" -e "s#{AGENT_TIMEOUT}#$AGENT_TIMEOUT#g" \
              "$PROMPT_FILE")"

cd "$WD"
echo "$PROMPT" > "$WD/_prompt.md"
# Run claude -p STANDALONE: unset the interactive-session vars so it connects to
# the real API directly (a child-session would route through the parent harness's
# local SSE gateway and get ConnectionRefused). No --max-turns: the agent
# iterates (reset-based) until it solves or the wall-clock budget runs out.
env -u CLAUDE_CODE_SSE_PORT -u CLAUDE_CODE_CHILD_SESSION -u CLAUDE_CODE_ENTRYPOINT \
    -u CLAUDECODE -u CLAUDE_CODE_SESSION_ID -u AI_AGENT \
    CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 DISABLE_AUTOUPDATER=1 \
  timeout --kill-after=15 "$AGENT_TIMEOUT" \
    claude -p "$(cat "$WD/_prompt.md")" \
      --model claude-opus-4-8 \
      --add-dir "$WD" --add-dir "$OUT" --add-dir "$REPO_ROOT/resources/robocasa" \
      --allowedTools "Bash Read Write Glob Grep" \
      --output-format stream-json --verbose \
      --permission-mode acceptEdits \
      > "$WD/agent.log" 2>&1
CC_RC=$?
echo "[explore:$TASK s$SEED] claude -p finished (exit=$CC_RC)"
# Archive the FULL primitive-level logs to a permanent dir before the workdir is
# wiped on the next run of this cell. Keeps log_NN.json (every primitive call+result),
# agent.log (claude stream-json reasoning), state_*.json, success_criteria.md.
RUNLOG="$OUT/run_logs/$TAG"
mkdir -p "$RUNLOG"
cp "$WD"/log_*.json "$RUNLOG"/ 2>/dev/null
cp "$WD"/state_*.json "$RUNLOG"/ 2>/dev/null
cp "$WD"/agent.log "$RUNLOG"/ 2>/dev/null
cp "$WD"/success_criteria.md "$RUNLOG"/ 2>/dev/null
# Archive the per-command observation snapshots (main + wrist camera PNGs) so the durable
# record includes imagery, not just json. These are the low-res (256²) command-boundary
# frames; the heavy 1024² hi-res PNGs are still cleaned below.
cp "$WD"/image_cam_*.png "$RUNLOG"/ 2>/dev/null
cp "$WD"/camera_meta_*.json "$RUNLOG"/ 2>/dev/null
# Archive the per-command rollout videos (if RLDX_VIDEO enabled) into run_logs/videos.
if ls "$WD"/videos/*.mp4 >/dev/null 2>&1; then
  mkdir -p "$RUNLOG/videos"
  cp "$WD"/videos/*.mp4 "$RUNLOG/videos"/ 2>/dev/null
  echo "[explore:$TASK s$SEED] archived $(ls $RUNLOG/videos/*.mp4 2>/dev/null | wc -l) rollout videos -> $RUNLOG/videos"
fi
echo "[explore:$TASK s$SEED] archived $(ls $RUNLOG/log_*.json 2>/dev/null | wc -l) primitive logs -> $RUNLOG"
# Fallback audit: if the agent didn't write one, synthesize from the final state
# so the cell is marked attempted (resume won't loop it forever).
if [ ! -f "$OUT/${TAG}.json" ]; then
  "$SIM_PY" - "$WD" "$OUT/${TAG}.json" "$TASK" "$SEED" "$CC_RC" <<'PY'
import json, glob, sys
wd, out, task, seed, rc = sys.argv[1:6]
fs = sorted(glob.glob(f"{wd}/state_*.json"))
s = json.load(open(fs[-1])) if fs else {}
json.dump({"suite":"robocasa365","task":task,"seed":int(seed),
           "success":bool(s.get("success")),"steps":len(fs),
           "strategy_notes":f"agent exited rc={rc} without writing audit; fallback from final state",
           "source":"fallback"}, open(out,"w"), indent=2)
PY
fi
echo "[explore:$TASK s$SEED] audit: $(ls $OUT/${TAG}.json 2>/dev/null || echo none)  attempts: $(ls $OUT/attempts/$TAG/ 2>/dev/null | wc -l)"
"$SIM_PY" -c "import json,glob; fs=sorted(glob.glob('$WD/state_*.json')); s=json.load(open(fs[-1])) if fs else {}; print('[explore:$TASK s$SEED] final success =', s.get('success'))" 2>/dev/null
# AUTO-CLEANUP (disk): delete ONLY the heavy npy dumps + the 1024² hi-res PNGs — the
# ~2-3GB/cell bloat that filled the container root. Do this ONLY AFTER VERIFYING the
# durable record is safely archived to run_logs, so the command sequence / recipe is
# NEVER lost. The npy (world_hi 6MB + world/depth/nav) and image_cam_hi PNGs were only
# needed LIVE during the run; the RECIPE is in log_*.json (every primitive call+result)
# + state_*.json + agent.log + the audit — all copied to run_logs and to explore_results.
NLOG_WD=$(ls "$WD"/log_*.json 2>/dev/null | wc -l)
NLOG_ARCH=$(ls "$RUNLOG"/log_*.json 2>/dev/null | wc -l)
if [ "$NLOG_ARCH" -ge "$NLOG_WD" ] && [ -f "$OUT/${TAG}.json" ]; then
  freed=$(du -shc "$WD"/*.npy "$WD"/image_cam_hi_*.png 2>/dev/null | tail -1 | awk '{print $1}')
  rm -f "$WD"/*.npy "$WD"/image_cam_hi_*.png 2>/dev/null
  echo "[explore:$TASK s$SEED] cleaned npy + hi-res PNG (archived $NLOG_ARCH/$NLOG_WD cmd logs; freed ~${freed:-0})"
else
  echo "[explore:$TASK s$SEED] SKIP cleanup — archive incomplete ($NLOG_ARCH/$NLOG_WD logs, audit=$([ -f "$OUT/${TAG}.json" ]&&echo yes||echo no)); keeping npy to be safe"
fi
