#!/bin/bash
# HARD AUDIT of the no-skill isolation — runs ONE cell with a FULL tool-call
# trace (--output-format stream-json --verbose), then lists every file the
# worker actually opened (Read/Glob/Grep targets + Bash commands) and flags any
# that touch a forbidden artifact (recipe_*.jsonl / prior audits / results dirs).
# Unlike verify_no_skill.sh (which prose-scans the text transcript), this sees
# EVERY tool call, so a clean result is proof — not best-effort.
#
# Usage:
#   SUITE=libero_object_swap TASK=2 SEED=7 GPU=3 bash hard_audit_no_skill.sh
set -u
SUITE=${SUITE:-libero_object_swap}
TASK=${TASK:-2}
SEED=${SEED:-7}
GPU=${GPU:-3}
MODEL=${MODEL:-claude-opus-4-7}
CELL_TIMEOUT_S=${CELL_TIMEOUT_S:-1200}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-2000}
# 1 => disable the Claude Code project auto-memory channel (true no-skill).
# 0 => leave it ON, to REPRODUCE the auto-memory leak for comparison.
DISABLE_AUTO_MEMORY=${DISABLE_AUTO_MEMORY:-1}
[ "$DISABLE_AUTO_MEMORY" = 1 ] && export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1

REPO=/mnt/public2/zhangyixian/RLinf_agentic
PERT=$REPO/examples/embodiment/primitives/workspace_pro
NS=$PERT/no_skill_exp
PY=/opt/venv/openpi/bin/python
MEMORY_DIR=$PERT/memory_snapshot
PROMPT_TEMPLATE=$NS/agent_task_prompt_no_skill.md

TAG=${SUITE/libero_/}_t${TASK}_s${SEED}
WORKDIR=/tmp/hybrid_repl_audit_${TAG}
OUTDIR=$NS/results/hard_audit
TRACE=$OUTDIR/trace_${TAG}.jsonl
DRIVER_LOG=/tmp/cc_driver_audit_${TAG}.log
mkdir -p "$OUTDIR"

echo "[$(date +%T)] [AUDIT $TAG] starting driver on GPU $GPU"
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"
LIBERO_TYPE=pro CUDA_VISIBLE_DEVICES=$GPU \
    "$PY" "$REPO/examples/embodiment/primitives/interactive_driver.py" \
    --suite "$SUITE" --task "$TASK" --seed "$SEED" \
    --workdir "$WORKDIR" --max_episode_steps "$MAX_EPISODE_STEPS" \
    --hide_object_coords --always_render \
    > "$DRIVER_LOG" 2>&1 &
DRIVER_PID=$!

echo "[$(date +%T)] [AUDIT $TAG] waiting for state_00.json (pid=$DRIVER_PID)"
T0=$(date +%s)
while [ ! -f "$WORKDIR/state_00.json" ]; do
    sleep 5
    kill -0 "$DRIVER_PID" 2>/dev/null || { echo "driver died before ready"; tail -30 "$DRIVER_LOG"; exit 2; }
    [ $(( $(date +%s) - T0 )) -gt 300 ] && { echo "driver not ready in 300s"; kill -9 "$DRIVER_PID"; exit 3; }
done
echo "[$(date +%T)] [AUDIT $TAG] driver ready in $(( $(date +%s) - T0 ))s"

PROMPT_FILE=$(mktemp /tmp/audit_prompt_${TAG}.XXXXXX.md)
sed -e "s|{SUITE}|$SUITE|g" -e "s|{TASK}|$TASK|g" -e "s|{SEED}|$SEED|g" \
    -e "s|{WORKDIR}|$WORKDIR|g" -e "s|{TAG}|$TAG|g" -e "s|{OUTPUT_DIR}|$OUTDIR|g" \
    "$PROMPT_TEMPLATE" > "$PROMPT_FILE"

echo "[$(date +%T)] [AUDIT $TAG] invoking claude -p with FULL stream-json trace -> $TRACE"
cd "$REPO"
timeout --kill-after=15 "$CELL_TIMEOUT_S" \
    claude -p "$(cat "$PROMPT_FILE")" \
        --model "$MODEL" \
        --output-format stream-json --verbose \
        --add-dir "$WORKDIR" --add-dir "$MEMORY_DIR" \
        --allowedTools "Bash Read Write Glob Grep" \
        --max-budget-usd 10 \
        > "$TRACE" 2>&1
echo "[$(date +%T)] [AUDIT $TAG] claude -p finished rc=$? (trace lines: $(wc -l < "$TRACE"))"
rm -f "$PROMPT_FILE"
echo '{"action": "exit"}' > "$WORKDIR/command.json"; sleep 2; kill -9 "$DRIVER_PID" 2>/dev/null || true

# ── analyze the trace: every file the worker touched ──
echo ""; echo "================ TOOL-CALL AUDIT ($TAG) ================"
"$PY" - "$TRACE" "$WORKDIR" "$TAG" <<'PY'
import json, sys, re
trace, workdir, tag = sys.argv[1], sys.argv[2], sys.argv[3]
# Forbidden channel 1: the recipe / results skill library.
FORBIDDEN = re.compile(r'recipe_[a-z0-9_]*\.jsonl|results_object_pert|results_spatial_pert|'
                       r'results_goal_pert|results_10_pert|results_all_object_new|'
                       r'results_claude_p|multi_seed_exp|baseline_pi0_', re.I)
# Forbidden channel 2: the Claude Code project AUTO-memory (per-cell solved
# experience). NOTE: the curated workspace_pro/memory_snapshot/ is ALLOWED and
# does NOT match this (it has no .claude_local path component).
AUTOMEM = re.compile(r'\.claude_local/.*?/memory/', re.I)
OWN = re.compile(rf'recipe_{re.escape(tag)}\.jsonl|{re.escape(tag)}\.json')
def is_forbidden(it):
    return (FORBIDDEN.search(it) or AUTOMEM.search(it)) and not OWN.search(it)
reads, bash_cmds, globs, greps, writes = [], [], [], [], []
n_tool = 0
for line in open(trace, errors='ignore'):
    line=line.strip()
    if not line or line[0] != '{':
        continue
    try: ev=json.loads(line)
    except Exception: continue
    msg=ev.get('message') or {}
    for blk in (msg.get('content') or []):
        if not isinstance(blk,dict) or blk.get('type')!='tool_use': continue
        n_tool+=1
        name=blk.get('name'); inp=blk.get('input') or {}
        if name=='Read':  reads.append(inp.get('file_path',''))
        elif name=='Write': writes.append(inp.get('file_path',''))
        elif name=='Bash': bash_cmds.append(inp.get('command',''))
        elif name=='Glob': globs.append(f"{inp.get('path','.')} :: {inp.get('pattern','')}")
        elif name=='Grep': greps.append(f"{inp.get('path','.')} :: {inp.get('pattern','')}")

def scan(label, items, field_is_cmd=False):
    hits=[it for it in items if is_forbidden(it)]
    print(f"\n[{label}] {len(items)} calls" + (f"  — {len(hits)} FORBIDDEN" if hits else "  — clean"))
    for h in hits:
        kind = "AUTO-MEMORY" if AUTOMEM.search(h) else "RECIPE/RESULTS"
        print(f"   ⚠ FORBIDDEN ({kind}):", h[:200])
    return hits

print(f"total tool calls parsed: {n_tool}")
h  = scan("Read", reads)
h += scan("Glob", globs)
h += scan("Grep", greps)
h += scan("Bash", bash_cmds)
# show what it DID read (non-forbidden) for transparency
print("\n--- distinct files Read (allowed) ---")
for f in sorted(set(reads)):
    if not is_forbidden(f):
        print("   ", f)
print("\n--- files Written (its own outputs) ---")
for f in sorted(set(writes)): print("   ", f)
print("\n========================================================")
if h:
    nr=sum(1 for x in h if not AUTOMEM.search(x)); na=len(h)-nr
    print(f"RESULT: ❌ ISOLATION BROKEN — {len(h)} forbidden access(es) "
          f"(recipe/results={nr}, auto-memory={na}). The worker READ skill experience.")
    sys.exit(1)
else:
    print("RESULT: ✅ ISOLATION HELD — the worker made ZERO accesses to BOTH channels: no "
          "recipe_*.jsonl / prior results-audit, AND no Claude Code project auto-memory "
          "(.claude_local/.../memory/). It solved without skill experience.")
PY
