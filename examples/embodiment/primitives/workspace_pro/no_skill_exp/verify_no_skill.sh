#!/bin/bash
# Audit a no-skill sweep: did the two isolations actually hold?
#
#   (A) PERCEPTION isolation  — SOLID check. Every state_*.json the worker saw
#       must carry `object_names` and NO `objects` coord block. This is enforced
#       by the driver (--hide_object_coords), so it is authoritative.
#   (B) NO-SKILL isolation    — BEST-EFFORT check. run_one_cell.sh runs the worker
#       with `--output-format text`, which captures only the worker's FINAL
#       narration, not a full tool-call trace. So we can only prose-scan
#       claude_<tag>.txt for tell-tale references to forbidden artifacts
#       (recipe_*.jsonl / prior audits). A clean scan is reassuring but NOT a
#       hard guarantee. For a hard guarantee, re-run the cell capturing every
#       tool call (see the "HARD AUDIT" note at the bottom) and grep that trace.
#
# Usage:
#   bash verify_no_skill.sh                         # scans ALL results/noskill_* dirs
#   bash verify_no_skill.sh results/noskill_object_swap_t2   # one dir
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY=/opt/venv/openpi/bin/python

DIRS=("$@")
if [ ${#DIRS[@]} -eq 0 ]; then
    mapfile -t DIRS < <(ls -d "$SCRIPT_DIR"/results/noskill_* 2>/dev/null)
fi
if [ ${#DIRS[@]} -eq 0 ]; then
    echo "no results/noskill_* dirs found under $SCRIPT_DIR/results — run a sweep first."
    exit 1
fi

# Forbidden artifacts the worker must never have read (Rule 5) — the recipe/
# results skill library AND the Claude Code project auto-memory (closed by
# CLAUDE_CODE_DISABLE_AUTO_MEMORY=1; this scan is a belt-and-suspenders check).
# The hard, authoritative check of the auto-memory channel is hard_audit_no_skill.sh
# (full tool-call trace); this text scan is best-effort (see header).
FORBIDDEN_RE='recipe_[a-z0-9_]*\.jsonl|results_object_pert|results_spatial_pert|results_goal_pert|results_10_pert|results_all_object_new|results_claude_p|multi_seed_exp|baseline_pi0_|\.claude_local/[^ ]*/memory/'

grand_total=0; grand_ok=0; leak_hits=0; coord_leaks=0
for dir in "${DIRS[@]}"; do
    [ -d "$dir" ] || continue
    echo "=== $dir ==="
    for audit in "$dir"/*.json; do
        [ -e "$audit" ] || continue
        case "$(basename "$audit")" in recipe_*) continue;; esac   # skip recipe jsonl if any .json
        tag=$(basename "$audit" .json)
        term=$("$PY" -c "import json;print(json.load(open('$audit')).get('libero_terminated'))" 2>/dev/null)
        grand_total=$((grand_total+1)); [ "$term" = "True" ] && grand_ok=$((grand_ok+1))

        # (A) perception coord-leak check on the workdir state files
        wd=/tmp/hybrid_repl_${tag}
        coord_note="state-files-gone"
        if ls "$wd"/state_*.json >/dev/null 2>&1; then
            if grep -l '"objects"' "$wd"/state_*.json >/dev/null 2>&1; then
                coord_note="COORD-LEAK!"; coord_leaks=$((coord_leaks+1))
            else
                coord_note="no-coords-ok"
            fi
        fi

        # (B) best-effort no-skill prose scan of the worker transcript
        ctxt="$dir/claude_${tag}.txt"
        skill_note="no-transcript"
        if [ -f "$ctxt" ]; then
            # Exclude the cell's OWN output filenames (the worker WRITES
            # recipe_<tag>.jsonl + <tag>.json and narrates that it did — that is
            # not a forbidden READ of a prior solution).
            own="recipe_${tag}\.jsonl|${tag}\.json"
            if grep -Ei "$FORBIDDEN_RE" "$ctxt" | grep -Eiv "$own" | grep -Eq .; then
                skill_note="POSSIBLE-RECIPE-REF (inspect $ctxt)"; leak_hits=$((leak_hits+1))
            else
                skill_note="clean-scan"
            fi
        fi
        printf "  %-32s term=%-5s  percep:%-12s  noskill:%s\n" "$tag" "$term" "$coord_note" "$skill_note"
    done
done

echo ""
echo "SUMMARY: $grand_ok/$grand_total solved | coord-leaks=$coord_leaks | possible-recipe-refs=$leak_hits"
if [ "$coord_leaks" -gt 0 ]; then
    echo "  ⚠ COORD LEAK: a state_*.json contained an 'objects' block — perception isolation BROKE."
fi
if [ "$leak_hits" -gt 0 ]; then
    echo "  ⚠ A transcript referenced a forbidden artifact — inspect it; the no-skill condition may be compromised."
fi
echo ""
echo "HARD AUDIT (only if you need certainty on no-skill): re-run a cell with a full"
echo "tool-call trace and grep it for Read/Bash hits on recipe_*.jsonl or results*/. e.g."
echo "  claude -p \"\$(cat <prompt>)\" --model claude-opus-4-7 --output-format stream-json --verbose \\"
echo "    --add-dir <workdir> --add-dir <memory_snapshot> --allowedTools 'Bash Read Write Glob Grep' \\"
echo "    > trace.jsonl 2>&1 ;  grep -E 'recipe_|results_.*_pert|multi_seed_exp' trace.jsonl"
