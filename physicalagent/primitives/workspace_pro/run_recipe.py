#!/usr/bin/env python3
"""Drive the interactive REPL with commands from a JSONL recipe.

Usage:
    python run_recipe.py /path/to/recipe.jsonl [--workdir /tmp/hybrid_repl]

Reads each JSON-line command, writes it to {workdir}/command.json, waits for
done_NN.flag, prints a one-line summary, then proceeds. Used for libero_object
PRO-cell runs where recipes are pre-derived and we just want to replay.
"""
import argparse, json, os, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe", type=str)
    ap.add_argument("--workdir", default="/tmp/hybrid_repl")
    ap.add_argument("--start_step", type=int, default=1,
                    help="step number to start writing commands at (1-indexed)")
    args = ap.parse_args()

    cmds = []
    with open(args.recipe) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cmds.append(json.loads(line))

    print(f"[runner] loaded {len(cmds)} commands from {args.recipe}")

    for i, cmd in enumerate(cmds):
        n = args.start_step + i
        nn = f"{n:02d}"
        # Drop the "note" field — env worker ignores it but cleaner this way.
        cmd_clean = {k: v for k, v in cmd.items() if k != "note"}
        cmd_path = os.path.join(args.workdir, "command.json")
        with open(cmd_path, "w") as f:
            json.dump(cmd_clean, f)
        action = cmd.get("action", "?")
        print(f"[runner] step {nn}: {action} ...", flush=True)
        flag = os.path.join(args.workdir, f"done_{nn}.flag")
        t0 = time.time()
        while not os.path.exists(flag):
            time.sleep(0.5)
            if time.time() - t0 > 600:
                print(f"[runner] TIMEOUT waiting for {flag}", flush=True)
                sys.exit(2)
        # summarize result
        log_path = os.path.join(args.workdir, f"log_{nn}.json")
        try:
            log = json.load(open(log_path))
            r = log.get("result", {})
            term = r.get("libero_terminated", False)
            extras = []
            if action == "move_to":
                extras.append(f"dist={r.get('final_dist_m'):.4f}")
                extras.append(f"steps={r.get('steps_used')}")
            elif action == "pi0_pick":
                extras.append(f"chunks={r.get('chunks_used')}")
                extras.append(f"peak_lift={r.get('peak_lift_m'):.4f}")
            elif action == "release":
                extras.append(f"grip_open={r.get('final_gripper_opening'):.4f}")
            print(f"[runner] step {nn}: {action} done in {time.time()-t0:.1f}s  "
                  f"libero_term={term}  " + "  ".join(extras), flush=True)
            if term:
                print(f"[runner] libero_terminated=True at step {nn}; stopping cleanly")
                break
        except Exception as e:
            print(f"[runner] step {nn}: could not parse log: {e}", flush=True)

    print("[runner] done")


if __name__ == "__main__":
    main()
