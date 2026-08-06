"""Phases 1-4 — RoboCasa LLM-in-the-loop interactive driver.

File-protocol driver (port of LIBERO interactive_driver.py) on top of the
Phase-0 RoboCasaInteractiveEnv adapter:

  agent WRITES {workdir}/command.json  (one primitive)
  driver consumes -> dumps state_NN.json + image_cam_NN.png + depth_NN.npy
                     + world_NN.npy (+ wrist_*) + camera_meta.json + done_NN.flag

Primitives:
  Phase 2 (OSC arm):  move_to, move_delta, rotate_pitch, set_gripper, release
  Phase 3 (NAV, new): navigate_to, move_base
  Phase 4 (grasp):    scripted_grasp  (descend + close; RLDX swap-in is a TODO)

Native PandaOmron 12-d action layout (verified):
  [0:3] arm OSC dpos | [3:6] arm OSC drot | [6] gripper(+1 close/-1 open)
  [7:10] base vel [fwd,lat,turn] | [10] torso | [11] base_mode(>0 base, <=0 arm)
OSC scale: action[-1,1] -> +-0.05 m / +-0.5 rad per step (output_max).
"""
import os, json, time
import traceback


class RoboCasaDriver:
    def __init__(self, primitives, workdir="/tmp/rc_repl"):
        self.primitives = primitives
        self.workdir = workdir

    # ---------- command dispatch + main loop ----------
    def execute(self, cmd):
        act = cmd.get("action")
        if act is None:
            return {"error": "missing 'action' field in command"}
        kwargs = {k: v for k, v in cmd.items() if k != "action"}
        handler = getattr(self.primitives, act, None)
        if handler is None:
            return {"error": f"unknown primitive: {act}"}
        return handler(**kwargs)

    def run(self, max_commands=700, poll=0.5):
        step = 0
        self.primitives.dump_state(step)               # initial state_00
        try:
            self.primitives.dump_success_criteria()    # success_criteria.md (what counts as done)
        except Exception as _e:
            print(f"[driver] success_criteria dump failed: {_e}", flush=True)
        print(f"[driver] ready. workdir={self.workdir}  state_00 dumped.", flush=True)
        while step < max_commands:
            cpath = f"{self.workdir}/command.json"
            if not os.path.exists(cpath):
                time.sleep(poll); continue
            try:
                cmd = json.load(open(cpath))
            except Exception:
                time.sleep(poll); continue
            os.remove(cpath)
            step += 1
            t0 = time.time()
            try:
                res = self.execute(cmd)
            except Exception as e:
                res = {"error": str(e), "trace": traceback.format_exc()[-800:]}
            try:
                st = self.primitives.dump_state(step)
            except Exception as e:
                print(f"[driver] dump_state failed (non-fatal): {e}\n{traceback.format_exc()[-600:]}", flush=True)
                st = {"success": False, "_dump_error": str(e)}
                with open(f"{self.workdir}/state_{step:02d}.json", "w") as f:
                    json.dump({"step": step, "success": False, "dump_error": str(e),
                               "state": {}}, f)
            with open(f"{self.workdir}/log_{step:02d}.json", "w") as f:
                json.dump({"command": cmd, "result": res, "dt": round(time.time() - t0, 2)}, f, indent=2, default=str)
            print(f"[driver] step {step}: {cmd.get('action')} -> "
                  f"{ {k: res[k] for k in ('ok','error','final_dist') if k in res} } "
                  f"success={st['success']}", flush=True)
            if st["success"]:
                print("[driver] TASK SUCCESS.", flush=True)
        print("[driver] command budget reached, exiting.", flush=True)
