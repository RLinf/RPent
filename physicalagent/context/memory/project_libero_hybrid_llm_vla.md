---
name: project-libero-hybrid-llm-vla
description: "Hybrid LLM-in-the-loop + Pi0.5 VLA pipeline solved libero_spatial task 0 (On(bowl,plate)); driver at physicalagent/primitives/"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9df802c0-b380-4d01-8d48-706e324854e2
---

Worked end-to-end on 2026-05-18 — libero_spatial task 0, seed 0, libero_terminated=True with bowl→plate xy err=10mm.

**Architecture:** `physicalagent/primitives/hybrid_pick_place.py` runs as one process, dumps `post_pick.png` + `state_after_pick.json` then blocks on `<workdir>/decision.json`. Claude reads image (Read tool) + privileged state (Bash), writes target_xyz to decision.json. Script unblocks, runs scripted `move_to(target_xyz, gripper=+1)` + `release()`, dumps result.

**Three lessons from this run (essential for any future Hybrid LLM+VLA experiment on LIBERO):**

1. **Pick terminator must require descend→close→ascend, not just |peak-min| ≥ thresh.**
   `peak_z - min_z` is the descent depth (peak stays at start, min drops). If the predicate fires at descent bottom (gripper just closed but no lift), `move_to` will drag the bowl out of a loose grip. Fix: `(start_z - min_z) ≥ 0.10 AND (post_min_peak_z - min_z) ≥ 0.05 AND gripper_closed`. Tracked in primitives.py.

2. **Bowl-EEF xy offset must be measured and compensated.**
   Pi0.5's pick leaves the bowl held with a 2–5 cm xy offset from EEF center (varies per grasp). Aiming EEF at `plate_xy` directly drops the bowl 3–5 cm off the plate. Use `target_eef_xy = plate_xy - (bowl_pos - eef_pos)` measured at post-pick.

3. **Release-on-contact beats release-from-height.**
   Dropping from 9 cm above plate caused +3.5 cm lateral drift (asymmetric finger opening pushes bowl). Aim `eef.z = plate_top + bowl_half_height + |bowl_eef_dz|` so the bowl bottom touches the plate surface BEFORE release fires.

**Why this matters:** validates the "Pi0.5 = local-skill executor; LLM = symbolic planner" pattern for downstream RL warm-starts or agentic policy ablations. The vision-blind nature of pi05_libero130_fullshot ([[project-pi05-libero-prompt-blind]]) makes it OK for the pick step — prompt doesn't matter, model picks whatever is on the table — and the LLM provides the placement reasoning the model lacks.

**How to apply:**
- The `LiberoPrimitiveDriver` (primitives.py) exposes `pick / place / move_to / release / render_agentview / get_privileged_state` — composable building blocks for any hybrid LLM-LIBERO task.
- For multi-step tasks (drawer-open-then-pick-then-place, etc.), chain primitives the same way; reuse the file-based handshake or extend to a richer JSON DSL.
- Generalize beyond libero_spatial: same driver works on libero_object (multi-object disambiguation needed — would test whether prompt-following matters for picking the *right* object).

**Batch result (2026-05-18, all 10 libero_spatial tasks × 2 seeds = 20 rollouts):**
- 1st pass (sub-instruction "pick up the black bowl" + scripted offset-compensated place): **14/20 libero_term**.
- 2nd pass on the 6 failures with FULL task description as the pick prompt: **6/6 rescued** → combined **20/20 = 100%**.
- Two regimes emerged:
  - **Simple tabletop pick (t0,1,2,3,5,6,7,8)**: sub-instr + scripted place = full task.
  - **Hard pick context (t4 drawer, t9 cabinet)**: sub-instr fails (Pi0.5 descends to table reflexively); full task prompt makes Pi0.5 do the *entire* task internally (pick→drawer-open→place), `stopped_at_pick=True` because libero_term fires DURING the pick primitive. Scripted place phase never runs.
- Implication: best LLM-orchestrated hybrid policy is graceful degradation — try sub-instr first (LLM does high-level planning), fall back to full task instruction (LLM becomes a dispatcher; Pi0.5 does everything).
- Code: `physicalagent/primitives/{test_hybrid_all_spatial.py, test_hybrid_failed_retry.py}`. Per-rollout JSONs in `results_all_spatial{,_retry}/`.

**libero_10 (libero_long) extension — 8/10 libero_term, 5 strict (2026-05-19):**
- Strict "Pi0 only pick + LLM all place" success on t0, t1, t4, t6, t7 — all are clean pick-place (two boxes/cans/mugs → basket/plates).
- Pi0 doubled as non-pick VLA on t2 (knob turn), t3 (drawer close), t5 (re-place into narrow caddy after LLM's release got stuck on caddy rim).
- HARD failures:
  - t8 (both moka pots → stove): Pi0 picked one moka with full prompt; my LLM placed it via offset compensation. Second pick FAILED — Pi0 repeatedly closed gripper but missed moka_1. Best guess: scene state with one pot already on burner is OOD vs Pi0's SFT distribution ("both pots on table" was probably canonical init). Need either fully scripted close-gripper-then-lift or per-pot multi-trial with reset on miss.
  - t9 (mug in microwave + close): Pi0 picked mug; LLM moved EEF toward microwave heating region but EEF stuck at y=0.30 (15 cm short of target). Cause: OSC controller can't navigate past microwave top plate at z≈1.12 — Panda wrist physically blocked. Workspace+IK constraints, not policy. Need explicit collision-aware path-planning (lift over → drop into cavity).
- Slow-step lesson generalized: libero_10 cylindrical/box objects (soup cans, butter, cream cheese) need `step_clip ≤ 0.02` to avoid mid-translation gripper slip; spatial bowl tolerated 0.04.
- Multi-stage move (lift → travel → descend) is mandatory for cross-table travel — single diagonal move always OSC-stalls or causes object drop.
- Object disturbance during travel: gripper fingers (open or closed) sweep adjacent items; basket / moka pots get pushed +5-10 cm by robot proximity. Always re-read state JSON before issuing next target.
- Code: `physicalagent/primitives/interactive_driver.py` extended with `--suite libero_10 --task <N>` flags and `reset` REPL command.

**Strict regime — "Pi0 only for pick, LLM does everything else" — 10/10 success on libero_spatial (2026-05-18):**
Key trick to enforce strictness on elevated/drawer pickups (t4, t9): added `track_obj` + `track_obj_lift_thresh` params to `pick()` that monitor the target object's z (privileged sim state) and break out of Pi0.5 mid-trajectory as soon as the object lifts 5 cm above its init z. With this hard cut, Pi0.5 has no opportunity to execute its trained pick-and-place sequence — the LLM (Claude) then takes over via scripted move_to + release.
- t9 (bowl on cabinet): full-prompt pi0_pick with track_obj triggers at chunk 13/40; LLM moves 24 cm to plate (single move_to); release in 3 steps → libero_term=True, bowl→plate err **3.1 mm**.
- t4 (bowl in drawer): pi0_pick triggers at chunk 12/50; LLM does 3-stage move (clear-drawer-z + horizontal-travel + descend) because OSC can't do large diagonal moves in one shot; release in 8 steps → libero_term=True, bowl→plate err **15.3 mm**.
- Interactive driver: `physicalagent/primitives/interactive_driver.py` — REPL via `$OUTPUT_DIR/hybrid_repl_<tag>/command.json` polling; supports move_to/pi0_pick/release/set_gripper/reset/exit actions; appends a step entry to `states.json` and writes `images/image_<N>.png` after every command.
- LIBERO OSC controller can NOT do large diagonal moves with a held object in one shot — break long path into stages (lift → translate → descend) for reliable convergence.
