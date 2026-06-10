---
name: project-libero-t9-pi0-only
description: libero_10 t9 (mug in microwave + close) is the one libero_10 task where strict hybrid (LLM-only place) fails due to Panda IK at the cavity; Pi0 end-to-end (~186 chunks) solves it non-strict.
metadata: 
  node_type: memory
  type: project
  originSessionId: 9df802c0-b380-4d01-8d48-706e324854e2
---

**Fact:** libero_10 task 9 ("put the yellow and white mug in the microwave
and close it") cannot be solved with strict hybrid (Pi0 pick + LLM place).
Pi0 end-to-end with the full task prompt solves it in ~186 chunks
(`max_chunks=100` × 2 calls; first call lifts and approaches, second call
finishes insertion + door close). Video at
`physicalagent/primitives/videos/t9_pi0_SUCCESS.mp4`.

**Why strict fails:** OSC stalls at y≈0.26, x∈[-0.05, +0.05], z∈[1.03, 1.10]
across 6+ staging variants. This is a Panda IK singularity at the workspace
location required to thread the EEF through the cavity opening — the cavity
(`x∈[-0.152, +0.055]`, `y∈[+0.273, +0.440]`, `z∈[+0.944, +1.088]`) is
narrow on the left (open door panel at x=-0.19) and bounded on the right
by a thick wall at x=+0.055. The Panda base at world (-0.66, 0, 0.912)
forces a ~0.72 m diagonal reach, and the link 6-7 chain cannot find a
configuration that fits. Scripted `move_to` with any `step_clip` up to 0.05
cannot break through; only Pi0's closed-loop visual servoing finds a
viable trajectory. Linked: [[feedback-read-image-before-decide]],
[[project-libero-hybrid-llm-vla]].

**How to apply:** If you re-open this task, don't repeat the scripted
attempts — go straight to Pi0 end-to-end. If you discover a strict
solution (e.g. by pre-closing the door to clear the workspace before
opening again), update this memory.

**libero_10 final tally:** 10/10 libero_term; 6/10 strict; 3/10 Pi0
doubled on a non-pick skill (knob/drawer/caddy); 1/10 (t9) Pi0 end-to-end.
