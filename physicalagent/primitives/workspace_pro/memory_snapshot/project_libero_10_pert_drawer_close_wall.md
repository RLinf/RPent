---
name: libero-10-pert-drawer-close-wall
description: "libero_10 task_t3/swap_t3 (object in bottom drawer + close): In is achievable but the white_cabinet drawer CLOSE is a hard wall — thin handle glances, pi0 defeated by perturbation"
metadata: 
  node_type: memory
  type: project
  originSessionId: 173e389d-6fb4-4726-99ec-b7c6d73b9da0
---

**2026-05-27.** Tried to finish libero_10 PRO `task_t3` (bottle→bottom drawer→close) and `swap_t3`
(bowl→bottom drawer→close). The drawer is ALREADY OPEN at init (qpos -0.1439). **In is achievable**:
a scripted OSC/joint-space grasp of the (knocked-flat) bottle + carry placed it in `white_cabinet_1_bottom_region`
(verified -0.011,0.176,0.922 within x[-0.036,0.024] y[0.098,0.25] z[0.85,1.054]). The bottle is narrow
(r0.022) and slips in a joint-space grasp UPRIGHT, but the LYING bottle grasps fine (gw0.049) — knock it
flat first (pi0 conveniently knocks it; or push it).

**The CLOSE is a genuine wall** (neither scripted nor pi0 closes it):
- **Close threshold is STRICT**: `white_cabinet` bottom_level `is_close` needs **qpos > ~0.005** (fully
  seated; articulation default_close_ranges [0.0,0.005], default_open_ranges [-0.16,-0.14]). Range [-0.16,0.01].
- **Scripted push maxes at qpos -0.035**, can't seat the final ~4cm: the **handle is a thin bar at z0.946**
  that GLANCES when pushed (~5:1 slip) and **shields the drawer face panel** (gripper hits the handle first at
  y0.07, can't reach the panel behind it). Pushing above the handle (z1.0) **passes over with no contact**;
  z1.035 contacts but **hits the closed MIDDLE drawer front** at the cabinet face (y~0.177) and stalls at -0.035.
- **Grip-the-bar + translate** (the swap_t0 winning trick): the z-straddle grip (gripper points ±y) **IK-diverges /
  drive-stalls** at this low bottom bar; the y-straddle grip (gripper-down rolled) catches the bar (gw0.0142) but
  is **too weak to drag** the drawer (thin-bar pinch slips, drawer just wiggles).
- **pi0_doubled close is defeated by the perturbation** (the whole point of LIBERO-Pro): on the P1-perturbed
  task pi0 does the ORIGINAL task — it won't grasp the bottle (gripper never closes) and won't close the
  perturbed-task drawer (descends but doesn't push).

So task_t3/swap_t3 reach **In but not Close**. Distinct from swap_t0/swap_t8 (SOLVED this session). The wall is
the thin-handle + strict-seating + pi0-perturbation-defeat combination. Scripts in workspace_pro/jointspace_experiments/
(js_t3.py, js_close2.py); close-threshold probe /tmp/closethresh.py. See [[no-teleport-rule]],
[[handle-bar-grasp-orientation]], [[gripper-ctrl-is-finger-position]].
