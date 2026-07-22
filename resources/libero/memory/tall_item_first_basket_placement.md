---
name: tall-item-first-basket-placement
description: Reusable shared-basket ordering rule that places a tall unstable bottle before a small box.
aliases:
  - put both the ketchup and the cream cheese box in the basket
  - ketchup cream cheese woven basket
metadata:
  node_type: memory
  type: skill
---

# Tall-item-first basket placement

## Applicable pattern

Use this pattern when a tall unstable object and a small box share a basket target and the tall object becomes unreliable once the basket is occupied.

## Reference task alias

`put both the ketchup and the cream cheese box in the basket`

> **REVISED** — supersedes a prior cream-cheese-first memory. A 5-attempt exploration disproved cream-cheese-first; the actual winning order is **KETCHUP FIRST**, then cream cheese, both with verbose pi0 auto-place prompts. The prior recipe's premise (tall bottle can autoplace into a basket already holding the box) is wrong — attempts 2/3/4 of this trail all tried it and the bottle ended on the rim or rolled out.

## Winning technique
Put BOTH the **ketchup** (tall orange bottle, white cap, ~10 cm) and the **cream_cheese** (small blue/white box) into the **woven basket** (right side, back edge clipped past +x). **Technique:** Pure pi0 auto-place, twice, **TALL ITEM FIRST into an empty basket**. Step 1: `pi0_pick "pick up the ketchup and place it in the basket"`, `max_chunks=30`, `lift_thresh=0.99`. Step 2: `move_to` home `(-0.058, 0, 0.681)` with `gripper=-1` to clear the eef and restore the home pose. Step 3: `pi0_pick "pick up the cream cheese and place it in the basket"`, same parameters. Pi0 internally re-localizes from RGB; no SAM3 needed for the canonical recipe. The whole recipe is 3 JSONL lines and on reference run fires `libero_terminated=True` inside Step 3 (12 chunks). **Order is non-negotiable.**

## Magic numbers (defaults + small ranges)
- **BOTH pi0_pick calls**: prompt = `"pick up the {X} and place it in the basket"` (verbose suffix REQUIRED), `max_chunks=30` (band 28–32 for ketchup; 25–35 for cream cheese), `lift_thresh=0.99` (≥0.95, keeps loop running THROUGH release), `gripper_closed_thresh=0.06`.
- **Home pose between picks**: `move_to [-0.058, 0.0, 0.681]`, `gripper=-1` (OPEN), `step_clip=0.025`. Suite-canonical, NOT layout-derived.
- **NEVER** raise ketchup `max_chunks` above ~32 — at 40 with the verbose suffix, pi0 wanders and knocks the bottle horizontal (attempt 2; no verified horizontal-cylinder recovery is known).
- **NEVER** call `set_gripper +1` after `pi0_pick` (crushes pi0's hold, pops the object).
- **NEVER** manually carry the bottle into a basket that already contains the box — descent stalls on box top at z≈0.50; release leaves bottle perched at z≈0.55-0.58 and it rolls out (attempts 3, 4).
- **NEVER** blindly re-pick an object touching the basket rim — pi0 descends through the rim wall and tips the basket (attempt 1) or tips the bottle horizontal (attempt 3). Neither resulting state has a verified recovery.
- Manual-carry fallback (only if pi0 auto-place truly fails): release at `z ≥ 0.68` over basket interior `(BX+0.02, BY+0.025)` from segment centroid.

## Failure modes to avoid (mined from 4 failed attempts)
| Failure mode | Root cause (attempt #) | Fix |
|---|---|---|
| Manual carry: cream_cheese released at `(0.075, 0.215, 0.62)` bounces off front rim → out | Release too low and too close to front rim (A1) | Use pi0 auto-place. If manual fallback: release at z ≥ 0.68 with `+0.02 x, +0.025 y` interior offset. |
| Re-picking a rim-perched object drives pi0 into basket wall → **BASKET TIPS OVER** | Pi0 descent passes through basket front wall when object is at rim (A1) | Do not repeat the same Pi0 call blindly. Continue only if current images show the basket is upright and the object has a distinct, physically safe grasp; otherwise no verified recovery is known. |
| `pi0_pick "pick up the ketchup and place it in the basket"` with `max_chunks=40` → pi0 wanders, knocks bottle horizontal | Verbose suffix + large budget lets pi0 explore; horizontal cylinder is OOD (A2) | Use `max_chunks=30` not 40. |
| Cream-cheese-first ordering: tall ketchup can't autoplace into basket containing the box | Pi0 descent stalls on box top (z≈0.50); release leaves bottle perched on back rim (z≈0.55-0.58) → rolls out | Reverse the order — **ketchup FIRST into empty basket** (A5 winning insight). |
| Manual ketchup carry into basket-with-box stalls at z≈0.694 → bottle bounces out | Tall bottle + occupied basket = no vertical clearance (A4) | Same fix — ketchup first into empty basket. |
| Pi0 can't grasp horizontal bottle (rescue attempts after tipping) | Franka finger reach below eef is short; at eef z=0.49, fingertips above bottle body z=0.47 (A2) | Prevention first — keep `max_chunks ≤ 32` for ketchup. Once tipped, do not repeat the same Pi0 call; continue only if current images support a distinct, physically feasible grasp because no verified horizontal-bottle recovery is known. |

## How to RE-LOCALIZE each entity per-scene
**DO NOT cache reference run absolutes** (K ≈ `(-0.180, -0.158, 0.508)`, CC ≈ `(-0.146, 0.060, 0.457)`, BK ≈ `(0.074, 0.219, 0.492)`) — items respawn at different xy on every scene.

The canonical 3-line recipe needs **NO segment calls** — both pi0_pick steps re-localize from the RGB image internally. Localize only for the manual-carry fallback:

- **K = ketchup** — SAM3 prompt **`"the orange ketchup bottle with red label"`** on `agentview`. Fallbacks: `"the orange ketchup bottle"` → `"the orange-and-white bottle"`. Sanity: `KZ ≈ 0.50–0.52`, tall narrow ORANGE silhouette with white cap. REJECT tomato_sauce (red CAN, not a bottle). Score ≥ 0.6.
- **CC = cream_cheese** — SAM3 prompt **`"the small white and blue cream cheese box"`** on `agentview`. Fallbacks: `"the small blue box"` → `"the small blue and silver box"`. Sanity: `CZ ≈ 0.45–0.46`. REJECT alphabet_soup (blue/white CAN, taller). Score ≥ 0.5.
- **BK = basket** — SAM3 prompt **`"the woven basket"`** on `agentview`. Score ≥ 0.3. **Basket back edge clips past +x image edge**, so visible centroid is biased ~10 cm toward −x. Estimated interior xy ≈ `(BX+0.02, BY+0.025)`. Floor z ≈ 0.46 — sample world_hi pixels inside the basket bbox where `z < 0.47` and take the median as a robust interior estimate.

## Scene-variation fragility flags
1. **Ketchup pi0 auto-place** (Step 1) — primary risk. If `peak_lift_m < 0.05` after 30 chunks or bottle ends on table: retry once with same params. If still failing → fall back to bare-prompt grasp (`"pick up the ketchup"`, `max_chunks=20`, `lift_thresh=0.05`) + `rotate_pitch target_pitch=1.0` (lay bottle horizontal — lower profile, stable) + manual carry to basket interior `(BX+0.02, BY+0.025, 0.56)` with `step_clip=0.008` + release.
2. **Bottle tipped horizontal at any point** — inspect the current images and do not repeat the same Pi0 call blindly. Continue only if a distinct, physically feasible grasp is available; otherwise stop because no verified horizontal-bottle recovery is known.
3. **Cream-cheese auto-place** (Step 3) — secondary risk, much more forgiving. If fails: retry once; then fall back to manual carry with high-drop release at `z ≥ 0.68` over interior offset `(BX+0.02, BY+0.025)`.

## Difficulty and reliability
**5 attempts to a win** (4 archived failures + 1 success). Difficulty: **MEDIUM-HIGH**.
- A1: manual carry → rim-bounce → basket tipped
- A2: cream-cheese-first + ketchup verbose `max_chunks=40` → bottle tipped horizontal
- A3: cream-cheese-first + ketchup short-prompt manual carry → bottle on back rim
- A4: cream-cheese-first + ketchup short-prompt manual carry deeper → descent stalled on box top, bottle bounced out
- A5 (WIN): **ketchup-first auto-place `max_chunks=30`** + home + cream-cheese auto-place `max_chunks=30`

Expected single-attempt success in unseen layouts with this recipe: **≥ 70%**.

## Cross-refs
- `[[can_and_small_package_basket_placement]]` — sibling multi-object-into-basket pattern with a different split. This pattern diverges because Pi0 wanders if given both objects in one prompt, and the tall-item-first rule is specific to the ketchup+basket geometry.
- `[[feedback_move_pose_covarying_reach]]` — informs why manual scripted descent into the basket fails and the `z ≥ 0.68` release floor for manual fallback.
- `[[feedback_pi0_delivery_service]]` — pi0 delivers itself when prompted within its trained skill; do not script the carry.
- `[[feedback_scripted_pick_limits]]` — never `set_gripper +1` after `pi0_pick`; pi0 owns the grip.
