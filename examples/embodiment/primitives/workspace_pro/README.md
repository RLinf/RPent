# LIBERO-Pro workspace

This directory holds artifacts specific to evaluating the hybrid LLM+Pi0.5
pipeline on **LIBERO-Pro** (Zhou et al., 2025,
`https://github.com/Zxy-MLlab/LIBERO-PRO`). Files:

- `env_calibration.md` — OSC reachable workspace bounds for PRO's two scene
  frames (LIVING_ROOM eef z≈0.68, KITCHEN eef z≈1.17). Cross-frame z range,
  xy ±0.30 single-step cap, basket-shift thresholds.
- `liberopro_register_perturbations.patch` — patch against
  `https://github.com/RLinf/LIBERO-PRO.git@0bcf736` (latest as of
  2026-05-20). Adds the 15 missing perturbation benchmark variants
  (`{libero_spatial,libero_object,libero_goal,libero_10}_{swap,task,lan,object}`,
  minus `libero_spatial_swap` which has no BDDL files). Also overrides
  `Task.language` to read the actual `:language` tag inside each BDDL, so
  the P1 Task and Semantic perturbations expose their perturbed instruction
  through the standard `benchmark.get_task(i).language` API.

## How to apply the patch

```bash
# After install_liberopro from the RLinf install.sh (or any clone of
# RLinf/LIBERO-PRO at commit 0bcf736):

cd /opt/venv/openpi/libero_pro   # or wherever liberopro is installed
git apply /path/to/RLinf_agentic/examples/embodiment/primitives/workspace_pro/liberopro_register_perturbations.patch
```

After applying, the following 15 new benchmark names become available via
`liberopro.liberopro.benchmark.get_benchmark(name)`:

```
libero_spatial_task  libero_spatial_lan  libero_spatial_object
libero_object_swap   libero_object_task  libero_object_lan   libero_object_object
libero_goal_swap     libero_goal_task    libero_goal_lan     libero_goal_object
libero_10_swap       libero_10_task      libero_10_lan       libero_10_object
```

(`libero_spatial_swap` is intentionally absent — the LIBERO-PRO repo ships
0 BDDL files for it, presumably because Position perturbation is ill-defined
when the task itself is "pick the X **between** the Y and the Z".)

## Perturbation taxonomy (LIBERO-Pro paper)

| Suffix | LIBERO-Pro name | Paper column | Description |
|---|---|---|---|
| `_swap` | Position | P2 | Object initial positions swapped; instruction + goal unchanged |
| `_task` | Task | P1 | Instruction and goal predicate changed; init positions unchanged |
| `_lan` | Semantic | — | Paraphrased instruction; goal predicate unchanged |
| `_object` | Object | — | Object appearance / colour / scale altered; instruction unchanged |

P1 and P2 are the two columns where, per the LIBERO-Pro paper, all current
end-to-end VLAs collapse to ≈0% on. They are the headline experimental
slots for the agentic LLM-in-the-loop hybrid.

## Sanity check after applying

```python
import os
os.environ["LIBERO_TYPE"] = "pro"
import liberopro.liberopro.benchmark as bench

# P2 Position: same instruction, different object placement
b = bench.get_benchmark("libero_10_swap")()
assert b.get_task(8).language == "put both moka pots on the stove"

# P1 Task: different instruction, same initial placement
b = bench.get_benchmark("libero_10_task")()
assert b.get_task(8).language == "put the left moka pot on the stove"

# Semantic: paraphrased instruction
b = bench.get_benchmark("libero_10_lan")()
assert b.get_task(8).language == "place both moka pots on stove"
```
