# LIBERO planner tool batching experiment

[简体中文](README.zh-CN.md)

This opt-in runner compares ordinary tools with bounded `execute_sequence`
calls. The planner submits a list of existing tools and their already-known
arguments. Each child runs synchronously through the ordinary LIBERO toolkit,
including state capture and cancellation checks. The sequence returns the child
results and the final observation. It does not batch VLA requests, change action
chunks, or delegate scripted movement to Pi0.5.

## Run

Run from a clean RPent checkout on a Linux GPU host with the
[LIBERO-Pro prerequisites](../../../docs/source-en/rst_source/simulators/libero.rst).
The Python environment must support the local simulator and Codex planner, with
Codex authentication already configured. Start compatible OpenPI and APXInf
RPent RPC servers and SAM3 separately; see the APXInf section in that guide.
Both VLA servers should use the same checkpoint, normalization, observation
routing and action-chunk settings. Shared servers remain running after the sweep.

Supply a frozen local evaluation-memory directory and a JSON provenance file.
The provenance JSON is operator-supplied evidence, not automatic verification:
record dataset/initial-state revisions and hashes, memory origin, checkpoint and
normalization hashes, RPent/RLinf/OpenPI/APXInf-robo/engine revisions, dtype,
chunk settings, GPU, CUDA and relevant environment variables. Do not include
credentials. Compare backend metadata and validate these inputs before running.

```bash
python -m robots.libero.experiments.benchmark_batching \
  --backend openpi=http://127.0.0.1:9012 \
  --backend apxinf=http://127.0.0.1:9014 \
  --sam3-endpoint http://127.0.0.1:9018 \
  --memory-dir /path/to/frozen-memory \
  --provenance /path/to/provenance.json \
  --model gpt-6-astra --reasoning-effort low \
  --tasks 0 1 2 3 4 5 6 7 8 9 --seeds 0 1 \
  --batch-sizes 0 4 \
  --output-dir logs/batching-object-01
```

Use an available Codex model consistently across all arms. The runner supports
`libero_object_task`: 10 pick-and-place tasks, each placing a different object in
the basket. This command creates 80 episodes: 10 tasks × 2 seeds × 2 backends ×
2 batching settings. The existing environment server maps each seed to an
initial-state index modulo the number of available states. This is not a full
LIBERO-Pro evaluation. The runner rotates arm order between task/seed pairs and
checks matching initial camera-image hashes. That check does not establish
identical trajectories or numerical equivalence of the VLA backends.

`--batch-sizes` controls the schema, tool description and appended planner policy:

- `0`: ordinary tools, no sequence tool; retain this baseline.
- `1`: sequence tool available with one child; a separate interface/prompt control.
- `4`, `8`, etc.: at most that many children in one call, not a target or minimum.

The planner can still use ordinary tools. Increasing the limit does not guarantee
longer batches. Measure actual actions per sequence and planner responses.
`--batch-sizes 0 1 4 8` with the same tasks and backends creates 160 episodes.
One child such as `pi0_pick` can itself make multiple VLA predictions; the batch
limit is a tool count, not a VLA-chunk count. Tool-internal budgets and success
thresholds remain unchanged. The treatment changes both tool availability and
prompt instructions, so their contributions are not isolated.

Defaults are 100 planner responses, 900 seconds of planner time and 10,000
environment steps. `--max-turns`, `--planner-timeout-s` and
`--max-episode-steps` change these limits for every arm. The worker watchdog allows
an additional 900 seconds for setup and cleanup. Each episode starts a fresh local
environment server; there is no real-robot or Flash mode.

## Sequence boundaries

All child names and schemas are validated before any motion. Allowed children
are `move_to`, `move_pose`, `rotate_wrist`, `set_gripper`, `pi0_pick`,
`pi0_doubled` and `release`. Nested sequences, reset and perception calls cannot
be batched. Unknown coordinates require a separate perception/planning step.

The sequence returns at the first of:

- toolkit cancellation or a tool error;
- native task success or environment truncation;
- an unconfirmed `pi0_pick` (`success` is not true);
- a reported convergence/position-hold failure;
- a missing or excessive final position distance from `move_to`/`move_pose`
  (tolerance plus 0.0001 m for rounded results);
- completion of `pi0_doubled`, so the planner inspects the contact outcome;
- the end of the submitted list.

`pi0_doubled.success` indicates whole-task termination, not contact-skill success.
It is therefore an observation boundary, not classified as a failed grasp.
These are experimental continuation rules over existing tool feedback, not new
simulator success criteria. They do not prove every subgoal was achieved:
for example, `move_pose` does not report a final yaw error and this runner does
not add a separate orientation verifier. The original tools own their internal
control loops. An early return usually gives control back to the planner rather
than declaring the whole episode failed.

## Evidence and resume

The manifest records the command settings, supplied provenance, frozen-memory
hashes, experiment source hashes and RPent revision. Every cell retains its own
memory copy, rendered system prompt, planner text/raw events, tool dispatch log,
RPC log, environment log, states/images/video and result. Artifact storage can
be large; provision space for the complete matrix. No artifacts are deleted.

`result.json` uses the environment's native termination flag, never the planner's
`finish` claim. Planner-budget exhaustion is a task outcome; transport/service
failures or an abnormal worker exit are infrastructure failures and stop the
sweep. Unknown planner errors are conservatively treated as infrastructure
failures for manual inspection. All attempts are retained; there is no automatic
retry or replacement of a failed result.

Repeat the exact command with `--resume` to skip completed valid cells. Resume
requires identical manifest inputs. An infrastructure failure or interrupted
cell is not silently retried: inspect its evidence, then use a fresh output
root for a separately disclosed rerun. Do not combine successful replacements
with the original attempts without reporting the selection policy.

`summary.json` reports valid successes/counts, all-outcome and successful-outcome
mean times, actual sequence adoption, VLA calls per planner response, VLA timing
share and paired success changes. Paired speedups include only pairs where both
arms succeeded, with the number of faster pairs; short failures are not speedups.
Each task has correlated seeds, so these are descriptive summaries, not a
statistical equivalence test.

Episode wall time starts before environment launch and ends after the planner
returns, excluding final cleanup and shared-server startup/readiness. It includes
toolkit initialization, memory copy, planning, actions and state capture. VLA RPC
time includes client encoding/transport only to the extent covered by the RPC
call; it is not pure GPU time and excludes simulator execution. Tool and RPC
intervals overlap: do not add those totals. The residual is not a measurement
of LLM latency. Different action trajectories and RPC latency can change wall
time independently of inference speed.

## Offline checks

```bash
pytest tests/unit_tests/robots/libero/test_batching_experiment.py -v
```

These tests cover dispatch/stop/cancellation behavior and aggregation with fakes.
They do not establish GPU task success or validate the benchmark's dependencies.
