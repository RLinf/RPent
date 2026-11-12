# Episode observation lists

[简体中文](OBSERVE_zh.md) · [Installation](INSTALL.md) · [Workflows](WORKFLOWS.md)

`observe` is optional for local `episode` runs. It combines `view_env`, `segment`
and `back_project` in a flat `calls` list. It never moves the robot or makes
additional model requests inside the tool.

## Start and configure

Follow the [first-task guide](README.md), configure `.rl-env.sh`, and start the
model and SAM3 services. From your checkout root:

```bash
source .venv/bin/activate
source .rl-env.sh
source tools/rl/env.sh
bash tools/rl/run.sh --simulator libero --workflow episode \
  --perception-mode batch --max-observe-calls 8 --observe-concurrency 8 \
  --suite libero_object --task 0 --seed 1 --cuda-device 0
```

| CLI option | Default | Meaning |
| --- | --- | --- |
| `--perception-mode` | `single` | `batch` replaces the three standalone perception tools with `observe` |
| `--max-observe-calls` | `8` | Positive subcall limit, including views |
| `--observe-concurrency` | `8` | Positive concurrent-subcall limit; SAM worker limits are unchanged |

The same options work in `python -m tools.rl.run_batch run` and are pinned in
the manifest for resume. `run.sh` also accepts `RPENT_PERCEPTION_MODE`,
`RPENT_MAX_OBSERVE_CALLS`, and `RPENT_OBSERVE_CONCURRENCY`. Non-default limits
require batch mode. `episode_image`, all `zap_*` workflows and remote toolkits
reject observe options before startup. Local episodes use the simulator's
actual camera names. Action tools and `finish` remain available; physical
`--action-mode batch` and history retention are independent. Each model reply
still contains exactly one top-level tool call.

## Sources and execution

Each item has a unique local `id` and `op`. `@` names are local to this call;
ordinary data IDs remain usable within the current episode.

| Operation | source | Other arguments |
| --- | --- | --- |
| `view_env` | Forbidden | `camera`, default `all`; reads latest recorded observation |
| `segment` | Image ID or `@view_id.camera` | Exactly one nonblank `prompt` or `point=[row,col]` |
| `back_project` | Image ID or `@view_id.camera` | Exactly one `pixel=[row,col]` or `box=[r0,c0,r1,c1]` |
| `back_project` | Mask ID or `@segment_id` | Neither pixel nor box; optional `z_range_m` |

`z_range_m=[low,high]` is an inclusive world-z filter in metres for box/mask,
never for pixel. Boxes are half-open with positive height and width. Pixels
refer to the displayed source image; mapping to native depth is automatic.
World geometry is in metres.

Initial state and action feedback expose `image_sources` (camera → image ID),
even without RGB. An opaque `image_id` binds an immutable frame, camera and
pixel mapping. Copy IDs; do not construct them. Older image/mask IDs deliberately
measure older frames and cannot be transferred to another episode. A mask
already fixes its frame and camera. Measurement items therefore have no `step`,
`camera`, `frame_from`, `mask_from`, or `mask_id` argument: use `source` only.

The complete graph is validated before any child executes. Bad arguments,
unknown/duplicate IDs, invalid coordinates and cycles reject the entire call.
Forward references work. Source references determine dependencies; independent
nodes run concurrently and dependents start as soon as their own source finishes.
All views in one call use its entry observation; actions cannot interleave.

A missing detection returns explanatory text; service errors are distinct.
Only dependents of failed sources are skipped. Independent results survive.
There is no hidden retry or query rewrite. To choose a query after seeing a
new image, return that view first, then decide in the next model reply.

## Examples

View only, without source:

```json
{"calls":[{"id":"v","op":"view_env","camera":"all"}]}
```

View, two independent segments, then measure one mask. `surface` waits for
`cup` only; it need not wait for `plate`. Each segment selects one mask, so
these two successful segment items produce two mask IDs.

```json
{"calls":[
  {"id":"v","op":"view_env","camera":"all"},
  {"id":"cup","op":"segment","source":"@v.agentview","prompt":"red cup"},
  {"id":"plate","op":"segment","source":"@v.wrist","prompt":"plate"},
  {"id":"surface","op":"back_project","source":"@cup","z_range_m":[0.8,1.2]}
]}
```

No view or segment is required to project a previously viewed image. Replace
the example IDs below with actual returned values. The mask measurement uses
a mask from an earlier call and runs independently of the pixel measurement:

```json
{"calls":[
  {"id":"point","op":"back_project","source":"img_EXAMPLE_0_agentview","pixel":[512,512]},
  {"id":"mask","op":"back_project","source":"m_EXAMPLE","z_range_m":[0.8,1.2]}
]}
```

Likewise, segment alone can use `image_sources.agentview`, without viewing:

```json
{"calls":[{"id":"cup","op":"segment","source":"img_EXAMPLE_0_agentview","prompt":"red cup"}]}
```

## Results and audit

The result is `{"results":[{"id":"...","op":"...","result":...}]}` in input
order. Each result is a success dict or explanatory text. Read every item.

- View: `step`, plus `images` metadata containing `image_id`, `camera`,
  `size=[height,width]` and zero-based `image_index` into attached RGB blocks.
  Repeated images are attached once. Only views return images.
- Segment: `image_id`, `mask_id`, `step`, `box` and visible-surface geometry.
  `world_xyz` is the median valid surface point; `top_z_m` its 95th-percentile z;
  `size_xyz_m` its per-axis 5th–95th percentile extent; `axis_yaw` the major
  xy heading in [-pi/2, pi/2), null when not elongated, not a gripper command. These are not hidden object-centre
  coordinates or guaranteed grasp poses. `score`/`found` are not shown.
- Back projection: `image_id`, source `step`, and depth-measured `world_xyz`.
  Mask/box measurements can include background and occlusion bias. Segment
  already returns geometry; another projection is useful for a different point,
  region or z filter, not automatically required.

`observe/sources.json` records frame/mask provenance; `observe/NNNN.json` records
nodes, dependencies, statuses, scheduling/worker wait and start/end times.
Concurrent times overlap and must not be summed as wall time. Existing native
frames, raw SAM audit, transcripts and `api_trace` remain saved. Returned `step`
is diagnostic metadata, not another source selector.
