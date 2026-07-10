# SAM3 service notes

SAM3 is an optional external perception service. `cli/main.py` does not
automatically start SAM3; the runtime `segment` tool only calls the service
configured by `SAM3_SERVER_URL`.

Start a SAM3-compatible HTTP service before running RPent, then export
its URL in the shell that launches RPent:

```bash
export SAM3_SERVER_URL=http://127.0.0.1:8114
```

When `SAM3_SERVER_URL` is not set, `segment` returns a structured fallback so
the agent can continue with image inspection and `back_project`.

The SAM3 server is independent from the VLA server and LIBERO env server. For
large batches, exclude the GPU used by SAM3 from the env-job GPU pool.

## Expected API

The external service should expose:

- `POST /segment`
  - payload: `image_base64` and `text_prompt`
  - response: a score-sorted result list with `mask_base64`, `shape`, `score`,
    and `box`; `overlay_base64` may be included when available
- `POST /segment_point`
  - payload: `image_base64` and `point_coords`
  - response: mask tensor fields such as `masks_base64`, `masks_shape`,
    `masks_dtype`, and optional `scores`

RPent decodes the returned mask and, when a matching LIBERO world map
artifact exists, writes `world_xyz` into collision-safe `segment_NN_XX.json`
artifacts. Repeated `segment()` calls on the same source step receive different
`XX` indexes instead of overwriting earlier evidence.

## Optional pre-run SAM3 service setup

This directory also includes an optional SAM3 startup helper for environments
that already provide a SAM3-compatible launcher:

```bash
export SAM3_LAUNCHER=/absolute/path/to/run_sam3_server.sh
export SAM3_GPU=0
export SAM3_HOST=127.0.0.1
export SAM3_PORT=8114
bash scripts/sam3/run_sam3_server.sh
```

`SAM3_HOST`, `SAM3_PORT`, `SAM3_GPU`, and `SAM3_CUDA_DEVICE` are passed through
to the external launcher when supported by that launcher. The helper prints the
`export SAM3_SERVER_URL=...` command, but it cannot export that variable into
the parent shell. Set `SAM3_SERVER_URL` explicitly before running RPent.
