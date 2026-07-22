# LIBERO Prompt Organization

## Purpose

This document describes how LIBERO prompts are assembled and how guide source
files are referenced by the active prompt.

It is intended for contributors who need to inspect or modify LIBERO prompt
behavior.

## Prompt Assembly Flow

The active prompt path is:

1. `rpent/cli/main.py` selects the environment spec and renders prompts.
2. `robots/libero/prompt_bundle.py` assembles the LIBERO prompt sections.
3. `robots/libero/prompts/system.py` defines the active LIBERO system prompt fragments.
4. `robots/libero/prompts/shared.py` defines shared guide-loading and runtime-adapter constants.
5. The rendered prompt is passed to the selected planner backend.

## Files and Responsibilities

| File | Responsibility |
| --- | --- |
| `prompt_bundle.py` | Returns the active LIBERO system and user prompt sections. |
| `prompts/system.py` | Defines `PREAMBLE`, `GOAL`, `RULES`, `LOCALIZATION`, `WORKFLOW`, `ENVIRONMENT`, `NEXT`, and `USER_MODE`. |
| `prompts/shared.py` | Defines shared constants such as `MCP_RUNTIME_ADAPTER` and `GUIDE_READ_INSTRUCTIONS`. |
| `guides/strict_hybrid_guide.md` | Guide source file for strict hybrid behavior. |
| `guides/pro_hybrid_guide.md` | Guide source file for PRO-specific guidance. |
| `guides/env_calibration.md` | Guide source file for environment calibration notes. |

## Guide Source Loading

The active prompt references guide source files through `GUIDE_READ_INSTRUCTIONS`.

`GUIDE_READ_INSTRUCTIONS` asks Claude Code to read these guide source files once
at the start of each run with the structured `Read` tool, before issuing the
first physical command:

```text
robots/libero/guides/strict_hybrid_guide.md
robots/libero/guides/pro_hybrid_guide.md
robots/libero/guides/env_calibration.md
```


## MCP Runtime Adapter

RPent uses structured runtime tools. Some guide content may include
legacy command examples or older command formats.

`MCP_RUNTIME_ADAPTER` is placed before guide-loading instructions so the active
prompt preserves guide strategy while using the current structured runtime tools.
Detailed runtime constraints, such as tool availability and forbidden legacy
paths, should stay in `prompts/system.py` and the guide-level runtime contract.

## SAM3 Segmentation

The `segment` tool is a perception helper backed by a separate SAM3 RPC
service. `rpent/cli/main.py` starts that service automatically unless an
existing endpoint is provided with `--sam3-endpoint`.

To start a reusable service manually:

```bash
export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
python -m robots.libero.sam3_server \
  --transport http --host 127.0.0.1 --port 8114

rpent --sam3-endpoint http://127.0.0.1:8114 ...
```

If SAM3 occupies a GPU, do not schedule LIBERO env jobs on that same GPU.
Both HTTP and socket RPC transports expose the same `segment` method for text
and single-positive-point prompts.

When a matching world map artifact exists, `segment` writes collision-safe
`segment_NN_XX.json` artifacts with mask statistics and `world_xyz`; repeated
`segment()` calls on the same source step receive different `XX` indexes instead
of overwriting earlier evidence. If the world map is missing or the mask shape
does not match, the tool records a `world_error` instead of failing the episode.

## Safe Checklist for Prompt Changes

Before changing LIBERO prompts:

1. Confirm the active prompt path: `rpent/cli/main.py` -> `prompt_bundle.py` -> `prompts/system.py`.
2. Check which shared constants are imported by `prompts/system.py`.
3. Keep runtime-tool guidance before guide-source instructions.
4. Keep guide source paths and guide-level runtime contracts accurate.
5. Avoid changing unrelated prompt behavior in the same patch.
6. For static prompt changes, inspect the rendered prompt or grep the active prompt fragments before relying on the new path.
