# Install the shared RL workflow

[Run tasks](README.md) · [简体中文](INSTALL_zh.md)

Use one `reason` environment for LIBERO rollout and RL training. Install
separate RoboCasa or RoboTwin environments if needed.

## 1. System prerequisites and checkout

Use the Linux x86_64 `reason` image with `/opt/venv/reason`, the NVIDIA driver,
CUDA 12.8 development toolkit, a C++ compiler and EGL/OpenGL. Keep both
`rpent-ppo` repositories under `$STACK_ROOT/repos/{RLinf,RPent}` and record
their revisions. The prepared package lives at
`$STACK_ROOT/envs/offline_package`; copy it in full to each node.

## 2. Planner, SAM3 and LIBERO

Install completely offline, then activate the existing environment:

```bash
export STACK_ROOT=/mnt/public/xzxuan
bash "$STACK_ROOT/envs/offline_package/install.sh"
source "$STACK_ROOT/envs/offline_package/activate.sh"
python -m pip check
rpent --help
```

The package verifies hashes and image versions, backs up replaced packages
and installs local wheels. Repeated installation keeps the same versions.
Models and scene assets remain separate shared inputs. The environment uses
Python 3.12.12, Torch 2.11.0/CUDA 12.8, torchvision 0.26.0, FA2 2.8.3,
Transformer Engine 2.17, Transformers 5.8.1, NumPy 1.26.4, MuJoCo 3.3.0 and
robosuite 1.5.2. It supplies SAM3 and LIBERO without OpenPI/VLA dependencies.
RLinf, SGLang, planner and simulator child processes all use this interpreter.

For dedicated collection GPUs, `python -m robots.rl.services` accepts
`--sam3-per-gpu 2` and `--envs-per-gpu N`. Measure environment capacity before
setting `N`; these options allocate service capacity without changing the
planner's workflow or episode budgets.

The package's activation file supplies the prepared shared paths. For a new
data root, configure the model, SAM3 and LIBERO paths in
`tools/rl/config.example.sh` and save a local copy as `.rl-env.sh`.

Request access to [facebook/sam3](https://huggingface.co/facebook/sam3) before
logging in. Download the full Qwen3.8-27B directory, including its tokenizer, processor
and chat template, and the SAM3 `sam3.pt` checkpoint. Downloads need network access and sufficient disk space:

```bash
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
hf auth login
hf download Qwen/Qwen3.8-27B --local-dir "$MODEL_PATH"
hf download facebook/sam3 sam3.pt --local-dir "$(dirname "$SAM3_CHECKPOINT_PATH")"
mkdir -p "$LIBERO_CONFIG_PATH" "$LIBERO_PRO_ASSET_PATH"
env -u LIBERO_PRO_ASSET_PATH liberopro-download-assets \
  --assets-dir "$LIBERO_PRO_ASSET_PATH" --skip-existing --no-cache
```

The temporary `env -u` avoids the downloader selecting an existing fallback asset
directory. If your network requires a Hugging Face mirror, set `HF_ENDPOINT`
before downloading. Use your own proxy settings; no private download helper
is required. Demonstration datasets are unnecessary for task execution.
BDDL and initialization states ship in `rpent-liberopro`. Generate its local
configuration after installing the package and downloading assets:

```bash
python - <<'PY'
import os
from pathlib import Path
import yaml
import liberopro.liberopro as libero
config = libero.get_default_path_dict()
config["assets"] = str(Path(os.environ["LIBERO_PRO_ASSET_PATH"]).resolve())
config["datasets"] = str(Path(os.environ["RPENT_DATA_ROOT"]) / "libero-pro/datasets")
for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
    assert Path(config[key]).is_dir(), (key, config[key])
path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
path.write_text(yaml.safe_dump(config))
print(path.read_text())
PY
```

Keep `benchmark_root` at the installed package location. Regenerate this YAML
after relocating the environment or assets, because its paths are absolute.
The guide uses LIBERO-PRO (`LIBERO_TYPE=pro`); standard/plus variants require
the corresponding package and asset/config setup.

## 3. Model and local environment services

SGLang `0.5.12.post1` uses RLinf's external Qwen image processor. RLinf captures
serving prompt IDs, processed pixels, image spans, grid and M-RoPE for learner
replay. The package installs the unmodified SGLang release with the image's
CUDA 12 kernels. Megatron-Core `0.18.0+rpent.cu128.1` includes the offload
checkpoint repair; Bridge `0.5.0+rpent.cu128.1` uses the compatible CUDA 12
profile. Build sources and the lock are included in the offline package.

Start one service manager per node for the four-node 80-candidate smoke:

```bash
python -m robots.rl.services --sam3-checkpoint "$SAM3_CHECKPOINT_PATH" \
  --envs-per-gpu 3 --output-dir "$STACK_ROOT/logs/services-node0"
```

It starts one SAM3 per GPU, a pooled endpoint on port 28424 and the environment
backend on port 28490. The service manager stops its child processes on exit. Use a fresh output
directory and keep the service manager alive while collecting. Three slots per GPU fit
80 concurrent candidates across 32 GPUs; a larger run needs its own capacity
check. The model service also uses `/opt/venv/reason/bin/python`:

```bash
MODEL_PATH="$MODEL_QWEN38_27B" ENABLE_MTP=0 bash tools/rl/serve.sh
```

[Run tasks](README.md#run-one-task) describes the standalone trajectory command.
Use the paired RLinf guide for Ray startup, the SFT model, complete groups,
training and checkpoint validation.

## 4. RoboCasa runtime and assets

From the checkout root, create the isolated Python 3.11 environment. The pinned
robosuite revision supplies `load_model_on_init`, which is absent in the
LIBERO environment's PyPI 1.5.2 version:

```bash
uv venv --python 3.11 .venv-robocasa
uv pip install --python .venv-robocasa/bin/python -r requirements/rl-robocasa.txt
uv pip check --python .venv-robocasa/bin/python
.venv-robocasa/bin/python -c "import robocasa, robosuite; print('RoboCasa imports OK')"
```

Download the assets with the simulator environment's downloader. This version
stores assets in the installed package's `robocasa/models/assets`; allocating
the venv on a disk with sufficient space is the simplest setup:

```bash
.venv-robocasa/bin/python -m robocasa.scripts.download_kitchen_assets --type all
```

Accept the downloader's asset confirmation; demonstration/trajectory datasets
are not required. To place assets on a different disk, **before this download**
copy the bundled directory to that disk and replace the package directory
with a symlink. Merely setting `ROBOCASA_ASSETS_PATH` does not redirect this
version's downloader. [The reference](REFERENCE.md#relocating-robocasa-assets)
provides the optional relocation command.

The example `.rl-env.sh` points `RPENT_ROBOCASA_PYTHON` at this venv. Activate
`/opt/venv/reason` for rollout commands; `--simulator robocasa` launches the
native server with the isolated interpreter. RLDX weights are not used.

## 5. RoboTwin runtime and assets

Ensure `nvcc --version` reports the toolkit used to build against Torch
(CUDA 12.8 here), and the NVIDIA Vulkan ICD is installed. `vulkaninfo --summary`
should list the NVIDIA GPU even on a headless host. Then install:

```bash
uv venv --python 3.11 .venv-robotwin
uv pip install --python .venv-robotwin/bin/python torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-robotwin/bin/python -r requirements/rl-robotwin.txt
uv pip install --python .venv-robotwin/bin/python ninja wheel 'setuptools>=69.5.1,<75.9'
uv pip install --python .venv-robotwin/bin/python --no-build-isolation 'git+https://github.com/NVlabs/curobo.git@v0.7.8'
```

RoboTwin also imports an RLinf environment adapter. Install its source path in
this venv without installing RLinf's full training dependency stack:

```bash
mkdir -p "$RPENT_DATA_ROOT/src"
git clone --branch rpent-ppo https://github.com/zhexuanxu/RLinf.git "$RPENT_DATA_ROOT/src/RLinf"
git -C "$RPENT_DATA_ROOT/src/RLinf" checkout f351c57e62818b18315ed4b2194312a44cafb697
RLINF_ROOT="$RPENT_DATA_ROOT/src/RLinf" .venv-robotwin/bin/python - <<'PY'
import os
import sysconfig
from pathlib import Path
root = Path(os.environ["RLINF_ROOT"]).resolve()
assert (root / "rlinf/envs/sim/robotwin/robotwin_env.py").is_file()
Path(sysconfig.get_path("purelib"), "rpent_robotwin_rlinf_source.pth").write_text(str(root) + "\n")
PY
uv pip check --python .venv-robotwin/bin/python
.venv-robotwin/bin/python -c "from robots.robotwin.rlinf_env import RoboTwinAgentEnv; print('RoboTwin adapter imports OK')"
.venv-robotwin/bin/robotwin-download-assets --output "$ROBOTWIN_ASSETS_PATH"
```

`ROBOTWIN_ASSETS_PATH` is the snapshot **root**, containing
`.robotwin-assets.json` and `assets/`, not the inner `assets/` directory.
The pinned runtime validates its asset snapshot. The default task preset is
`demo_randomized`; choose a published valid seed for that task/preset.
Activate `/opt/venv/reason` to launch tasks. The example configuration
selects `.venv-robotwin/bin/python` via `RPENT_ROBOTWIN_PYTHON`; LingBot/VLA
weights are unnecessary.

## 6. Verify and troubleshoot

Run the four single-task commands in [README.md](README.md#run-one-task), one
at a time. These exercise LIBERO episode/ZAP and native RoboCasa/RoboTwin
episode through the same planner. Inspect both native success and the final
tool results; an unsuccessful task can still complete a valid episode.

| Symptom | First check |
| --- | --- |
| Planner cannot connect | `/v1/models`, model name, API key and endpoint from the planner host |
| SAM3 fails to load | Granted model access, full checkpoint path, GPU memory and service log |
| LIBERO missing XML/BDDL/init states | Asset inventory and generated `config.yaml`; regenerate after moving paths |
| RoboCasa missing `load_model_on_init` | `RPENT_ROBOCASA_PYTHON` selects the pinned isolated robosuite, not `.venv` |
| RoboTwin missing `rlinf` | `.pth` file and the exact checked-out adapter directory in the simulator venv |
| cuRobo build/import fails | Torch/CUDA toolkit match, C++ compiler, build dependencies installed first |
| EGL or Vulkan create-device failure | NVIDIA EGL/Vulkan driver libraries, ICD and actual device allocation; the message alone does not prove CUDA OOM |
| CUDA library symbol mismatch | New clean terminal and isolated interpreter; remove inherited conflicting Torch/NVIDIA library paths |
| Unknown task or invalid seed | Native task name, preset and published RoboTwin seed table |

If a platform stores drivers outside the loader's search paths, configure
`LD_LIBRARY_PATH` and `VK_ICD_FILENAMES` for **that platform's installed
driver**, not paths copied from another machine. CUDA/Vulkan GPU enumeration
can differ; check the actual process allocation with `nvidia-smi` during the
first single-environment run. Increase concurrency only after measuring
resident scene memory and cold-start overhead.
