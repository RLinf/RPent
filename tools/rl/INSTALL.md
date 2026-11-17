# Install the shared RL workflow

[Run tasks](README.md) · [简体中文](INSTALL_zh.md)

This guide starts from a fresh Linux checkout. Install the common planner,
SAM3 and LIBERO environment first, then the isolated RoboCasa and/or RoboTwin
runtime you need. All paths are derived from your checkout and chosen data
root. No cluster activation scripts, existing checkpoints or VLA weights are
assumed. Run commands from the checkout root unless stated otherwise.

## 1. System prerequisites and checkout

Use Linux x86_64, Git, [uv](https://docs.astral.sh/uv/getting-started/installation/)
and NVIDIA GPUs with a working driver. `nvidia-smi` must succeed. The example
SGLang wheels use CUDA 13 and require a compatible driver; the simulator stack
uses CUDA 12.8. Activating a Python environment does not install the host
driver or repair inherited CUDA library paths. RoboTwin also needs NVIDIA
Vulkan/EGL support, a CUDA 12.8 toolkit (`nvcc`) and a C++ compiler for cuRobo.
On Ubuntu install the remaining system libraries:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential ffmpeg libegl1 libopengl0 libgl1 libglib2.0-0 libvulkan1 vulkan-tools
# Install uv using its linked official instructions if it is not available.
uv python install 3.12 3.11
git clone --branch rpent-ppo https://github.com/zhexuanxu/RPent.git
cd RPent
cp tools/rl/config.example.sh .rl-env.sh
```

Use a revision containing `robots/rl`; record `git rev-parse HEAD` with your
results. Edit `.rl-env.sh` to select the data disk, model/SAM3 locations and
service endpoints. Its default data root is `${XDG_DATA_HOME:-$HOME/.local/share}/rpent`.
Open new terminals from this checkout before activating each environment;
do not inherit `PYTHONPATH` or CUDA library overrides from an RL/SFT job.

## 2. Planner, SAM3 and LIBERO

The common environment owns the planner, geometry, SAM3 and LIBERO. RoboCasa
and RoboTwin run as subprocesses in their own environments to avoid conflicting
robosuite, NumPy, SciPy and Torch requirements.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --prerelease=allow --torch-backend cu128 -r requirements/libero-rl.txt
source .venv/bin/activate
source .rl-env.sh
source tools/rl/env.sh
uv pip check --python .venv/bin/python
rpent --help
```

The requirements select the matching RLinf `rpent-ppo` runtime, SAM3 and public
LIBERO forks. They do not use the older `rpent[rlinf]` extra on its own, and do
not need OpenPi/Pi0.5. The tested main versions are Python 3.12, Torch 2.11.0
(CUDA 12.8), torchvision 0.26.0, Transformers 4.57.6, MuJoCo 3.3.0,
robosuite 1.5.2 and NumPy 1.26.4. These requirement files are not full
transitive lockfiles; save `uv pip freeze` for reproducible deployments.

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

## 3. Model serving

In a new terminal at the checkout root, install SGLang separately:

```bash
uv venv --python 3.12 .venv-sglang
uv pip install --python .venv-sglang/bin/python --prerelease=allow "sglang==0.5.23" "torchvision==0.29.2"
uv pip check --python .venv-sglang/bin/python
```

The validated serving stack is SGLang 0.5.23, sgl-kernel 0.4.10,
Torch 2.14.1/CUDA 13.0 and Transformers 5.19.0. Keep it separate from the
simulator's CUDA 12.8 wheels. [README.md](README.md#start-the-shared-services)
starts the service, checks `/v1/models`, starts SAM3 and runs the first LIBERO
task. A remote compatible endpoint can replace this local serving install.
For a checkpoint without MTP weights set `ENABLE_MTP=0` when running `serve.sh`.

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
the common `.venv` for rollout commands; `--simulator robocasa` launches the
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
Activate the common planner `.venv` to launch tasks. The example configuration
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
