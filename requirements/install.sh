#!/usr/bin/env bash
# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  cat <<'EOF'
Usage: bash requirements/install.sh (--env <name> | --model <name> | --core) [options]

Selection:
  --env <name>     libero, libero-pro, libero-plus, robocasa, robotwin, franka,
                   robodojo, yam. Includes the default model dependencies below.
  --model <name>   openpi, rldx, lingbotvla, sam3, molmo.
                   Installs a standalone model service; cannot be combined with --env.
  --core           RPent base dependencies only (CPU development).

Environment defaults:
  libero / libero-pro / libero-plus: OpenPI + SAM3
  robocasa: RLDX; robotwin: LingBot-VLA; franka: OpenPI
  robodojo: SAM3; install VPP2 or Awomo05 separately using the platform guide
  yam: hardware dependencies; use the YAM guide for the OpenPI inference setup

Examples:
  bash requirements/install.sh --env libero-pro
  bash requirements/install.sh --env robocasa
  bash requirements/install.sh --env robodojo
  bash requirements/install.sh --model molmo
  bash requirements/install.sh --core --test

Options:
  --venv <path>            Default: <repo>/.venv-<env>, .venv-<model> or .venv-core
  --torch-backend <name>   uv Torch backend, e.g. cu126, cu128, cpu or auto
                           Defaults to UV_TORCH_BACKEND, otherwise auto for non-core.
                           Explicit selection reinstalls torch/torchvision
                           (and torchaudio for RoboDojo).
  --test                  Include RPent's offline test dependencies
  --flywheel              Include LeRobot export dependencies (not with robotwin)
  --use-mirror            Use the Tsinghua PyPI mirror; Git and assets are unchanged
  --dry-run               Print commands without installing or resolving packages
  -h, --help              Show this help

Python: 3.10 for robocasa or rldx; 3.11 otherwise. Existing environments must
match. Environment installs require Linux. Install uv and the platform's
system dependencies first. Assets, weights, credentials and robot setup are
separate steps in the corresponding guide. No GPU tasks are launched here.
EOF
}

fail() {
  echo "install.sh: $*" >&2
  exit 2
}

if [[ $# -eq 0 ]]; then
  usage
  exit 0
fi
env_name=""
model=""
core=0
venv_dir=""
torch_backend=${UV_TORCH_BACKEND:-}
with_test=0
with_flywheel=0
use_mirror=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env|--model|--venv|--torch-backend)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value"
      case "$1" in
        --env)
          [[ -z "$env_name" ]] || fail "only one --env can be specified"
          env_name=$2
          ;;
        --model)
          [[ -z "$model" ]] || fail "only one --model can be specified"
          model=$2
          ;;
        --venv) venv_dir=$2 ;;
        --torch-backend) torch_backend=$2 ;;
      esac
      shift 2
      ;;
    --core) core=1; shift ;;
    --test) with_test=1; shift ;;
    --flywheel) with_flywheel=1; shift ;;
    --use-mirror) use_mirror=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) fail "unknown option: $1 (see --help)" ;;
  esac
done

case "$env_name" in
  ""|libero|libero-pro|libero-plus|robocasa|robotwin|franka|robodojo|yam) ;;
  *) fail "unknown environment: $env_name (see --help)" ;;
esac
case "$model" in
  ""|openpi|rldx|lingbotvla|sam3|molmo) ;;
  *) fail "unknown model: $model (see --help)" ;;
esac
if [[ "$core" == 1 ]]; then
  [[ -z "$env_name$model" ]] || fail "--core cannot be combined with --env or --model"
elif [[ -z "$env_name$model" ]]; then
  fail "select --env, --model or --core (see --help)"
fi
if [[ -n "$env_name" && -n "$model" ]]; then
  fail "--env includes its default models; use --model separately for a standalone service"
fi
if [[ "$with_flywheel" == 1 && ( "$env_name" == robotwin || "$model" == lingbotvla ) ]]; then
  fail "RoboTwin/LingBot-VLA and flywheel require separate environments"
fi

requirements=()
requirement_options=()
install_options=(--prerelease allow)
python_version=3.11
linux_only=0
# Select RLinf first, then environment dependencies, then model dependencies.
if [[ "$env_name" == robodojo ]]; then
  requirement_options=("rlinf @ git+https://github.com/littleZ05/RLinf.git@rpent/robodojo")
elif [[ -n "$env_name" || "$model" == openpi || "$model" == rldx || "$model" == lingbotvla ]]; then
  requirement_options=("rlinf @ git+https://github.com/RLinf/RLinf.git@main")
fi
if [[ -n "$env_name" ]]; then
  linux_only=1
  requirements=("envs/$env_name.txt")
  case "$env_name" in
    libero|libero-pro|libero-plus)
      requirements+=(models/openpi.txt models/sam3.txt)
      ;;
    franka) requirements+=(models/openpi.txt) ;;
    robocasa) requirements+=(models/rldx.txt) ;;
    robotwin) requirements+=(models/lingbotvla.txt) ;;
    yam)
      install_options+=(--build-constraints
        "$repo_root/requirements/envs/yam-build-constraints.txt")
      ;;
    robodojo)
      requirements+=(models/sam3.txt)
      install_options+=(--extra-index-url https://pypi.nvidia.com
        --override "$repo_root/requirements/envs/robodojo-override.txt")
      ;;
  esac
fi
if [[ -n "$model" ]]; then requirements+=("models/$model.txt"); fi
if [[ "$env_name" == robocasa || "$model" == rldx ]]; then python_version=3.10; fi
target=${env_name:-${model:-core}}
venv_dir=${venv_dir:-"$repo_root/.venv-$target"}

# Never install into an activated shell environment by accident.
case "$venv_dir" in /*) ;; *) venv_dir="$PWD/$venv_dir" ;; esac
python="$venv_dir/bin/python"
if [[ -e "$venv_dir" ]]; then
  [[ -f "$venv_dir/pyvenv.cfg" && -x "$python" ]] || fail \
    "$venv_dir is not a Python virtual environment; select a new --venv"
  actual_version=$("$python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  [[ "$actual_version" == "$python_version" ]] || fail \
    "$target needs Python $python_version, found $actual_version; select a new --venv"
fi

if [[ "$dry_run" == 0 ]]; then
  command -v uv >/dev/null 2>&1 || fail "install uv first: https://docs.astral.sh/uv/getting-started/installation/"
  if [[ "$linux_only" == 1 && "$(uname -s)" != Linux ]]; then
    fail "$target requires Linux; use --dry-run to inspect its installation commands"
  fi
fi

if [[ -n "$torch_backend" ]]; then
  install_options+=(--torch-backend "$torch_backend"
    --reinstall-package torch --reinstall-package torchvision)
  case "$env_name" in
    robodojo) install_options+=(--reinstall-package torchaudio) ;;
  esac
elif [[ "$target" != core ]]; then
  install_options+=(--torch-backend auto)
fi
if [[ "$use_mirror" == 1 ]]; then
  install_options+=(--default-index https://pypi.tuna.tsinghua.edu.cn/simple)
fi
project="$repo_root"
extras=()
if [[ "$with_test" == 1 ]]; then extras+=(test); fi
if [[ "$with_flywheel" == 1 ]]; then extras+=(flywheel); fi
if [[ ${#extras[@]} -gt 0 ]]; then
  project+="[$(IFS=,; echo "${extras[*]}")]"
fi
if [[ ${#requirements[@]} -gt 0 ]]; then
  for requirement in "${requirements[@]}"; do
    requirement_options+=(-r "$repo_root/requirements/$requirement")
  done
fi

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$dry_run" == 0 ]]; then "$@"; fi
}

echo "Installing $target (Python $python_version) into $venv_dir"
if [[ ! -e "$venv_dir" ]]; then
  run uv venv "$venv_dir" --python "$python_version"
fi
run uv pip install --python "$python" --editable "$project" "${install_options[@]}" ${requirement_options[@]+"${requirement_options[@]}"}
if [[ "$env_name" == robotwin ]]; then
  # cuRobo's setup.py imports Torch while preparing metadata. Install the
  # runtime first, then compile against that Torch rather than pinning it twice.
  curobo_options=(--no-build-isolation-package nvidia-curobo
    --torch-backend "${torch_backend:-auto}")
  if [[ "$use_mirror" == 1 ]]; then
    curobo_options+=(--default-index https://pypi.tuna.tsinghua.edu.cn/simple)
  fi
  run uv pip install --python "$python" "${curobo_options[@]}" \
    "${requirement_options[@]}" \
    "nvidia-curobo @ git+https://github.com/NVlabs/curobo.git@v0.7.8"
fi
printf 'Activate with: source %q\n' "$venv_dir/bin/activate"
echo "Continue with the platform guide to configure assets, checkpoints and services."
