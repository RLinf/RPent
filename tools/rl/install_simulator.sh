#!/usr/bin/env bash
# Install one simulator and its RPent entry points into an isolated environment.
set -euo pipefail

if [[ $# != 3 ]]; then
  echo 'Usage: install_simulator.sh {libero|robotwin|robocasa|robodojo} VENV RLINF_ROOT' >&2
  exit 2
fi

simulator="$1"
venv="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$2")"
rlinf_root="$(cd "$3" && pwd)"
cd "$(dirname "$0")/../.."
case "$simulator" in
  libero)
    python_version=3.12
    profile=libero-rl.txt
    ;;
  robotwin|robocasa|robodojo)
    python_version=3.11
    profile="rl-$simulator.txt"
    ;;
  *)
    echo "Unknown simulator: $simulator" >&2
    exit 2
    ;;
esac
uv venv --python "$python_version" "$venv"
python="$venv/bin/python"
pip=(uv pip install --python "$python")
case "$simulator" in
  libero)
    "${pip[@]}" --index https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
    ;;
  robotwin)
    "${pip[@]}" --index https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
    ;;
  robocasa)
    "${pip[@]}" --index https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
    ;;
  robodojo)
    "${pip[@]}" --index https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0
    ;;
esac
options=()
if [[ "$simulator" == robodojo ]]; then
  options=(--extra-index-url https://pypi.nvidia.com --override requirements/robodojo-override.txt)
fi
"${pip[@]}" -r "requirements/$profile" "${options[@]}"
if [[ "$simulator" == robotwin ]]; then
  "${pip[@]}" ninja wheel 'setuptools>=69.5.1,<75.9'
  "${pip[@]}" --no-build-isolation 'git+https://github.com/NVlabs/curobo.git@v0.7.8'
fi
"$python" - "$rlinf_root" <<'PY'
from pathlib import Path
import sys
import sysconfig

root = Path(sys.argv[1])
if not (root / "rlinf/envs").is_dir():
    raise SystemExit(f"Expected an RLinf checkout at {root}")
Path(sysconfig.get_path("purelib"), "rlinf-source.pth").write_text(f"{root}\n")
PY
"$python" -m robots.rl.remote_manager --help > /dev/null
"$python" -c 'from transformers import AutoProcessor'
