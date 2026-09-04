#!/usr/bin/env bash
# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Do not let a runner's interactive shell or previously activated environment
# affect imports or uv's target environment.
unset PYTHONPATH VIRTUAL_ENV CONDA_PREFIX

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <libero-pro|robocasa|robotwin> <output-dir> <venv-root>"
  exit 2
fi

target=$1
output_dir=$2
venv_root=$3
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must expose exactly one selected GPU"
  exit 1
fi
if [[ -e "$output_dir" || -e "$venv_root" ]]; then
  echo "Output and venv paths must not already exist"
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to create clean Python 3.10/3.11 environments"
  exit 1
fi
for command in timeout setsid ps; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "$command is required to bound tests and verify process cleanup"
    exit 1
  fi
done

install_timeout=${RPENT_INSTALL_TIMEOUT:-45m}
test_timeout=${RPENT_TEST_TIMEOUT:-45m}
pytest_timeout=${RPENT_GPU_PYTEST_TIMEOUT_SECONDS:-1200}

if [[ ! "$pytest_timeout" =~ ^[1-9][0-9]*$ ]]; then
  echo "RPENT_GPU_PYTEST_TIMEOUT_SECONDS must be a positive integer"
  exit 1
fi

mkdir -p "$output_dir" "$venv_root"

cleanup() {
  if [[ -d "$venv_root" ]]; then
    find "$venv_root" -depth -delete
  fi
}
trap cleanup EXIT

require_dir() {
  local name=$1
  local value=${!name:-}
  if [[ -z "$value" || ! -d "$value" ]]; then
    echo "$name must point to a directory"
    exit 1
  fi
}

require_file() {
  local name=$1
  local value=${!name:-}
  if [[ -z "$value" || ! -f "$value" ]]; then
    echo "$name must point to a file"
    exit 1
  fi
}

session_processes() {
  local session_id=$1
  ps -o pid= --sid "$session_id" 2>/dev/null | awk '{$1=$1}; NF {print}' || true
}

wait_for_session_exit() {
  local session_id=$1
  local attempts=$2
  local attempt

  for ((attempt = 0; attempt < attempts; attempt++)); do
    if [[ -z "$(session_processes "$session_id")" ]]; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

terminate_session_processes() {
  local session_id=$1
  local -a pids=()

  mapfile -t pids < <(session_processes "$session_id")
  if ((${#pids[@]} == 0)); then
    return
  fi

  kill -TERM "${pids[@]}" 2>/dev/null || true
  if wait_for_session_exit "$session_id" 50; then
    return
  fi

  mapfile -t pids < <(session_processes "$session_id")
  if ((${#pids[@]} > 0)); then
    kill -KILL "${pids[@]}" 2>/dev/null || true
    wait_for_session_exit "$session_id" 25 || true
  fi
}

verify_test_session_cleanup() {
  local name=$1
  local session_file=$2
  local report="$output_dir/$name-daemon-cleanup.txt"
  local session_id
  local residual_before
  local residual_after

  if [[ ! -s "$session_file" ]]; then
    echo "cleanup_status=failed" >"$report"
    echo "reason=pytest session id was not recorded" >>"$report"
    echo "$name cleanup verification failed: session id was not recorded"
    return 1
  fi
  session_id=$(<"$session_file")
  if [[ ! "$session_id" =~ ^[1-9][0-9]*$ ]]; then
    echo "cleanup_status=failed" >"$report"
    echo "reason=invalid pytest session id: $session_id" >>"$report"
    echo "$name cleanup verification failed: invalid session id"
    return 1
  fi

  # Parent-watch daemons may need a moment to observe EOF after pytest exits.
  wait_for_session_exit "$session_id" 25 || true
  residual_before=$(session_processes "$session_id")
  if [[ -n "$residual_before" ]]; then
    terminate_session_processes "$session_id"
  fi
  residual_after=$(session_processes "$session_id")

  {
    echo "session_id=$session_id"
    if [[ -n "$residual_before" ]]; then
      echo "residual_before_cleanup=${residual_before//$'\n'/,}"
    else
      echo "residual_before_cleanup=none"
    fi
    if [[ -n "$residual_after" ]]; then
      echo "residual_after_cleanup=${residual_after//$'\n'/,}"
      echo "cleanup_status=failed"
    else
      echo "residual_after_cleanup=none"
      echo "cleanup_status=passed"
    fi
  } >"$report"

  if [[ -n "$residual_after" ]]; then
    echo "$name cleanup verification failed; see $report"
    return 1
  fi
  if [[ -n "$residual_before" ]]; then
    echo "$name left processes running; they were terminated; see $report"
    return 1
  fi
  echo "$name daemon cleanup verified"
}

run_gpu_pytest() {
  local name=$1
  local python=$2
  local test_dir=$3
  local suite_output=$4
  local junit_path=$5
  local session_file="$output_dir/$name-pytest-session.txt"
  local test_status
  local cleanup_status=0

  set +e
  RPENT_E2E_OUTPUT_DIR="$suite_output" \
    timeout --foreground --kill-after=2m "$test_timeout" \
    bash -c '
      set -euo pipefail
      session_file=$1
      shift
      setsid "$@" &
      session_id=$!
      echo "$session_id" >"$session_file"
      cleanup() {
        kill -TERM -- "-$session_id" 2>/dev/null || true
      }
      trap cleanup EXIT HUP INT TERM
      wait "$session_id"
    ' bash "$session_file" \
      "$python" -m pytest -v "$test_dir" \
      --timeout="$pytest_timeout" \
      --junitxml="$junit_path"
  test_status=$?
  set -e

  verify_test_session_cleanup "$name" "$session_file" || cleanup_status=$?
  if ((test_status != 0)); then
    return "$test_status"
  fi
  return "$cleanup_status"
}

run_in_clean_env() {
  local name=$1
  local python_version=$2
  local extra=$3
  local test_dir=$4
  local suite_output=$5
  local venv_dir="$venv_root/$name"

  timeout --foreground --kill-after=2m "$install_timeout" \
    uv venv "$venv_dir" --python "$python_version"
  # Torch and torchvision version compatibility must come from the selected
  # public RPent extra and its transitive package metadata. The CUDA wheel
  # backend is the separate runner/install-policy input UV_TORCH_BACKEND.
  # Do not hide a dependency-contract failure with a workflow-only version pin.
  timeout --foreground --kill-after=2m "$install_timeout" uv pip install \
    --python "$venv_dir/bin/python" \
    --editable "${repo_root}[$extra,test]"
  timeout --foreground --kill-after=2m "$install_timeout" \
    uv pip check --python "$venv_dir/bin/python"
  (
    cd "$repo_root"
    run_gpu_pytest \
      "$name" \
      "$venv_dir/bin/python" \
      "$test_dir" \
      "$suite_output" \
      "$output_dir/$name-junit.xml"
  )
  find "$venv_dir" -depth -delete
}

case "$target" in
  libero-pro)
    require_dir PI05_CHECKPOINT_PATH
    require_file SAM3_CHECKPOINT_PATH
    require_dir LIBERO_PRO_ASSET_PATH
    venv_dir="$venv_root/libero-pro"
    timeout --foreground --kill-after=2m "$install_timeout" \
      uv venv "$venv_dir" --python 3.11
    timeout --foreground --kill-after=2m "$install_timeout" uv pip install \
      --python "$venv_dir/bin/python" \
      --editable "${repo_root}[libero-pro,test]"
    # LIBERO-PRO resolves package data through realpath(__file__). Materialize
    # this distribution so its package path remains inside the job venv even
    # when the rest of the environment uses the shared cache via symlinks.
    liberopro_version=$(
      "$venv_dir/bin/python" -c \
        'from importlib.metadata import version; print(version("rpent-liberopro"))'
    )
    timeout --foreground --kill-after=2m "$install_timeout" \
      env UV_LINK_MODE=copy uv pip install \
      --python "$venv_dir/bin/python" \
      --force-reinstall \
      --no-deps \
      "rpent-liberopro==$liberopro_version"
    timeout --foreground --kill-after=2m "$install_timeout" \
      uv pip check --python "$venv_dir/bin/python"
    timeout --foreground --kill-after=2m "$install_timeout" \
      "$venv_dir/bin/liberopro-download-assets" --link "$LIBERO_PRO_ASSET_PATH"
    (
      cd "$repo_root"
      export RPENT_LIBERO_VARIANT=pro
      run_gpu_pytest \
        libero-pro \
        "$venv_dir/bin/python" \
        tests/e2e_tests/libero \
        "$output_dir/pro" \
        "$output_dir/libero-pro-junit.xml"
    )
    find "$venv_dir" -depth -delete
    ;;
  robocasa)
    require_dir RLDX_MODEL_PATH
    require_file ROBOCASA_MACROS_PATH
    require_dir ROBOCASA_ASSETS_PATH
    run_in_clean_env \
      robocasa 3.10 robocasa tests/e2e_tests/robocasa "$output_dir/robocasa"
    ;;
  robotwin)
    require_dir LINGBOT_MODEL_PATH
    require_dir ROBOTWIN_ASSETS_PATH
    if [[ ! -d "$LINGBOT_MODEL_PATH/qwen_base" ]]; then
      echo "LINGBOT_MODEL_PATH must contain qwen_base"
      exit 1
    fi
    run_in_clean_env \
      robotwin 3.11 robotwin tests/e2e_tests/robotwin "$output_dir/robotwin"
    ;;
  *)
    echo "Unknown target: $target"
    exit 2
    ;;
esac
