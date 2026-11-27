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
unset PYTHONPATH VIRTUAL_ENV CONDA_PREFIX UV_NO_CACHE

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
install_timeout=${RPENT_INSTALL_TIMEOUT:-45m}
test_timeout=${RPENT_TEST_TIMEOUT:-45m}
pytest_timeout=${RPENT_GPU_PYTEST_TIMEOUT_SECONDS:-1200}

if [[ ! "$pytest_timeout" =~ ^[1-9][0-9]*$ ]]; then
  echo "RPENT_GPU_PYTEST_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
fi

require_host_tools() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to create a clean Python environment" >&2
    exit 1
  fi
  for command in timeout setsid ps python3; do
    if ! command -v "$command" >/dev/null 2>&1; then
      echo "$command is required to bound tests and verify process cleanup" >&2
      exit 1
    fi
  done
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
  local output_dir=$2
  local session_file=$3
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

junit_all_skipped() {
  local junit_path=$1
  [[ -f "$junit_path" ]] || return 1
  python3 - "$junit_path" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)
failures = sum(int(s.attrib.get("failures", 0)) for s in suites)
errors = sum(int(s.attrib.get("errors", 0)) for s in suites)
sys.exit(0 if tests > 0 and skipped == tests and failures == 0 and errors == 0 else 1)
PY
}

run_gpu_pytest() {
  local suite_name=$1
  local case_name=$2
  local output_dir=$3
  local python=$4
  shift 4

  local run_tag="$suite_name"
  if [[ -n "$case_name" ]]; then
    run_tag="${suite_name}-${case_name}"
  fi
  local suite_output="$output_dir/$suite_name"
  local junit_path="$output_dir/$run_tag-junit.xml"
  local session_file="$output_dir/$run_tag-pytest-session.txt"
  local test_status
  local cleanup_status=0

  set +e
  PYTHONPATH="$repo_root" \
    RPENT_E2E_SUITE="$suite_name" \
    RPENT_E2E_CASE="$case_name" \
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
      "$python" -m pytest -v -rs \
      --timeout="$pytest_timeout" \
      --basetemp="$output_dir/$run_tag-pytest-tmp" \
      --junitxml="$junit_path" "$@"
  test_status=$?
  set -e

  verify_test_session_cleanup "$run_tag" "$output_dir" "$session_file" || cleanup_status=$?
  if ((test_status != 0)); then
    return "$test_status"
  fi
  if [[ -z "$(find "$output_dir" -name tool-coverage.json -print -quit)" ]]; then
    if [[ -n "$case_name" ]] && junit_all_skipped "$junit_path"; then
      echo "$run_tag skipped due to missing optional prerequisites"
      return "$cleanup_status"
    fi
    echo "$run_tag did not execute a planner/tool coverage test; check skipped prerequisites" >&2
    return 1
  fi
  return "$cleanup_status"
}

run_rpent_install() {
  # Runners can preload verified wheels alongside the shared uv cache.
  if [[ -z "${UV_FIND_LINKS:-}" && -n "${UV_CACHE_DIR:-}" && -d "$UV_CACHE_DIR/wheelhouse" ]]; then
    export UV_FIND_LINKS="$UV_CACHE_DIR/wheelhouse"
  fi
  timeout --kill-after=2m "$install_timeout" "$repo_root/rpent-install.sh" "$@"
}

install_test_runner() {
  local python=$1
  timeout --kill-after=2m "$install_timeout" \
    uv pip install --python "$python" pytest pytest-timeout
}

materialize_package_files() {
  local python=$1
  shift
  "$python" - "$@" <<'PY'
import importlib.metadata
import os
import shutil
import sys
from pathlib import Path

for pkg in sys.argv[1:]:
    dist = importlib.metadata.distribution(pkg)
    for rel in dist.files or ():
        path = Path(dist.locate_file(rel))
        if path.is_symlink():
            target = path.resolve()
            if target.is_file():
                tmp = path.with_name(f".{path.name}.materialize.{os.getpid()}")
                shutil.copy2(target, tmp)
                os.replace(tmp, path)
PY
}

reinstall_copy_packages() {
  local python=$1
  shift
  local pkg
  for pkg in "$@"; do
    [[ -n "$pkg" ]] || continue
    local requirement
    requirement=$(uv pip freeze --python "$python" \
      | awk -v pkg="$pkg" '$1 == pkg || index($1, pkg "==") == 1 {print}')
    if [[ -z "$requirement" ]]; then
      echo "Expected $pkg in $python" >&2
      return 1
    fi
    timeout --kill-after=2m "$install_timeout" uv pip install \
      --python "$python" --link-mode copy \
      --reinstall-package "$pkg" --no-deps "$requirement"
  done
}

strip_repo_editable_pth() {
  local python=$1
  "$python" - <<'PY'
import site
from pathlib import Path

for sp in site.getsitepackages():
    sp_dir = Path(sp)
    if not sp_dir.is_dir():
        continue
    # Remove both legacy .pth and PEP 660 setuptools editable finder files for rpent
    # so shared cached/run venvs always resolve rpent from PYTHONPATH="$repo_root".
    for pattern in ("*rpent*.pth", "__editable___rpent_*_finder.py"):
        for path in sp_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass
PY
}

build_component_venv() {
  local component=$1
  local output_dir=$2
  local venv_dir=$3
  local portable=${4:-0}

  local INSTALL_COMPONENT=""
  local INSTALL_HAS_EVAL="0"
  local INSTALL_CACHE_HASH=""
  local -a INSTALL_MATERIALIZE_PKGS=()
  local -a INSTALL_REINSTALL_PKGS=()
  local -a INSTALL_POST_CMDS=()
  eval "$(python3 "$repo_root/tests/e2e_tests/suite.py" install-meta "$component" "$venv_dir/bin/python")"

  mkdir -p "$output_dir" "$(dirname "$venv_dir")"
  rm -rf "$venv_dir"

  run_rpent_install install "$INSTALL_COMPONENT" --venv "$venv_dir"
  cp "$venv_dir/rpent-install.json" "$output_dir/"
  run_rpent_install doctor --venv "$venv_dir"
  if [[ -x "$venv_dir/bin/rpent" ]]; then
    "$venv_dir/bin/rpent" --help >/dev/null
  fi

  if [[ "$INSTALL_HAS_EVAL" == "1" ]]; then
    install_test_runner "$venv_dir/bin/python"
    if ((${#INSTALL_MATERIALIZE_PKGS[@]} > 0)); then
      materialize_package_files "$venv_dir/bin/python" "${INSTALL_MATERIALIZE_PKGS[@]}"
    fi
    if ((${#INSTALL_REINSTALL_PKGS[@]} > 0)); then
      reinstall_copy_packages "$venv_dir/bin/python" "${INSTALL_REINSTALL_PKGS[@]}"
    fi
    if ((${#INSTALL_POST_CMDS[@]} > 0)); then
      local post_cmd
      for post_cmd in "${INSTALL_POST_CMDS[@]}"; do
        [[ -n "$post_cmd" ]] || continue
        PATH="$venv_dir/bin:$PATH" timeout --kill-after=2m "$install_timeout" bash -c "exec $post_cmd"
      done
    fi
    if [[ "$portable" == "1" ]]; then
      strip_repo_editable_pth "$venv_dir/bin/python"
    fi
    date -u +"%Y-%m-%dT%H:%M:%SZ" >"$venv_dir/.rpent-ci-ready"
  else
    # Standalone component install without downstream eval: verify & clean up.
    find "$venv_dir" -depth -delete
  fi
}

ensure_cached_venv() {
  local component=$1
  local cache_root=$2
  local output_dir=$3

  local INSTALL_COMPONENT=""
  local INSTALL_HAS_EVAL="0"
  local INSTALL_CACHE_HASH=""
  local -a INSTALL_MATERIALIZE_PKGS=()
  local -a INSTALL_REINSTALL_PKGS=()
  local -a INSTALL_POST_CMDS=()
  eval "$(python3 "$repo_root/tests/e2e_tests/suite.py" install-meta "$component")"

  local cached_dir="$cache_root/${component}-${INSTALL_CACHE_HASH}"
  if [[ -f "$cached_dir/.rpent-ci-ready" ]]; then
    echo "$cached_dir"
    return 0
  fi

  mkdir -p "$cache_root/.locks"
  local lock_file="$cache_root/.locks/${component}-${INSTALL_CACHE_HASH}.lock"
  python3 - "$lock_file" "$cached_dir" "$component" "$output_dir" "$repo_root" <<'PY'
import fcntl
import os
import subprocess
import sys
from pathlib import Path

lock_file, cached_dir, component, output_dir, repo_root = sys.argv[1:6]
Path(lock_file).parent.mkdir(parents=True, exist_ok=True)
fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o666)
try:
    fcntl.flock(fd, fcntl.LOCK_EX)
    if not (Path(cached_dir) / ".rpent-ci-ready").is_file():
        cmd = [
            "python3",
            f"{repo_root}/tests/tooling/gpu_slots.py",
            "--kind",
            "install",
            "--slots",
            "1",
            "--",
            "bash",
            f"{repo_root}/tests/e2e_tests/run_gpu_suite.sh",
            "install",
            component,
            output_dir,
            cached_dir,
            "--portable",
        ]
        # Redirect build output to stderr so stdout only returns cached_dir.
        subprocess.run(cmd, check=True, stdout=sys.stderr)
finally:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
PY
  echo "$cached_dir"
}

run_eval_subcommand() {
  local suite_name=$1
  local case_name=$2
  local output_dir=$3
  local venv_arg=$4
  local venv_mode=${RPENT_VENV_MODE:-run}

  if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "CUDA_VISIBLE_DEVICES must list one or more numeric GPU ordinals (got ${CUDA_VISIBLE_DEVICES:-<unset>})" >&2
    exit 1
  fi
  if [[ ! -f "$repo_root/tests/e2e_tests/suites/$suite_name.toml" ]]; then
    echo "Unknown suite: $suite_name" >&2
    exit 2
  fi

  mkdir -p "$output_dir"
  output_dir=$(cd "$output_dir" && pwd)

  local SUITE_COMPONENT=""
  local SUITE_CACHE_HASH=""
  local -a SUITE_POST_INSTALL_CMD=()
  eval "$(python3 "$repo_root/tests/e2e_tests/suite.py" eval-meta "$suite_name" "$output_dir")"

  local venv_dir="$venv_arg"
  if [[ "$venv_mode" == "cached" ]]; then
    venv_dir=$(ensure_cached_venv "$SUITE_COMPONENT" "$venv_arg" "$output_dir/cache-install")
  fi

  if [[ ! -f "$venv_dir/.rpent-ci-ready" ]]; then
    echo "Upstream install for $SUITE_COMPONENT is not ready at $venv_dir" >&2
    exit 1
  fi

  # Re-resolve per-output post_install_cmd with the actual venv python path.
  eval "$(python3 "$repo_root/tests/e2e_tests/suite.py" eval-meta "$suite_name" "$output_dir" "$venv_dir/bin/python")"
  if ((${#SUITE_POST_INSTALL_CMD[@]} > 0)); then
    local cmd_bin="${SUITE_POST_INSTALL_CMD[0]}"
    if [[ -x "$venv_dir/bin/$cmd_bin" ]]; then
      SUITE_POST_INSTALL_CMD[0]="$venv_dir/bin/$cmd_bin"
    fi
    # Serialize per-venv asset linking across concurrent eval jobs.
    python3 - "$venv_dir/.rpent-post-install.lock" "$install_timeout" "${SUITE_POST_INSTALL_CMD[@]}" <<'PY'
import fcntl
import os
import subprocess
import sys

lock_path, timeout_val, *cmd = sys.argv[1:]
fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
try:
    fcntl.flock(fd, fcntl.LOCK_EX)
    subprocess.run(["timeout", "--kill-after=2m", timeout_val, *cmd], check=True)
finally:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
PY
  fi

  run_gpu_pytest "$suite_name" "$case_name" "$output_dir" "$venv_dir/bin/python" tests/e2e_tests/test_suites.py
}

run_declarative_suite_legacy() {
  local name=$1
  local output_dir=$2
  local venv_root=$3

  if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "CUDA_VISIBLE_DEVICES must list one or more numeric GPU ordinals" >&2
    exit 1
  fi
  if [[ -e "$output_dir" || -e "$venv_root" ]]; then
    echo "Output and venv paths must not already exist" >&2
    exit 1
  fi

  mkdir -p "$output_dir" "$venv_root"
  output_dir=$(cd "$output_dir" && pwd)
  venv_root=$(cd "$venv_root" && pwd)

  cleanup_legacy() {
    if [[ -d "$venv_root" ]]; then
      find "$venv_root" -depth -delete
    fi
  }
  trap cleanup_legacy EXIT

  local SUITE_COMPONENT=""
  local SUITE_CACHE_HASH=""
  local -a SUITE_POST_INSTALL_CMD=()
  eval "$(python3 "$repo_root/tests/e2e_tests/suite.py" eval-meta "$name" "$output_dir")"

  local venv_dir="$venv_root/$name"
  build_component_venv "$SUITE_COMPONENT" "$output_dir" "$venv_dir" 0

  eval "$(python3 "$repo_root/tests/e2e_tests/suite.py" eval-meta "$name" "$output_dir" "$venv_dir/bin/python")"
  if ((${#SUITE_POST_INSTALL_CMD[@]} > 0)); then
    local cmd_bin="${SUITE_POST_INSTALL_CMD[0]}"
    if [[ -x "$venv_dir/bin/$cmd_bin" ]]; then
      SUITE_POST_INSTALL_CMD[0]="$venv_dir/bin/$cmd_bin"
    fi
    timeout --kill-after=2m "$install_timeout" "${SUITE_POST_INSTALL_CMD[@]}"
  fi

  run_gpu_pytest "$name" "" "$output_dir" "$venv_dir/bin/python" tests/e2e_tests/test_suites.py
}

require_host_tools
cd "$repo_root"

case "${1:-}" in
  install)
    if [[ $# -lt 4 || $# -gt 5 ]]; then
      echo "Usage: $0 install <component> <output-dir> <venv-dir> [--portable]" >&2
      exit 2
    fi
    portable_flag=0
    if [[ "${5:-}" == "--portable" || "${RPENT_CI_PORTABLE_VENV:-0}" == "1" ]]; then
      portable_flag=1
    fi
    build_component_venv "$2" "$3" "$4" "$portable_flag"
    ;;
  prepare-cache)
    if [[ $# -ne 4 ]]; then
      echo "Usage: $0 prepare-cache <component> <output-dir> <cache-root>" >&2
      exit 2
    fi
    ensure_cached_venv "$2" "$4" "$3" >/dev/null
    ;;
  eval)
    if [[ $# -ne 5 ]]; then
      echo "Usage: $0 eval <suite> <case> <output-dir> <venv-dir-or-cache-root>" >&2
      exit 2
    fi
    run_eval_subcommand "$2" "$3" "$4" "$5"
    ;;
  all)
    if [[ $# -ne 3 ]]; then
      echo "Usage: $0 all <output-dir> <venv-root>" >&2
      exit 2
    fi
    status=0
    while IFS= read -r suite; do
      [[ -n "$suite" ]] || continue
      bash "$repo_root/tests/e2e_tests/run_gpu_suite.sh" \
        "$suite" "$2/$suite" "$3/$suite" || status=1
    done < <(python3 "$repo_root/tests/e2e_tests/suite.py" list)
    exit "$status"
    ;;
  *)
    if [[ $# -ne 3 ]]; then
      echo "Usage:" >&2
      echo "  $0 <suite|all> <output-dir> <venv-root>" >&2
      echo "  $0 install <component> <output-dir> <venv-dir> [--portable]" >&2
      echo "  $0 prepare-cache <component> <output-dir> <cache-root>" >&2
      echo "  $0 eval <suite> <case> <output-dir> <venv-dir-or-cache-root>" >&2
      exit 2
    fi
    if [[ ! -f "$repo_root/tests/e2e_tests/suites/$1.toml" ]]; then
      echo "Unknown target: $1" >&2
      exit 2
    fi
    run_declarative_suite_legacy "$1" "$2" "$3"
    ;;
esac
