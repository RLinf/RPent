#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$RPENT_REPO_ROOT"
exec "${RPENT_PYTHON:-python}" -m robots.rl.rollout_options "$@"
