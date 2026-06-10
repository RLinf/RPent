#!/usr/bin/env bash
# Compatibility wrapper. The maintained installer lives at the repository root.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$ROOT_DIR/install_libero_pro_plus.sh" "$@"
