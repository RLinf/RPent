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
#
# Run repository structure checks from a fresh checkout.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

case "${1:-}" in
  layout) module=tests.tooling.layout ;;
  *)
    echo "usage: $0 layout [args...]" >&2
    exit 2
    ;;
esac
shift

exec uv run --no-project --python 3.11 --with packaging -- python -m "$module" "$@"
