#!/bin/bash
# Sync the live memory dir into the in-repo snapshot.
#
# Run any time the live /root/.claude/.../memory/ has gained a new
# entry relevant to these experiments. The snapshot is what fresh
# clones see; the live dir continues accumulating between syncs.
#
# The snapshot lives under the per-env memory folder
# (resources/<env>/memory); set ENV_NAME to target a different env.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIVE=${MEMORY_LIVE}
ENV_NAME=${ENV_NAME:-libero}
SNAPSHOT="$SCRIPT_DIR/../resources/$ENV_NAME/memory"

if [ ! -d "$LIVE" ]; then
    echo "[sync_memory] live memory dir not found: $LIVE" >&2
    echo "[sync_memory] (set MEMORY_LIVE=... if it's elsewhere)" >&2
    exit 1
fi

mkdir -p "$SNAPSHOT"

# Only sync .md files (and keep the snapshot's README.md if present)
RM_LIST=$(find "$SNAPSHOT" -maxdepth 1 -name "*.md" ! -name "README.md")
[ -n "$RM_LIST" ] && rm -f $RM_LIST

cp "$LIVE"/*.md "$SNAPSHOT/"
n=$(ls "$SNAPSHOT" | grep -c '\.md$')
echo "[sync_memory] copied $n .md files from $LIVE -> $SNAPSHOT"
echo "[sync_memory] resources/ is not tracked in git; a maintainer must publish"
echo "[sync_memory] the update to the HuggingFace dataset (see docs: Memory Management)"