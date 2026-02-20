#!/usr/bin/env bash
# ============================================================================
# Setup Blackbird dataset + WebDAV on a remote server
#
# Run this on the remote machine where the MP3 files live:
#   /home/k4/Datasets/Music_Part1.01_Test/
#
# Prerequisites:
#   - pip install blackbird  (or install from this repo)
#   - Ubuntu (for the WebDAV nginx setup)
# ============================================================================

set -euo pipefail

DATASET_PATH="/home/k4/Datasets/Music_Part1.01_Part01"
WEBDAV_PORT=8091
WEBDAV_USER="blackbird"
WEBDAV_PASS="dataset"

echo "=== Step 0: Install nginx with WebDAV support ==="
sudo apt-get update -qq
sudo apt-get install -y -qq nginx libnginx-mod-http-dav-ext

SCHEMA_FILE="$DATASET_PATH/.blackbird/schema.json"

if [[ -f "$SCHEMA_FILE" ]]; then
    echo "=== Schema already exists at $SCHEMA_FILE, skipping discovery ==="
else
    echo "=== Step 1: Discover schema from MP3 files ==="
    blackbird schema discover "$DATASET_PATH" --test-run

    echo ""
    echo "=== Step 2: Save discovered schema ==="
    blackbird schema discover "$DATASET_PATH"
fi

echo ""
echo "=== Step 3: Add result components for streaming uploads ==="
# These are produced by process_remote_streaming.py and uploaded back.
# Adding them to the schema makes them visible in stats/find-tracks/reindex.
#blackbird schema add "$DATASET_PATH" "mir.json"  "*.mir.json"
#blackbird schema add "$DATASET_PATH" "vocal.mp3" "*_vocal.mp3"

echo ""
echo "=== Step 4: Build index ==="
blackbird reindex "$DATASET_PATH"

echo ""
echo "=== Step 5: Check dataset stats ==="
blackbird stats "$DATASET_PATH"

echo ""
echo "=== Step 6: Setup WebDAV server ==="
blackbird webdav setup "$DATASET_PATH" \
    --port "$WEBDAV_PORT" \
    --username "$WEBDAV_USER" \
    --password "$WEBDAV_PASS" \
    --non-interactive

echo ""
echo "=== Step 7: Verify WebDAV is running ==="
blackbird webdav list

echo ""
echo "============================================================"
echo "  WebDAV server is running!"
echo ""
echo "  URL:  webdav://${WEBDAV_USER}:${WEBDAV_PASS}@<SERVER_IP>:${WEBDAV_PORT}/"
echo ""
echo "  From the client machine, run:"
echo "    blackbird stats webdav://${WEBDAV_USER}:${WEBDAV_PASS}@<SERVER_IP>:${WEBDAV_PORT}/"
echo "    python examples/process_remote_streaming.py"
echo ""
echo "  After processing, reindex on the server to pick up new files:"
echo "    blackbird reindex ${DATASET_PATH}"
echo "============================================================"
