#!/usr/bin/env bash
# Music_Part1.02: next 1.5 TB after .01 → rsync to VDS → split into 4 parts → WebDAV.
#
# Ports 8091–8094 are already Music_Part1.01_Part01–04.
# This slice uses 8095–8098.
#
# From this directory, with NAS mounted:
#   ./upload_part1_02.sh --filelist-only   # scan NAS, write filelist_part1.02.txt
#   ./upload_part1_02.sh                   # filelist (if missing) + rsync + remote split + webdav
#   ./upload_part1_02.sh --rsync-only      # rsync existing filelist, no split
#   ./upload_part1_02.sh --setup-only      # on already-split VDS dirs: schema/reindex/webdav
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HOST="${HOST:-188.120.253.126}"
KEY="${KEY:-$ROOT/dev-233158-kiberchaika.pem}"
BASE="${BASE:-/media/k4_nas/disk1/Datasets/Music_Part1}"
LIST="$ROOT/filelist_part1.02.txt"
REMOTE_STAGING="/home/k4/Datasets/Music_Part1.02"
SCHEMA_SRC="/home/k4/Datasets/Music_Part1.01_Part01/.blackbird/schema.json"
PYTHON="${PYTHON:-python3}"

FILELIST_ONLY=0
RSYNC_ONLY=0
SETUP_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --filelist-only) FILELIST_ONLY=1; shift ;;
    --rsync-only) RSYNC_ONLY=1; shift ;;
    --setup-only) SETUP_ONLY=1; shift ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

ssh_vds() {
  ssh -i "$KEY" -o StrictHostKeyChecking=no root@"$HOST" "$@"
}

need_key() {
  [[ -f "$KEY" ]] || { echo "Missing SSH key: $KEY" >&2; exit 1; }
}

make_filelist() {
  echo "=== filelist slice 2 (next 1.5 TB after .01, same path sort) ==="
  "$PYTHON" "$ROOT/make_filelists_for_transfer.py" --slice 2 --base "$BASE" -o "$LIST"
}

do_rsync() {
  need_key
  [[ -s "$LIST" ]] || { echo "Empty/missing $LIST — run --filelist-only first" >&2; exit 1; }
  echo "=== rsync → $HOST:$REMOTE_STAGING ==="
  ssh_vds "mkdir -p '$REMOTE_STAGING'"
  rsync -avz --progress --files-from="$LIST" \
    -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
    "$BASE/" \
    "root@${HOST}:${REMOTE_STAGING}"
}

do_split_and_webdav() {
  need_key
  echo "=== remote split into Music_Part1.02_Part01..04 + WebDAV 8095–8098 ==="
  scp -q -i "$KEY" -o StrictHostKeyChecking=no \
    "$ROOT/split_dataset.py" "root@${HOST}:/tmp/split_dataset.py"
  ssh_vds bash -s <<'REMOTE'
set -euo pipefail
source /home/k4/.venv/bin/activate
STAGING=/home/k4/Datasets/Music_Part1.02
SCHEMA_SRC=/home/k4/Datasets/Music_Part1.01_Part01/.blackbird/schema.json
USER=blackbird
PASS=dataset

if [[ ! -d "$STAGING" ]]; then
  echo "Missing $STAGING" >&2
  exit 1
fi

# If already split, staging may be gone and Part0N exist.
if [[ -d /home/k4/Datasets/Music_Part1.02_Part01 ]]; then
  echo "Parts already exist, skip split"
else
  python3 /tmp/split_dataset.py "$STAGING" 4
fi

for i in 01 02 03 04; do
  dest=/home/k4/Datasets/Music_Part1.02_Part$i
  port=$((8094 + 10#$i))   # 8095..8098
  mkdir -p "$dest/.blackbird"
  if [[ -f "$SCHEMA_SRC" ]]; then
    cp -n "$SCHEMA_SRC" "$dest/.blackbird/schema.json" || true
  fi
  echo "=== reindex $dest ==="
  blackbird reindex "$dest"
  echo "=== webdav $dest :$port ==="
  blackbird webdav setup "$dest" \
    --port "$port" \
    --username "$USER" \
    --password "$PASS" \
    --non-interactive
done

blackbird webdav list
df -h /home/k4/Datasets
REMOTE
}

if [[ "$FILELIST_ONLY" -eq 1 ]]; then
  make_filelist
  exit 0
fi

if [[ "$SETUP_ONLY" -eq 1 ]]; then
  do_split_and_webdav
  exit 0
fi

if [[ "$RSYNC_ONLY" -eq 1 ]]; then
  [[ -s "$LIST" ]] || make_filelist
  do_rsync
  exit 0
fi

[[ -s "$LIST" ]] || make_filelist
do_rsync
do_split_and_webdav

echo
echo "Done. Process later with process_remote_streaming.py, e.g.:"
echo "  python process_remote_streaming.py --server http://$HOST:8095/ \\"
echo "    --dataset /home/k4/Datasets/Music_Part1.02_Part01 --mode all"
