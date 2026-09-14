#!/usr/bin/env bash
# Sequential GPU processing of Music_Part1.0N slices on 188.120.253.126.
#
# Already-complete tracks are skipped by process_remote_streaming.py:
#   --mode all           needs beats + vocal + music + lyrics
#   --mode roformer-asr  needs vocal + music + lyrics
#   --mode asr           needs lyrics with real text (empty stubs are re-done)
#   --mode beats         needs beats
#
# Usage:
#   ./run_slices.sh --gpu 0                # 1.01 Part01 → Part04 on GPU 0
#   ./run_slices.sh --set 1.02 --gpu 0 01  # Music_Part1.02_Part01
#   ./run_slices.sh --set 1.02 --gpu 1 02
#   ./run_slices.sh --gpu 1 03
#   ./run_slices.sh --gpu 1 --mode beats 02
#   ./run_slices.sh --set 1.02 --mode asr --gpu 0 01   # port 8095, lyrics only
#   ./run_slices.sh --test                 # include Music_Part1.01_Test
#   ./run_slices.sh --dry-run 01 02
#
# Env:
#   PYTHON, HOST, BATCH_SIZE, KEEP_GOING=1, OPUS_WORKERS=8

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HOST="${HOST:-188.120.253.126}"
# nginx on these ports is plain HTTP (https handshake fails)
SCHEME="${SCHEME:-http}"
PYTHON="${PYTHON:-python3}"
MODE="all"
GPU="${GPU:-0}"
SET="${SET:-1.01}"
BATCH_SIZE="${BATCH_SIZE:-4}"
INCLUDE_TEST=0
DRY_RUN=0
KEEP_GOING="${KEEP_GOING:-0}"

usage() {
  sed -n '2,19p' "$0" | sed 's/^# \?//'
  exit 1
}

PARTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:?}"; shift 2 ;;
    --mode) MODE="${2:?}"; shift 2 ;;
    --set) SET="${2:?}"; shift 2 ;;
    --test) INCLUDE_TEST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --keep-going) KEEP_GOING=1; shift ;;
    -h|--help) usage ;;
    01|02|03|04|test) PARTS+=("$1"); shift ;;
    1|2|3|4) PARTS+=("0$1"); shift ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

case "$SET" in
  1.01)
    declare -A PORT=(
      [01]=8091 [02]=8092 [03]=8093 [04]=8094 [test]=8085
    )
    declare -A DATASET=(
      [01]="/home/k4/Datasets/Music_Part1.01_Part01"
      [02]="/home/k4/Datasets/Music_Part1.01_Part02"
      [03]="/home/k4/Datasets/Music_Part1.01_Part03"
      [04]="/home/k4/Datasets/Music_Part1.01_Part04"
      [test]="/home/k4/Datasets/Music_Part1.01_Test"
    )
    DEFAULT_PARTS=(01 02 03 04)
    ;;
  1.02)
    declare -A PORT=(
      [01]=8095 [02]=8096
    )
    declare -A DATASET=(
      [01]="/home/k4/Datasets/Music_Part1.02_Part01"
      [02]="/home/k4/Datasets/Music_Part1.02_Part02"
    )
    DEFAULT_PARTS=(01 02)
    ;;
  *)
    echo "Invalid --set $SET (1.01|1.02)" >&2
    exit 1
    ;;
esac

if [[ ${#PARTS[@]} -eq 0 ]]; then
  PARTS=("${DEFAULT_PARTS[@]}")
fi
if [[ "$INCLUDE_TEST" -eq 1 ]]; then
  if [[ -z "${PORT[test]+x}" ]]; then
    echo "--test is only valid with --set 1.01" >&2
    exit 1
  fi
  PARTS+=(test)
fi

case "$MODE" in
  all|beats|roformer-asr|asr) ;;
  *) echo "Invalid --mode $MODE (all|beats|roformer-asr|asr)" >&2; exit 1 ;;
esac

case "$GPU" in
  0|1) ;;
  *) echo "Invalid --gpu $GPU (use 0 or 1)" >&2; exit 1 ;;
esac
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ ! -f "$ROOT/dev-233158-kiberchaika.pem" ]]; then
  echo "Missing SSH key: $ROOT/dev-233158-kiberchaika.pem" >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "Host     $HOST"
echo "Set      $SET"
echo "GPU      $GPU  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Mode     $MODE"
echo "Parts    ${PARTS[*]}"
echo "Python   $PYTHON ($("$PYTHON" -c 'import sys; print(sys.executable)'))"
echo "Batch    $BATCH_SIZE"
echo

run_part() {
  local id="$1"
  local port="${PORT[$id]}"
  local dataset="${DATASET[$id]}"
  local server="${SCHEME}://${HOST}:${port}/"
  local work="/tmp/blackbird_processing_${SET}_${id}"
  local log="$ROOT/logs/slice_${SET}_${id}_${MODE}_${STAMP}.log"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Part $id   port $port"
  echo "  dataset $dataset"
  echo "  server  $server"
  echo "  log     $log"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  local cmd=(
    "$PYTHON" "$ROOT/process_remote_streaming.py"
    --server "$server"
    --dataset "$dataset"
    --mode "$MODE"
    --batch-size "$BATCH_SIZE"
    --ssh-key "$ROOT/dev-233158-kiberchaika.pem"
    --work-dir "$work"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  DRY RUN: CUDA_VISIBLE_DEVICES=%q' "$GPU"
    printf ' %q' "${cmd[@]}"; echo
    return 0
  fi

  mkdir -p "$work"
  if "${cmd[@]}" 2>&1 | tee "$log"; then
    echo "Part $id finished OK"
  else
    local rc=${PIPESTATUS[0]}
    echo "Part $id failed (exit $rc). Log: $log" >&2
    if [[ "$KEEP_GOING" -ne 1 ]]; then
      return "$rc"
    fi
  fi
}

for id in "${PARTS[@]}"; do
  if [[ -z "${PORT[$id]+x}" ]]; then
    echo "Unknown part id: $id" >&2
    exit 1
  fi
  run_part "$id"
done
