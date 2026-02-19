#!/usr/bin/env python3
"""Stream MP3 files from a remote Blackbird WebDAV server, process them,
create .mir.json + _vocal.mp3, and upload both back.

Dataset schema on the server should have at least:
  - mp3 (pattern: *.mp3)

Optionally add these so uploaded results are recognized by the dataset:
  blackbird schema add <dataset_path> "mir.json"  "*.mir.json"
  blackbird schema add <dataset_path> "vocal.mp3" "*_vocal.mp3"

Usage:
    python process_remote_streaming.py

Configuration:
    Edit SERVER_URL, USERNAME, PASSWORD below to match your remote server.
    See setup_remote_server.sh for how to set up the server side.
"""

import json
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

from blackbird.streaming import StreamingPipeline


# ---------------------------------------------------------------------------
# Configuration — edit these to match your remote server
# ---------------------------------------------------------------------------

SERVER_URL = "https://188.120.253.126:8085/"   # remote WebDAV URL
USERNAME = "blackbird"
PASSWORD = "dataset"

# SSH settings for remote reindex
SSH_KEY = "dev-233158-kiberchaika.pem"
SSH_HOST = "188.120.253.126"
REMOTE_DATASET_PATH = "/home/k4/Datasets/Music_Part1.01_Test"

# Which component to stream (must exist in dataset schema)
COMPONENTS = ["mp3"]

# Processing settings
BATCH_SIZE = 2           # items per take()
QUEUE_SIZE = BATCH_SIZE * 8           # prefetch buffer
PREFETCH_WORKERS = 4     # download threads
UPLOAD_WORKERS = 4       # upload threads
WORK_DIR = "/tmp/blackbird_processing"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def remote_reindex() -> None:
    """SSH into the server and run blackbird reindex, wait for completion."""
    cmd = [
        "ssh", "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        f"root@{SSH_HOST}",
        f"source /home/k4/.venv/bin/activate && blackbird reindex '{REMOTE_DATASET_PATH}'",
    ]
    print(f"Running remote reindex: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"Reindex FAILED (exit code {result.returncode}):")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout)
    print("Remote reindex completed.\n")


def format_size(size_bytes: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_speed(size_bytes: int, elapsed_sec: float) -> str:
    """Format transfer speed as Mbit/s."""
    if elapsed_sec <= 0:
        return "- Mbit/s"
    mbits = (size_bytes * 8) / (1024 * 1024)
    return f"{mbits / elapsed_sec:.2f} Mbit/s"


# ---------------------------------------------------------------------------
# Processing functions — replace with your real analysis
# ---------------------------------------------------------------------------

def analyze_mp3(mp3_path: Path) -> dict:
    """Analyze an MP3 file and return metadata dict.

    Replace this with your actual MIR logic, e.g.:
      - librosa / essentia for audio features
      - whisper for lyrics transcription

    This placeholder reads the file and computes basic stats.
    """
    data = mp3_path.read_bytes()

    result = {
        "filename": mp3_path.name,
        "file_size": len(data),
        "md5": hashlib.md5(data).hexdigest(),
        # Add your real features here, e.g.:
        # "bpm": librosa.beat.tempo(y, sr=sr)[0],
        # "key": estimated_key,
        # "loudness_lufs": loudness,
    }
    return result


def extract_vocals(mp3_path: Path) -> Path:
    """Extract vocals from an MP3 file.

    Replace this with your actual vocal separation, e.g.:
      - demucs.separate.main(["--two-stems", "vocals", str(mp3_path)])
      - spleeter separate -p spleeter:2stems -o output audio.mp3

    This placeholder copies the source as a stand-in.
    """
    vocal_path = mp3_path.with_name(mp3_path.stem + "_vocal.mp3")

    # ---- Replace this block with real separation ----
    shutil.copy2(mp3_path, vocal_path)
    # -------------------------------------------------

    return vocal_path


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def main() -> None:
    # Step 1: reindex on the server so we get a fresh index
    remote_reindex()

    # Step 2: connect and stream with updated index
    print(f"Connecting to {SERVER_URL} ...")
    print(f"Components: {COMPONENTS}")
    print(f"Work dir:   {WORK_DIR}")
    print()

    pipeline = StreamingPipeline(
        url=SERVER_URL,
        username=USERNAME,
        password=PASSWORD,
        components=COMPONENTS,
        queue_size=QUEUE_SIZE,
        prefetch_workers=PREFETCH_WORKERS,
        upload_workers=UPLOAD_WORKERS,
        work_dir=WORK_DIR,
    )

    processed = 0
    skipped = 0
    total_download_bytes = 0
    total_upload_bytes = 0
    total_download_time = 0.0
    total_upload_time = 0.0
    pipeline_start = time.time()

    with pipeline:
        while True:
            # Measure download (take) time
            t0 = time.time()
            items = pipeline.take(count=BATCH_SIZE)
            dl_time = time.time() - t0

            if not items:
                break

            batch_dl_bytes = sum(
                item.local_path.stat().st_size for item in items
                if item.local_path.exists()
            )
            total_download_bytes += batch_dl_bytes
            total_download_time += dl_time

            print(f"  -- batch downloaded: {len(items)} files, "
                  f"{format_size(batch_dl_bytes)}, "
                  f"{dl_time:.2f}s, "
                  f"{format_speed(batch_dl_bytes, dl_time)}")

            for item in items:
                artist = item.metadata.get("artist", "?")
                album = item.metadata.get("album", "?")
                track = item.metadata.get("track", "?")
                file_size = item.local_path.stat().st_size if item.local_path.exists() else 0

                print(f"[{processed + 1}] {artist} / {album} / {track}  "
                      f"({format_size(file_size)})")

                try:
                    # 1) MIR analysis -> .mir.json
                    result = analyze_mp3(item.local_path)
                    json_path = item.local_path.with_suffix(".mir.json")
                    json_path.write_text(json.dumps(result, indent=2))
                    json_size = json_path.stat().st_size

                    t1 = time.time()
                    pipeline.submit_result(
                        item=item,
                        result_path=json_path,
                        remote_name=f"{track}.mir.json",
                    )
                    ul_time_json = time.time() - t1
                    total_upload_bytes += json_size
                    total_upload_time += ul_time_json
                    print(f"     -> queued {track}.mir.json ({format_size(json_size)})")

                    # 2) Vocal separation -> _vocal.mp3
                    vocal_path = extract_vocals(item.local_path)
                    vocal_size = vocal_path.stat().st_size

                    t2 = time.time()
                    pipeline.submit_result(
                        item=item,
                        result_path=vocal_path,
                        remote_name=f"{track}_vocal.mp3",
                    )
                    ul_time_vocal = time.time() - t2
                    total_upload_bytes += vocal_size
                    total_upload_time += ul_time_vocal
                    print(f"     -> queued {track}_vocal.mp3 ({format_size(vocal_size)})")

                    processed += 1
                    print("sleep")
                    time.sleep(5)
                    print("unsleep")


                except Exception as e:
                    print(f"     ERROR: {e} — skipping")
                    pipeline.skip(item)
                    skipped += 1

    total_time = time.time() - pipeline_start

    print()
    print("=" * 60)
    print(f"  Processed : {processed} files")
    print(f"  Skipped   : {skipped} files")
    print(f"  Total time: {total_time:.1f}s")
    print()
    print(f"  Downloaded: {format_size(total_download_bytes)} "
          f"in {total_download_time:.1f}s "
          f"({format_speed(total_download_bytes, total_download_time)})")
    print(f"  Uploaded  : {format_size(total_upload_bytes)} "
          f"in {total_time:.1f}s "
          f"({format_speed(total_upload_bytes, total_time)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
