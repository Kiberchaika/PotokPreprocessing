#!/usr/bin/env python3
"""Stream MP3 files from a remote Blackbird WebDAV server, process them
with the audio pipeline (beats, vocal separation, ASR), and upload results.

Dataset schema components produced (see scheme.json):
  - beats    (*_beats.json)   – beat & downbeat timestamps
  - lyrics   (*_lyrics.json)  – ASR transcription with word timestamps
  - vocal    (*_voc.opus)     – isolated vocals
  - music    (*_music.opus)   – accompaniment

Usage:
    python process_remote_streaming.py
    python process_remote_streaming.py --server https://1.2.3.4:8085/ --dataset /path/to/dataset
    python process_remote_streaming.py --server https://1.2.3.4 --port 9090 --dataset /data/Music
    python process_remote_streaming.py --mode beats --batch-size 8
    python process_remote_streaming.py --mode roformer-asr
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import torch

from audio_pipeline import (
    load_beat_model,
    load_roformer_model,
    load_asr_model,
    detect_beats,
    separate_vocals,
    save_opus,
    transcribe_audio,
    logger,
    OPUS_BITRATE_VOCAL,
    OPUS_BITRATE_MUSIC,
)

from blackbird.streaming import StreamingPipeline


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SERVER_URL = "https://188.120.253.126:8085/"
DEFAULT_USERNAME = "blackbird"
DEFAULT_PASSWORD = "dataset"
DEFAULT_SSH_KEY = "dev-233158-kiberchaika.pem"
DEFAULT_DATASET_PATH = "/home/k4/Datasets/Music_Part1.01_Test"
DEFAULT_MODE = "all"
DEFAULT_BATCH_SIZE = 4
DEFAULT_PREFETCH_WORKERS = 4
DEFAULT_UPLOAD_WORKERS = 4
DEFAULT_WORK_DIR = "/tmp/blackbird_processing"

COMPONENTS = ["mp3"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Stream MP3 from Blackbird WebDAV, process with audio pipeline, upload results")
    p.add_argument("--server", default=DEFAULT_SERVER_URL,
                   help=f"WebDAV server URL (default: {DEFAULT_SERVER_URL})")
    p.add_argument("--port", type=int, default=None,
                   help="Override server port (replaces port in --server URL)")
    p.add_argument("--username", default=DEFAULT_USERNAME,
                   help=f"WebDAV username (default: {DEFAULT_USERNAME})")
    p.add_argument("--password", default=DEFAULT_PASSWORD,
                   help=f"WebDAV password (default: {DEFAULT_PASSWORD})")
    p.add_argument("--ssh-key", default=DEFAULT_SSH_KEY,
                   help=f"SSH key for remote reindex (default: {DEFAULT_SSH_KEY})")
    p.add_argument("--dataset", default=DEFAULT_DATASET_PATH,
                   help=f"Remote dataset path (default: {DEFAULT_DATASET_PATH})")
    p.add_argument("--mode", choices=["beats", "roformer-asr", "all"],
                   default=DEFAULT_MODE,
                   help=f"Processing mode (default: {DEFAULT_MODE})")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"Items per take() batch (default: {DEFAULT_BATCH_SIZE})")
    p.add_argument("--work-dir", default=DEFAULT_WORK_DIR,
                   help=f"Local work directory (default: {DEFAULT_WORK_DIR})")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def remote_reindex(ssh_key: str, ssh_host: str, dataset_path: str) -> None:
    """SSH into the server and run blackbird reindex, wait for completion."""
    cmd = [
        "ssh", "-i", ssh_key,
        "-o", "StrictHostKeyChecking=no",
        f"root@{ssh_host}",
        f"source /home/k4/.venv/bin/activate && blackbird reindex '{dataset_path}'",
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


def submit_and_log(pipeline, item, result_path: Path, remote_name: str,
                   stats: dict) -> None:
    """Submit a result file for upload and accumulate stats."""
    file_size = result_path.stat().st_size
    t0 = time.time()
    pipeline.submit_result(
        item=item,
        result_path=result_path,
        remote_name=remote_name,
    )
    ul_time = time.time() - t0
    stats["upload_bytes"] += file_size
    stats["upload_time"] += ul_time
    print(f"     -> queued {remote_name} ({format_size(file_size)})")


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Build server URL with optional port override
    server_url = args.server
    if args.port is not None:
        parsed = urlparse(server_url)
        server_url = urlunparse(parsed._replace(netloc=f"{parsed.hostname}:{args.port}"))

    # Extract SSH host from server URL
    ssh_host = urlparse(server_url).hostname

    mode = args.mode
    batch_size = args.batch_size
    queue_size = batch_size * 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 0: load models
    logger.info(f"Loading models for mode={mode} on {device}...")
    t_load = time.perf_counter()
    beat_model = roformer_model = asr_model = None
    if mode in ("beats", "all"):
        beat_model = load_beat_model(device)
    if mode in ("roformer-asr", "all"):
        roformer_model = load_roformer_model(device)
        asr_model = load_asr_model(device)
    logger.info(f"Models loaded in {time.perf_counter() - t_load:.1f}s")

    # Step 1: reindex on the server so we get a fresh index
    remote_reindex(args.ssh_key, ssh_host, args.dataset)

    # Step 2: connect and stream with updated index
    print(f"Connecting to {server_url} ...")
    print(f"Components: {COMPONENTS}")
    print(f"Mode:       {mode}")
    print(f"Dataset:    {args.dataset}")
    print(f"Work dir:   {args.work_dir}")
    print()

    pipeline = StreamingPipeline(
        url=server_url,
        username=args.username,
        password=args.password,
        components=COMPONENTS,
        queue_size=queue_size,
        prefetch_workers=DEFAULT_PREFETCH_WORKERS,
        upload_workers=DEFAULT_UPLOAD_WORKERS,
        work_dir=args.work_dir,
    )

    processed = 0
    skipped = 0
    stats = {
        "download_bytes": 0,
        "download_time": 0.0,
        "upload_bytes": 0,
        "upload_time": 0.0,
    }
    pipeline_start = time.time()

    run_beats = mode in ("beats", "all")
    run_sep = mode in ("roformer-asr", "all")

    with pipeline:
        while True:
            # Measure download (take) time
            t0 = time.time()
            items = pipeline.take(count=batch_size)
            dl_time = time.time() - t0

            if not items:
                break

            batch_dl_bytes = sum(
                item.local_path.stat().st_size for item in items
                if item.local_path.exists()
            )
            stats["download_bytes"] += batch_dl_bytes
            stats["download_time"] += dl_time

            print(f"  -- batch downloaded: {len(items)} files, "
                  f"{format_size(batch_dl_bytes)}, "
                  f"{dl_time:.2f}s, "
                  f"{format_speed(batch_dl_bytes, dl_time)}")

            for item in items:
                artist = item.metadata.get("artist", "?")
                album = item.metadata.get("album", "?")
                track = item.metadata.get("track", "?")
                mp3_path = item.local_path
                file_size = mp3_path.stat().st_size if mp3_path.exists() else 0

                print(f"[{processed + 1}] {artist} / {album} / {track}  "
                      f"({format_size(file_size)})")

                try:
                    stem = mp3_path.stem
                    parent = mp3_path.parent

                    # 1) Beat detection -> _beats.json
                    if run_beats:
                        t1 = time.perf_counter()
                        beats, downbeats = detect_beats(beat_model, str(mp3_path))
                        ms = (time.perf_counter() - t1) * 1000
                        logger.info(f"[{track}] Beat detection: {len(beats)} beats, "
                                    f"{len(downbeats)} downbeats [{ms:.0f}ms]")

                        beats_data = {"beats": beats, "downbeats": downbeats}
                        beats_path = parent / f"{stem}_beats.json"
                        beats_path.write_text(json.dumps(beats_data, indent=2))
                        submit_and_log(pipeline, item, beats_path,
                                       f"{track}_beats.json", stats)

                    # 2) Vocal separation -> _voc.opus + _music.opus
                    #    ASR on vocals  -> _lyrics.json
                    if run_sep:
                        t1 = time.perf_counter()
                        vocals_np, music_np = separate_vocals(
                            roformer_model, str(mp3_path), device)
                        ms_sep = (time.perf_counter() - t1) * 1000
                        logger.info(f"[{track}] Separation [{ms_sep:.0f}ms]")

                        # Encode to Opus
                        t2 = time.perf_counter()
                        vocal_path = parent / f"{stem}_voc.opus"
                        music_path = parent / f"{stem}_music.opus"
                        save_opus(vocals_np, str(vocal_path), bitrate=OPUS_BITRATE_VOCAL)
                        save_opus(music_np, str(music_path), bitrate=OPUS_BITRATE_MUSIC)
                        ms_opus = (time.perf_counter() - t2) * 1000
                        logger.info(f"[{track}] Opus saved [{ms_opus:.0f}ms]")

                        submit_and_log(pipeline, item, vocal_path,
                                       f"{track}_voc.opus", stats)
                        submit_and_log(pipeline, item, music_path,
                                       f"{track}_music.opus", stats)

                        # ASR transcription
                        t3 = time.perf_counter()
                        asr_out = transcribe_audio(asr_model, vocals_np)
                        ms_asr = (time.perf_counter() - t3) * 1000
                        logger.info(f"[{track}] ASR: {len(asr_out['text'])} chars [{ms_asr:.0f}ms]")

                        lyrics_data = {
                            "text": asr_out["text"],
                            "segments": asr_out["segments"],
                            "words": asr_out["words"],
                        }
                        lyrics_path = parent / f"{stem}_lyrics.json"
                        lyrics_path.write_text(
                            json.dumps(lyrics_data, ensure_ascii=False, indent=2))
                        submit_and_log(pipeline, item, lyrics_path,
                                       f"{track}_lyrics.json", stats)

                    processed += 1

                except Exception as e:
                    logger.error(f"[{track}] Processing failed: {e}", exc_info=True)
                    print(f"     ERROR: {e} — skipping")
                    pipeline.skip(item)
                    skipped += 1

    total_time = time.time() - pipeline_start

    print()
    print("=" * 60)
    print(f"  Mode      : {mode}")
    print(f"  Processed : {processed} files")
    print(f"  Skipped   : {skipped} files")
    print(f"  Total time: {total_time:.1f}s")
    print()
    print(f"  Downloaded: {format_size(stats['download_bytes'])} "
          f"in {stats['download_time']:.1f}s "
          f"({format_speed(stats['download_bytes'], stats['download_time'])})")
    print(f"  Uploaded  : {format_size(stats['upload_bytes'])} "
          f"in {total_time:.1f}s "
          f"({format_speed(stats['upload_bytes'], total_time)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
