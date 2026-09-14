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
    python process_remote_streaming.py --mode asr --port 8095
    python process_remote_streaming.py --set-aware  # see run_slices.sh --set 1.02 --mode asr 01
"""


"""

transformers==4.48.3
huggingface_hub==0.36.2

"""

import argparse
import json
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlparse, urlunparse

import librosa
import requests
import torch
import torchaudio
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

from audio_pipeline import (
    load_beat_model,
    load_roformer_model,
    load_asr_model,
    detect_beats,
    separate_vocals,
    transcribe_audio,
    transcribe_file,
    logger,
    OPUS_BITRATE_VOCAL,
    OPUS_BITRATE_MUSIC,
    OpusEncodePool,
)

from blackbird.streaming import StreamingPipeline


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SERVER_URL = "https://188.120.253.126:8091/"
DEFAULT_USERNAME = "blackbird"
DEFAULT_PASSWORD = "dataset"
DEFAULT_SSH_KEY = "dev-233158-kiberchaika.pem"
DEFAULT_DATASET_PATH = "/home/k4/Datasets/Music_Part1.01_Part01"
DEFAULT_MODE = "all"
DEFAULT_BATCH_SIZE = 4
DEFAULT_PREFETCH_WORKERS = 4
DEFAULT_UPLOAD_WORKERS = 4
DEFAULT_WORK_DIR = "/tmp/blackbird_processing"
MIN_LYRICS_BYTES = 80  # empty ASR stub is ~49 bytes

COMPONENTS_BY_MODE = {
    "beats": ["mp3"],
    "roformer-asr": ["mp3"],
    "all": ["mp3"],
    "asr": ["vocal"],
}


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
    p.add_argument("--mode", choices=["beats", "roformer-asr", "all", "asr"],
                   default=DEFAULT_MODE,
                   help="beats | roformer-asr | all | asr (Parakeet on existing _voc.opus)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run ASR even if a non-empty *_lyrics.json already exists")
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


def complete_components(track_info) -> set:
    """Components that already exist with real content (empty lyrics stubs do not count)."""
    done = set()
    for name, path in track_info.files.items():
        size = track_info.file_sizes.get(path)
        if name == "lyrics" and (size is None or size < MIN_LYRICS_BYTES):
            continue
        done.add(name)
    return done


def source_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_voc", "_vocal", "_music"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def lyrics_relpath(remote_path: str) -> str:
    p = Path(remote_path)
    return str(p.with_name(f"{source_stem(p)}_lyrics.json")).replace("\\", "/")


def dav_content_length(session: requests.Session, base: str, rel: str) -> Optional[int]:
    url = f"{base.rstrip('/')}/{quote(rel.lstrip('/'), safe='/')}"
    try:
        resp = session.head(url, timeout=20, allow_redirects=True)
        if resp.status_code == 404:
            return None
        cl = resp.headers.get("Content-Length")
        if resp.status_code == 200 and cl is not None:
            return int(cl)
        resp = session.get(url, timeout=20, stream=True)
        try:
            if resp.status_code == 404:
                return None
            if resp.status_code != 200:
                return None
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl is not None else None
        finally:
            resp.close()
    except Exception:
        return None


def lyrics_already_done(session: requests.Session, base: str, remote_path: str) -> bool:
    size = dav_content_length(session, base, lyrics_relpath(remote_path))
    return size is not None and size >= MIN_LYRICS_BYTES


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
                   stats: dict, stats_lock: threading.Lock) -> None:
    """Submit a result file for upload and accumulate stats."""
    file_size = result_path.stat().st_size
    t0 = time.time()
    pipeline.submit_result(
        item=item,
        result_path=result_path,
        remote_name=remote_name,
    )
    ul_time = time.time() - t0
    with stats_lock:
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
    if mode == "asr":
        asr_model = load_asr_model(device)
    logger.info(f"Models loaded in {time.perf_counter() - t_load:.1f}s")

    components = COMPONENTS_BY_MODE[mode]

    # Step 1: reindex on the server so we get a fresh index
    remote_reindex(args.ssh_key, ssh_host, args.dataset)

    # Step 2: connect and stream with updated index
    print(f"Connecting to {server_url} ...")
    print(f"Components: {components}")
    print(f"Mode:       {mode}")
    print(f"Dataset:    {args.dataset}")
    print(f"Work dir:   {args.work_dir}")
    print()

    pipeline = StreamingPipeline(
        url=server_url,
        username=args.username,
        password=args.password,
        components=components,
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
    stats_lock = threading.Lock()
    pipeline_start = time.time()

    run_beats = mode in ("beats", "all")
    run_sep = mode in ("roformer-asr", "all")
    run_asr_only = mode == "asr"
    overwrite = args.overwrite

    dav = requests.Session()
    dav.auth = HTTPBasicAuth(args.username, args.password)
    dav.trust_env = False

    # Skip tracks that already have the outputs this mode would write.
    needed_components = set()
    if run_beats:
        needed_components.add("beats")
    if run_sep:
        needed_components.update(("vocal", "music", "lyrics"))
    if run_asr_only:
        needed_components.add("lyrics")

    with pipeline:
        # Pre-filter file list: remove tracks that already have all needed components
        idx = pipeline._index
        if idx and needed_components and not overwrite:
            # Build lookup: (artist, album, track) -> set of existing components
            track_components = {}
            for _, track_info in idx.tracks.items():
                key = (track_info.artist,
                       track_info.album_path.split("/")[-1],
                       track_info.base_name)
                track_components[key] = complete_components(track_info)

            original_count = len(pipeline._file_list)
            filtered = []
            for entry in pipeline._file_list:
                meta = entry["metadata"]
                key = (meta["artist"], meta["album"], meta["track"])
                existing = track_components.get(key, set())
                if needed_components.issubset(existing):
                    skipped += 1
                else:
                    filtered.append(entry)
            pipeline._file_list = filtered
            if skipped > 0:
                print(f"Skipped {skipped}/{original_count} tracks "
                      f"(already have {', '.join(sorted(needed_components))})")
                print(f"Remaining: {len(filtered)} tracks to process\n")

        total = len(pipeline._file_list)
        pbar = tqdm(total=total, desc="Processing", unit="file")

        with ExitStack() as stack:
            opus_pool = stack.enter_context(OpusEncodePool()) if run_sep else None
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

                tqdm.write(f"  -- batch downloaded: {len(items)} files, "
                           f"{format_size(batch_dl_bytes)}, "
                           f"{dl_time:.2f}s, "
                           f"{format_speed(batch_dl_bytes, dl_time)}")

                for item in items:
                    artist = item.metadata.get("artist", "?")
                    album = item.metadata.get("album", "?")
                    track = item.metadata.get("track", "?")
                    mp3_path = item.local_path
                    file_size = mp3_path.stat().st_size if mp3_path.exists() else 0

                    tqdm.write(f"[{processed + 1}] {artist} / {album} / {track}  "
                               f"({format_size(file_size)})")

                    try:
                        try:
                            info = torchaudio.info(str(mp3_path))
                            dur = info.num_frames / max(info.sample_rate, 1)
                            if dur > 4 * 3600:
                                dur = 0.0
                        except Exception:
                            dur = librosa.get_duration(path=str(mp3_path))
                        file_start = time.perf_counter()
                        stem = source_stem(mp3_path)
                        parent = mp3_path.parent
                        beats_path = None

                        if run_asr_only:
                            remote_lyrics = lyrics_relpath(item.remote_path)
                            local_lyrics = parent / f"{stem}_lyrics.json"
                            if not overwrite and (
                                (local_lyrics.is_file() and local_lyrics.stat().st_size >= MIN_LYRICS_BYTES)
                                or lyrics_already_done(dav, server_url, item.remote_path)
                            ):
                                tqdm.write(f"     skip ASR, exists: {remote_lyrics}")
                                pipeline.skip(item)
                                skipped += 1
                                pbar.update(1)
                                continue
                            t1 = time.perf_counter()
                            asr_out = transcribe_file(asr_model, str(mp3_path))
                            ms_asr = (time.perf_counter() - t1) * 1000
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
                                           f"{stem}_lyrics.json", stats, stats_lock)
                            elapsed = time.perf_counter() - file_start
                            dur_hours = dur / 3600
                            speed = elapsed / dur_hours if dur_hours > 0 else 0
                            tqdm.write(f"     done in {elapsed:.1f}s "
                                       f"({dur:.0f}s audio, {speed:.0f}s per hour of audio)")
                            processed += 1
                            pbar.update(1)
                            continue

                        # 1) Beat detection — write JSON now, upload after GPU
                        #    is done with the mp3 (upload workers delete the source).
                        if run_beats:
                            t1 = time.perf_counter()
                            beats, downbeats = detect_beats(beat_model, str(mp3_path))
                            ms = (time.perf_counter() - t1) * 1000
                            logger.info(f"[{track}] Beat detection: {len(beats)} beats, "
                                        f"{len(downbeats)} downbeats [{ms:.0f}ms]")

                            beats_data = {"beats": beats, "downbeats": downbeats}
                            beats_path = parent / f"{stem}_beats.json"
                            beats_path.write_text(json.dumps(beats_data, indent=2))

                        # 2) Vocal separation -> queue Opus, ASR on vocals
                        if run_sep:
                            t1 = time.perf_counter()
                            vocals_np, music_np = separate_vocals(
                                roformer_model, str(mp3_path), device)
                            ms_sep = (time.perf_counter() - t1) * 1000
                            logger.info(f"[{track}] Separation [{ms_sep:.0f}ms]")

                            vocal_path = parent / f"{stem}_voc.opus"
                            music_path = parent / f"{stem}_music.opus"

                            def _opus_done(path, ms, error, _item=item, _track=track):
                                remote_name = Path(path).name
                                if error is not None:
                                    logger.error(
                                        f"[{_track}] Opus {remote_name} failed: {error}")
                                    return
                                logger.info(
                                    f"[{_track}] Opus saved {remote_name} [{ms:.0f}ms]")
                                submit_and_log(
                                    pipeline, _item, Path(path), remote_name,
                                    stats, stats_lock)

                            opus_pool.submit(
                                vocals_np, str(vocal_path),
                                bitrate=OPUS_BITRATE_VOCAL, on_done=_opus_done)
                            opus_pool.submit(
                                music_np, str(music_path),
                                bitrate=OPUS_BITRATE_MUSIC, on_done=_opus_done)
                            logger.info(f"[{track}] Opus encode queued (background)")

                            if not overwrite and lyrics_already_done(dav, server_url, item.remote_path):
                                tqdm.write(f"     skip ASR, exists: {lyrics_relpath(item.remote_path)}")
                            else:
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
                                               f"{stem}_lyrics.json", stats, stats_lock)

                        if beats_path is not None:
                            submit_and_log(pipeline, item, beats_path,
                                           f"{stem}_beats.json", stats, stats_lock)

                        elapsed = time.perf_counter() - file_start
                        dur_hours = dur / 3600
                        speed = elapsed / dur_hours if dur_hours > 0 else 0
                        tqdm.write(f"     done in {elapsed:.1f}s "
                                   f"({dur:.0f}s audio, {speed:.0f}s per hour of audio)")

                        processed += 1
                        pbar.update(1)

                    except Exception as e:
                        logger.error(f"[{track}] Processing failed: {e}", exc_info=True)
                        tqdm.write(f"     ERROR: {e} — skipping")
                        pipeline.skip(item)
                        skipped += 1
                        pbar.update(1)

        pbar.close()

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
