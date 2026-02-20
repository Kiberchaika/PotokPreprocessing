#!/usr/bin/env python3
"""
Audio processing pipeline: beat detection + vocal separation/ASR.

Modes:
    beats        – Beat This! → beats & downbeats
    roformer-asr – Windowed Roformer → vocal/music opus 160 kbps → Parakeet ASR
    all          – both (sequentially)

Output per file: _voc.opus, _music.opus, _beats.json, _lyrics.json

Usage:
    python audio_pipeline.py --mode beats
    python audio_pipeline.py --mode roformer-asr --batch-size 4
    python audio_pipeline.py --mode all
"""

import os
import sys
import ctypes

# ── CUDA / env setup (before any torch import) ────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")


import argparse
import fcntl
import glob
import json
import logging
import subprocess
import tempfile
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)
#logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
#logging.getLogger("torch._inductor").setLevel(logging.ERROR)
#logging.getLogger("torch.fx").setLevel(logging.ERROR)
#logging.getLogger("torch._functorch").setLevel(logging.ERROR)

torch.set_float32_matmul_precision("high")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(threadName)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Default configuration ──────────────────────────────────────────────────
BATCH_SIZE = 8
ROFORMER_CHUNK_SIZE = 8          # seconds
ROFORMER_BACKEND = "inductor"    # inductor recommended (TRT fails on complex64)
BEAT_BACKEND = "inductor"        # TRT produces wrong logits → 0 beats
ASR_BACKEND = "inductor"         # inductor recommended for NeMo
USE_FP16 = True
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
ASR_TARGET_SR = 16_000
OPUS_BITRATE_VOCAL = "96k"
OPUS_BITRATE_MUSIC = "128k"

ROFORMER_CKPT_NAME = "mbr-win10-sink8.ckpt"
ROFORMER_CKPT_URL = (
    "https://huggingface.co/smulelabs/windowed-roformer/resolve/main/mbr-win10-sink8.ckpt"
)


# ── Data classes ───────────────────────────────────────────────────────────
@dataclass
class FileResult:
    input_file: str
    beats: List[float] = field(default_factory=list)
    downbeats: List[float] = field(default_factory=list)
    text: str = ""
    segments: List[Dict[str, Any]] = field(default_factory=list)
    words: List[Dict[str, Any]] = field(default_factory=list)
    vocal_path: str = ""
    music_path: str = ""
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
#  Model loaders (called once, reused across files)
# ═══════════════════════════════════════════════════════════════════════════

def _compile(model, backend: str, label: str = "model", dynamic: bool = False):
    """Compile a model with torch.compile; fall back to inductor then raw."""
    logger.info(f"Compiling {label} with backend={backend}, dynamic={dynamic}...")
    try:
        if backend == "tensorrt":
            import torch_tensorrt  # noqa: F401 – registers backend
            return torch.compile(
                model,
                backend="torch_tensorrt",
                options={
                    "enabled_precisions": {torch.float16} if USE_FP16 else {torch.float32},
                    "min_block_size": 1,
                    "truncate_double": True,
                },
            )
        else:
            return torch.compile(
                model,
                mode="default",
                backend="inductor",
                fullgraph=False,
                dynamic=dynamic,
            )
    except Exception as exc:
        logger.warning(f"{label} compile ({backend}) failed: {exc}")
        if backend != "inductor":
            try:
                compiled = torch.compile(
                    model,
                    mode="default",
                    backend="inductor",
                    fullgraph=False,
                    dynamic=dynamic,
                )
                logger.info(f"{label}: inductor fallback OK.")
                return compiled
            except Exception:
                pass
        logger.info(f"{label}: using uncompiled model.")
        return model


# ── Beat This! ─────────────────────────────────────────────────────────────
def load_beat_model(device: str):
    from beat_this.inference import Audio2Beats
    logger.info("Loading Beat This! model...")
    model = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
    model.model.eval()
    model.model = _compile(model.model, BEAT_BACKEND, label="BeatThis")
    return model


# ── Windowed Roformer ──────────────────────────────────────────────────────
def load_roformer_model(device: str):
    _roformer_dir = os.path.join(os.path.dirname(__file__), "windowed-roformer")
    if _roformer_dir not in sys.path:
        sys.path.insert(0, _roformer_dir)
    from model import MelBandRoformerWSA
    # Download checkpoint if needed
    if not os.path.exists(ROFORMER_CKPT_NAME):
        logger.info(f"Downloading Roformer checkpoint {ROFORMER_CKPT_NAME}...")
        torch.hub.download_url_to_file(ROFORMER_CKPT_URL, ROFORMER_CKPT_NAME)

    logger.info("Loading Windowed Roformer model...")
    model = MelBandRoformerWSA()
    ckpt = torch.load(ROFORMER_CKPT_NAME, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device).eval()
    model = _compile(model, ROFORMER_BACKEND, label="Roformer")
    return model


# ── Parakeet ASR ───────────────────────────────────────────────────────────
def load_asr_model(device: str):
    import nemo.collections.asr as nemo_asr
    logger.info(f"Loading ASR model: {ASR_MODEL_NAME}...")
    asr = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
    asr.eval()
    if device == "cuda":
        asr = asr.cuda()
    return asr


def save_json_locked(json_path: str, updates: Dict[str, Any]) -> None:
    """Atomically read-merge-write a JSON file under an exclusive file lock."""
    lock_path = json_path + ".lock"
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            payload = {}
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            payload.update(updates)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    try:
        os.unlink(lock_path)
    except OSError:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Processing functions
# ═══════════════════════════════════════════════════════════════════════════

def detect_beats(beat_model, audio_path: str) -> Tuple[List[float], List[float]]:
    """Run Beat This! on a single audio file. Returns (beats, downbeats) in seconds."""
    from beat_this.preprocessing import load_audio as bt_load_audio
    audio_tensor, sr = bt_load_audio(audio_path)
    with torch.no_grad():
        beats, downbeats = beat_model(audio_tensor, sr)
    return beats.tolist(), downbeats.tolist()


def detect_beats_batch(
    beat_model, audio_paths: List[str],
) -> List[Tuple[List[float], List[float]]]:
    """
    Batched beat detection: pre-compute spectrograms, collect all chunks,
    run model in batches of BATCH_SIZE, reassemble per-file, post-process.
    """
    from beat_this.preprocessing import load_audio as bt_load_audio
    from beat_this.inference import split_piece, aggregate_prediction

    chunk_size = 1500
    border_size = 6
    overlap_mode = "keep_first"
    device = beat_model.device

    # 1. Compute spectrograms
    spects = []
    for path in audio_paths:
        audio, sr = bt_load_audio(path)
        spect = beat_model.signal2spect(audio, sr)
        spects.append(spect)

    # 2. Split all spectrograms into chunks
    all_chunks = []
    file_info = []  # (starts, spect_len, n_chunks)
    for spect in spects:
        chunks, starts = split_piece(spect, chunk_size, border_size=border_size, avoid_short_end=True)
        # Pad short chunks to chunk_size for uniform batching
        padded = []
        for c in chunks:
            if c.shape[0] < chunk_size:
                c = torch.nn.functional.pad(c, (0, 0, 0, chunk_size - c.shape[0]))
            padded.append(c)
        file_info.append((starts, spect.shape[0], len(padded)))
        all_chunks.extend(padded)

    # 3. Batch model inference (pad last mini-batch to avoid recompilation)
    all_chunks_t = torch.stack(all_chunks)
    n_real = len(all_chunks_t)
    remainder = n_real % BATCH_SIZE
    if remainder != 0:
        pad_n = BATCH_SIZE - remainder
        all_chunks_t = torch.cat([all_chunks_t, all_chunks_t[:pad_n]], dim=0)
    all_beat = []
    all_db = []
    with torch.inference_mode():
        with torch.autocast(enabled=beat_model.float16, device_type=device.type):
            for ptr in range(0, len(all_chunks_t), BATCH_SIZE):
                batch = all_chunks_t[ptr : ptr + BATCH_SIZE]
                pred = beat_model.model(batch)
                all_beat.append(pred["beat"].float())
                all_db.append(pred["downbeat"].float())
    all_beat = torch.cat(all_beat, dim=0)[:n_real]
    all_db = torch.cat(all_db, dim=0)[:n_real]

    # 4. Reassemble per-file and post-process
    results = []
    chunk_ptr = 0
    for (starts, spect_len, n_chunks) in file_info:
        pred_chunks = []
        for j in range(n_chunks):
            pred_chunks.append({
                "beat": all_beat[chunk_ptr + j],
                "downbeat": all_db[chunk_ptr + j],
            })
        chunk_ptr += n_chunks
        beat_logits, db_logits = aggregate_prediction(
            pred_chunks, starts, spect_len, chunk_size, border_size,
            overlap_mode, device,
        )
        beats, downbeats = beat_model.frames2beats(beat_logits, db_logits)
        results.append((beats.tolist(), downbeats.tolist()))

    return results


def separate_vocals(
    roformer_model,
    audio_path: str,
    device: str,
    sample_rate: int = 44_100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Windowed Roformer vocal separation.
    Returns (vocal_numpy, music_numpy) at original sample_rate.
    """
    from main import load_audio as rf_load_audio

    mix = rf_load_audio(audio_path, sample_rate)
    mix_t = torch.tensor(mix, dtype=torch.float32)

    chunk_samples = ROFORMER_CHUNK_SIZE * sample_rate
    n_samples = mix_t.shape[1]
    full = int(np.ceil(n_samples / chunk_samples) * chunk_samples)
    if full > n_samples:
        mix_t = torch.nn.functional.pad(mix_t, (0, full - n_samples))

    chunks = mix_t.unfold(1, chunk_samples, chunk_samples).permute(1, 0, 2)
    n_chunks = chunks.shape[0]

    outputs = []
    with torch.cuda.amp.autocast(enabled=(device == "cuda" and USE_FP16)):
        with torch.inference_mode():
            ptr = 0
            while ptr < n_chunks:
                batch = chunks[ptr : ptr + BATCH_SIZE]
                actual = batch.shape[0]
                # Pad last batch to BATCH_SIZE to keep tensor shape constant
                # and avoid torch.compile recompilation
                if actual < BATCH_SIZE:
                    pad = chunks[:BATCH_SIZE - actual]
                    batch = torch.cat([batch, pad], dim=0)
                out = roformer_model(batch.to(device)).cpu()
                if actual < BATCH_SIZE:
                    out = out[:actual]
                outputs.append(out)
                ptr += BATCH_SIZE

    outputs = torch.cat(outputs, dim=0)
    ch = outputs.shape[1]
    vocals = outputs.permute(1, 0, 2).reshape(ch, -1)[:, :n_samples]

    mix_np = np.asarray(mix[:, :n_samples])
    vocals_np = vocals.numpy()
    music_np = mix_np - vocals_np

    return vocals_np, music_np


def save_opus(audio_np: np.ndarray, path: str, sample_rate: int = 44_100,
              bitrate: str = OPUS_BITRATE_VOCAL):
    """Save numpy audio array as Opus via ffmpeg."""
    # Write intermediate wav to a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tensor = torch.from_numpy(audio_np).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        torchaudio.save(tmp.name, tensor, sample_rate)
        cmd = [
            "ffmpeg", "-y", "-i", tmp.name,
            "-c:a", "libopus", "-b:a", bitrate,
            path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        os.unlink(tmp.name)


def transcribe_audio(asr_model, audio_np: np.ndarray, sample_rate: int = 44_100) -> Dict:
    """Resample vocal to 16 kHz mono, transcribe with Parakeet, return result dict."""
    tensor = torch.from_numpy(audio_np).float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    # To mono
    if tensor.shape[0] > 1:
        tensor = tensor.mean(dim=0, keepdim=True)
    # Resample
    if sample_rate != ASR_TARGET_SR:
        tensor = torchaudio.transforms.Resample(sample_rate, ASR_TARGET_SR)(tensor)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        torchaudio.save(tmp.name, tensor, ASR_TARGET_SR)
        output = asr_model.transcribe([tmp.name], batch_size=BATCH_SIZE, timestamps=True)
        hyp = output[0]

        result: Dict[str, Any] = {"text": "", "segments": [], "words": []}
        if hasattr(hyp, "text") and hyp.text:
            result["text"] = hyp.text

        if hasattr(hyp, "timestamp") and hyp.timestamp:
            ts = hyp.timestamp
            if "segment" in ts and ts["segment"]:
                result["segments"] = [
                    {"start": s.get("start", 0), "end": s.get("end", 0), "text": s.get("segment", "")}
                    for s in ts["segment"]
                ]
            if "word" in ts and ts["word"]:
                result["words"] = [
                    {"start": w.get("start", 0), "end": w.get("end", 0), "word": w.get("word", "")}
                    for w in ts["word"]
                ]
        return result
    finally:
        os.unlink(tmp.name)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-file pipeline
# ═══════════════════════════════════════════════════════════════════════════

def process_file(
    audio_path: str,
    output_dir: str,
    input_dir: str,
    mode: str,
    beat_model=None,
    roformer_model=None,
    asr_model=None,
    device: str = "cuda",
) -> FileResult:
    """Process a single audio file. mode: 'beats', 'roformer-asr', or 'all'."""
    rel = os.path.relpath(audio_path, input_dir)
    base = os.path.join(output_dir, os.path.splitext(rel)[0])
    os.makedirs(os.path.dirname(base), exist_ok=True)
    stem = os.path.splitext(rel)[0]

    result = FileResult(input_file=audio_path)
    result.vocal_path = base + "_voc.opus"
    result.music_path = base + "_music.opus"

    run_beats = mode in ("beats", "all")
    run_sep = mode in ("roformer-asr", "all")

    # ── Beat detection ────────────────────────────────────────────────
    if run_beats:
        try:
            t0 = time.perf_counter()
            logger.info(f"[{stem}] Starting beat detection...")
            result.beats, result.downbeats = detect_beats(beat_model, audio_path)
            ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[{stem}] Beat detection done: {len(result.beats)} beats, {len(result.downbeats)} downbeats [{ms:.0f}ms]")
        except Exception as exc:
            logger.error(f"[{stem}] Beat detection failed: {exc}")
            traceback.print_exc()
            result.error = f"beat detection: {exc}"

    # ── Vocal separation + opus + ASR ─────────────────────────────────
    if run_sep:
        try:
            t0 = time.perf_counter()
            logger.info(f"[{stem}] Starting vocal separation...")
            vocals, music = separate_vocals(roformer_model, audio_path, device)
            ms_sep = (time.perf_counter() - t0) * 1000
            logger.info(f"[{stem}] Separation done [{ms_sep:.0f}ms]. Encoding Opus...")

            t1 = time.perf_counter()
            save_opus(vocals, result.vocal_path, bitrate=OPUS_BITRATE_VOCAL)
            save_opus(music, result.music_path, bitrate=OPUS_BITRATE_MUSIC)
            ms_opus = (time.perf_counter() - t1) * 1000
            logger.info(f"[{stem}] Opus files saved [{ms_opus:.0f}ms].")

            t2 = time.perf_counter()
            logger.info(f"[{stem}] Starting ASR on vocals...")
            asr_out = transcribe_audio(asr_model, vocals)
            ms_asr = (time.perf_counter() - t2) * 1000
            ms_total = (time.perf_counter() - t0) * 1000
            logger.info(f"[{stem}] ASR done: {len(asr_out['text'])} chars [{ms_asr:.0f}ms] | total [{ms_total:.0f}ms]")

            result.text = asr_out["text"]
            result.segments = asr_out["segments"]
            result.words = asr_out["words"]
        except Exception as exc:
            logger.error(f"[{stem}] Separation/ASR failed: {exc}")
            traceback.print_exc()
            result.error = (result.error or "") + f" separation/asr: {exc}"

    # ── Save JSON files ─────────────────────────────────────────────
    if run_beats:
        beats_path = base + "_beats.json"
        save_json_locked(beats_path, {
            "beats": result.beats,
            "downbeats": result.downbeats,
        })
        logger.info(f"[{stem}] Beats saved → {beats_path}")
    if run_sep:
        lyrics_path = base + "_lyrics.json"
        save_json_locked(lyrics_path, {
            "text": result.text,
            "segments": result.segments,
            "words": result.words,
        })
        logger.info(f"[{stem}] Lyrics saved → {lyrics_path}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Audio pipeline: beats + separation + ASR")
    p.add_argument("--mode", choices=["beats", "roformer-asr", "all"], default="all",
                   help="Pipeline mode: beats, roformer-asr, or all (default: all)")
    p.add_argument("--input_dir", default="./input", help="Folder with *.mp3 files")
    p.add_argument("--output", default="./output", help="Output folder (default: <input_dir>/output)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Roformer chunk batch size")
    p.add_argument("--backend", default=None, help="Override all compile backends")
    p.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    p.add_argument("--bench", action="store_true",
                   help="Benchmark: test every mode with batch sizes 1–10, save results to JSON")
    return p.parse_args()


def run_pipeline(
    mode: str,
    mp3_files: List[str],
    input_dir: str,
    output_dir: str,
    device: str,
    beat_model=None,
    roformer_model=None,
    asr_model=None,
    total_audio_hours: float = 0.0,
) -> Tuple[int, float, float]:
    """
    Run the processing pipeline.
    Returns (ok_count, total_time_sec, avg_time_per_hour_audio).
    """
    total_audio_sec = total_audio_hours * 3600

    # ── Warmup (one pass to trigger JIT) ───────────────────────────────
    logger.info("Warmup pass...")
    try:
        if mode in ("beats",):
            detect_beats_batch(beat_model, mp3_files[:min(BATCH_SIZE, len(mp3_files))])
        elif mode == "roformer-asr":
            _ = process_file(mp3_files[0], output_dir, input_dir, mode,
                             None, roformer_model, asr_model, device)
        else:
            _ = process_file(mp3_files[0], output_dir, input_dir, mode,
                             beat_model, roformer_model, asr_model, device)
        if device == "cuda":
            torch.cuda.synchronize()
        logger.info("Warmup complete.")
    except Exception as exc:
        logger.warning(f"Warmup had issues (continuing): {exc}")

    # ── Process all files ─────────────────────────────────────────────
    results: List[FileResult] = []
    total_start = time.perf_counter()

    if mode == "beats":
        for batch_start in range(0, len(mp3_files), BATCH_SIZE):
            batch_paths = mp3_files[batch_start : batch_start + BATCH_SIZE]
            batch_end = batch_start + len(batch_paths)
            logger.info(f"\n{'═'*60}")
            logger.info(f"Batch [{batch_start+1}–{batch_end}/{len(mp3_files)}] ({len(batch_paths)} files)")
            logger.info(f"{'═'*60}")

            file_start = time.perf_counter()
            try:
                batch_results = detect_beats_batch(beat_model, batch_paths)
                ms = (time.perf_counter() - file_start) * 1000
                for path, (beats, downbeats) in zip(batch_paths, batch_results):
                    rel = os.path.relpath(path, input_dir)
                    base = os.path.join(output_dir, os.path.splitext(rel)[0])
                    os.makedirs(os.path.dirname(base), exist_ok=True)

                    r = FileResult(input_file=path, beats=beats, downbeats=downbeats)
                    results.append(r)

                    beats_path = base + "_beats.json"
                    save_json_locked(beats_path, {
                        "beats": beats,
                        "downbeats": downbeats,
                    })

                    logger.info(f"  [{rel}] {len(beats)} beats, {len(downbeats)} downbeats")
                logger.info(f"Batch done [{ms:.0f}ms]")
            except Exception as exc:
                logger.error(f"Batch failed: {exc}")
                traceback.print_exc()
                for path in batch_paths:
                    results.append(FileResult(input_file=path, error=str(exc)))

    elif mode == "roformer-asr":
        for batch_start in range(0, len(mp3_files), BATCH_SIZE):
            batch_paths = mp3_files[batch_start : batch_start + BATCH_SIZE]
            batch_end = batch_start + len(batch_paths)
            logger.info(f"\n{'═'*60}")
            logger.info(f"Batch [{batch_start+1}–{batch_end}/{len(mp3_files)}] ({len(batch_paths)} files)")
            logger.info(f"{'═'*60}")

            batch_t0 = time.perf_counter()

            sep_results = []
            for path in batch_paths:
                rel = os.path.relpath(path, input_dir)
                stem = os.path.splitext(rel)[0]
                base = os.path.join(output_dir, stem)
                os.makedirs(os.path.dirname(base), exist_ok=True)
                try:
                    t0 = time.perf_counter()
                    vocals, music = separate_vocals(roformer_model, path, device)
                    ms_sep = (time.perf_counter() - t0) * 1000
                    logger.info(f"  [{stem}] Separation [{ms_sep:.0f}ms]")

                    t1 = time.perf_counter()
                    vocal_path = base + "_voc.opus"
                    music_path = base + "_music.opus"
                    save_opus(vocals, vocal_path, bitrate=OPUS_BITRATE_VOCAL)
                    save_opus(music, music_path, bitrate=OPUS_BITRATE_MUSIC)
                    ms_opus = (time.perf_counter() - t1) * 1000
                    logger.info(f"  [{stem}] Opus saved [{ms_opus:.0f}ms]")

                    sep_results.append((path, rel, base, vocals, vocal_path, music_path))
                except Exception as exc:
                    logger.error(f"  [{stem}] Separation failed: {exc}")
                    traceback.print_exc()
                    results.append(FileResult(input_file=path, error=f"separation: {exc}"))
                    sep_results.append(None)

            ms_phase1 = (time.perf_counter() - batch_t0) * 1000
            logger.info(f"  Phase 1 (separation+opus) done [{ms_phase1:.0f}ms]")

            t_asr0 = time.perf_counter()
            for item in sep_results:
                if item is None:
                    continue
                path, rel, base, vocals, vocal_path, music_path = item
                stem = os.path.splitext(rel)[0]
                try:
                    t0 = time.perf_counter()
                    asr_out = transcribe_audio(asr_model, vocals)
                    ms_asr = (time.perf_counter() - t0) * 1000
                    logger.info(f"  [{stem}] ASR: {len(asr_out['text'])} chars [{ms_asr:.0f}ms]")

                    r = FileResult(input_file=path, vocal_path=vocal_path, music_path=music_path,
                                   text=asr_out["text"], segments=asr_out["segments"],
                                   words=asr_out["words"])
                    results.append(r)

                    lyrics_path = base + "_lyrics.json"
                    save_json_locked(lyrics_path, {
                        "text": asr_out["text"],
                        "segments": asr_out["segments"],
                        "words": asr_out["words"],
                    })
                except Exception as exc:
                    logger.error(f"  [{stem}] ASR failed: {exc}")
                    traceback.print_exc()
                    results.append(FileResult(input_file=path, error=f"asr: {exc}"))

            ms_phase2 = (time.perf_counter() - t_asr0) * 1000
            ms_batch = (time.perf_counter() - batch_t0) * 1000
            logger.info(f"  Phase 2 (ASR) done [{ms_phase2:.0f}ms]")
            logger.info(f"Batch done [{ms_batch:.0f}ms]")

    else:
        for idx, mp3 in enumerate(mp3_files, 1):
            logger.info(f"\n{'═'*60}")
            logger.info(f"Processing [{idx}/{len(mp3_files)}]: {os.path.relpath(mp3, input_dir)}")
            logger.info(f"{'═'*60}")

            file_start = time.perf_counter()
            try:
                r = process_file(mp3, output_dir, input_dir, mode,
                                 beat_model, roformer_model, asr_model, device)
                results.append(r)
            except Exception as exc:
                logger.error(f"Failed on {mp3}: {exc}")
                traceback.print_exc()
                results.append(FileResult(input_file=mp3, error=str(exc)))

            elapsed = time.perf_counter() - file_start
            logger.info(f"[{os.path.relpath(mp3, input_dir)}] finished in {elapsed:.1f}s")

    total_time = time.perf_counter() - total_start

    # ── Summary ────────────────────────────────────────────────────────
    ok = sum(1 for r in results if r.error is None)
    avg_per_hour = total_time / total_audio_hours if total_audio_hours > 0 else 0.0

    logger.info(f"\n{'═'*60}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'═'*60}")
    logger.info(f"Files processed: {ok}/{len(mp3_files)}")
    logger.info(f"Total audio:     {total_audio_sec:.1f}s ({total_audio_hours:.2f}h)")
    logger.info(f"Total time:      {total_time:.1f}s")
    if mp3_files:
        logger.info(f"Avg per file:    {total_time/len(mp3_files):.1f}s")
    if total_audio_hours > 0:
        logger.info(f"Avg time per hour audio: {avg_per_hour:.1f}s")
    logger.info(f"Output:          {output_dir}")
    logger.info(f"{'═'*60}")

    return ok, total_time, avg_per_hour


def get_gpu_name() -> str:
    """Return short GPU name, e.g. 'NVIDIA_RTX_4090'."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return name.replace(" ", "_")
    return "CPU"


def main():
    args = parse_args()

    global BATCH_SIZE, USE_FP16, ROFORMER_BACKEND, BEAT_BACKEND, ASR_BACKEND

    if args.no_fp16:
        USE_FP16 = False
    if args.backend:
        ROFORMER_BACKEND = args.backend
        BEAT_BACKEND = args.backend
        ASR_BACKEND = args.backend

    input_dir = os.path.abspath(args.input_dir)
    output_dir = args.output or os.path.join(input_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    mp3_files = sorted(glob.glob(os.path.join(input_dir, "**/*.mp3"), recursive=True))
    if not mp3_files:
        logger.error(f"No *.mp3 files found in {input_dir} (recursive)")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logger.warning("CUDA not available – running unoptimized on CPU.")

    # ── Compute total audio duration ─────────────────────────────────
    import librosa
    total_audio_sec = 0.0
    for mp3 in mp3_files:
        try:
            total_audio_sec += librosa.get_duration(path=mp3)
        except Exception:
            pass
    total_audio_hours = total_audio_sec / 3600

    # ── Bench mode ────────────────────────────────────────────────────
    if args.bench:
        gpu_name = get_gpu_name()
        bench_file = f"bench_preprocessing_on_{gpu_name}.json"

        # Load existing results
        if os.path.exists(bench_file):
            with open(bench_file, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
        else:
            bench_data = {"results": []}

        # Load ALL models once
        logger.info("Bench: loading all models...")
        t0 = time.perf_counter()
        beat_model = load_beat_model(device)
        roformer_model = load_roformer_model(device)
        asr_model = load_asr_model(device)
        logger.info(f"All models loaded in {time.perf_counter() - t0:.1f}s")

        modes = ["beats", "roformer-asr", "all"]

        for bs in range(1, 11):
            for mode in modes:
                BATCH_SIZE = bs
                logger.info(f"\n{'█'*60}")
                logger.info(f"BENCH: mode={mode}, batch_size={bs}")
                logger.info(f"{'█'*60}")

                ok, total_time, avg_per_hour = run_pipeline(
                    mode=mode,
                    mp3_files=mp3_files,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    device=device,
                    beat_model=beat_model,
                    roformer_model=roformer_model,
                    asr_model=asr_model,
                    total_audio_hours=total_audio_hours,
                )

                bench_data["results"].append({
                    "gpu": gpu_name,
                    "mode": mode,
                    "batch_size": bs,
                    "avg_time_per_hour": round(avg_per_hour, 2),
                    "total_time": round(total_time, 2),
                    "files": len(mp3_files),
                    "ok": ok,
                })

                # Save after every run (crash-safe)
                with open(bench_file, "w", encoding="utf-8") as f:
                    json.dump(bench_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Bench result saved → {bench_file}")

        logger.info(f"\n{'█'*60}")
        logger.info(f"BENCHMARK COMPLETE: {len(bench_data['results'])} results in {bench_file}")
        logger.info(f"{'█'*60}")
        return 0

    # ── Normal mode ───────────────────────────────────────────────────
    mode = args.mode
    BATCH_SIZE = args.batch_size

    logger.info(f"Found {len(mp3_files)} MP3 file(s) in {input_dir}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: batch_size={BATCH_SIZE}, fp16={USE_FP16}")
    logger.info(f"Device: {device}")

    # ── Load only the models needed for this mode ─────────────────────
    t0 = time.perf_counter()
    beat_model = roformer_model = asr_model = None
    if mode in ("beats", "all"):
        beat_model = load_beat_model(device)
    if mode in ("roformer-asr", "all"):
        roformer_model = load_roformer_model(device)
        asr_model = load_asr_model(device)
    logger.info(f"Models loaded in {time.perf_counter() - t0:.1f}s")

    ok, total_time, avg_per_hour = run_pipeline(
        mode=mode,
        mp3_files=mp3_files,
        input_dir=input_dir,
        output_dir=output_dir,
        device=device,
        beat_model=beat_model,
        roformer_model=roformer_model,
        asr_model=asr_model,
        total_audio_hours=total_audio_hours,
    )

    return 0 if ok == len(mp3_files) else 1


if __name__ == "__main__":
    sys.exit(main())
