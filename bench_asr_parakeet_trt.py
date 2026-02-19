#!/usr/bin/env python3
"""
Benchmark script for NVIDIA Parakeet TDT ASR model with TensorRT optimization.
Runs inference and reports timing statistics.

Supported languages: bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk
"""
# pip install -U nemo_toolkit['asr']

import os
import sys
import ctypes

# Set CUDA to only use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set up CUDA 13 library path for TensorRT BEFORE importing torch
_cuda13_lib = "/home/k4/Projects/PeoplePoseEstimationTestsAug2025/.env/lib/python3.10/site-packages/nvidia/cu13/lib"
if os.path.exists(_cuda13_lib):
    # Add to LD_LIBRARY_PATH
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _cuda13_lib not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{_cuda13_lib}:{current_ld_path}"
    # Preload the CUDA runtime library
    try:
        ctypes.CDLL(os.path.join(_cuda13_lib, "libcudart.so.13"), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

import time
import logging
import tempfile
import traceback
import warnings

import torch
import torchaudio
import numpy as np

# Suppress excessive warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)

# Enable TensorFloat32 for better performance
torch.set_float32_matmul_precision('high')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

INPUT_FILE = "/home/k4/Python/windowed-roformer/input_vocal.wav"
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"  # v3 supports multilingual
NUM_RUNS = 20
TARGET_SR = 16000

# Optimization configuration
BATCH_SIZE = 1  # Configurable batch size
USE_FP16 = True  # Use FP16 for better performance
# Backend options: "tensorrt", "inductor", "cudagraphs"
# Note: NeMo models may have complex ops, inductor is more compatible
BACKEND = "inductor"  # Use inductor (TensorRT has memory/op issues with NeMo models)


def prepare_audio(input_path: str) -> str:
    """Convert audio to mono 16kHz WAV for Parakeet model."""
    waveform, sr = torchaudio.load(input_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        waveform = resampler(waveform)

    # Save to temp file
    tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    torchaudio.save(tmp_file.name, waveform, TARGET_SR)
    logger.info(f"Prepared audio: {waveform.shape[1]/TARGET_SR:.1f}s @ {TARGET_SR}Hz mono")
    return tmp_file.name


def compile_model(model, device='cuda'):
    """Compile model with torch.compile using specified backend."""
    logger.info(f"Compiling model with torch.compile (backend={BACKEND})...")

    try:
        if BACKEND == "tensorrt":
            # Import torch_tensorrt to register the backend
            import torch_tensorrt
            compiled_model = torch.compile(
                model,
                backend="torch_tensorrt",
                options={
                    "enabled_precisions": {torch.float16} if USE_FP16 else {torch.float32},
                    "min_block_size": 1,
                    "truncate_double": True,
                }
            )
        elif BACKEND == "inductor":
            compiled_model = torch.compile(
                model,
                mode="max-autotune-no-cudagraphs",
                backend="inductor",
                fullgraph=False,
            )
        else:  # cudagraphs or default
            compiled_model = torch.compile(
                model,
                mode="default",
                backend="inductor",
            )

        logger.info(f"torch.compile ({BACKEND}) successful.")
        return compiled_model

    except Exception as e:
        logger.warning(f"Compilation with {BACKEND} failed: {e}")
        logger.info("Falling back to inductor...")
        try:
            compiled_model = torch.compile(
                model,
                mode="max-autotune-no-cudagraphs",
                backend="inductor",
                fullgraph=False,
            )
            logger.info("torch.compile (inductor fallback) successful.")
            return compiled_model
        except Exception as e2:
            logger.warning(f"Inductor fallback also failed: {e2}")
            logger.info("Using original model without optimization.")
            return model


def main():
    # Check input file
    if not os.path.exists(INPUT_FILE):
        logger.error(f"File not found: {INPUT_FILE}")
        return 1

    logger.info(f"Input file: {INPUT_FILE}")
    logger.info(f"Backend: {BACKEND}")
    logger.info(f"FP16 mode: {USE_FP16}")

    # Prepare audio (convert to mono 16kHz)
    logger.info("Preparing audio...")
    prepared_file = prepare_audio(INPUT_FILE)

    try:
        import nemo.collections.asr as nemo_asr

        logger.info(f"Loading model: {MODEL_NAME}...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        asr_model.eval()

        # Move to GPU
        if torch.cuda.is_available():
            asr_model = asr_model.cuda()

            # Compile encoder with torch.compile for optimization
            if hasattr(asr_model, 'encoder'):
                logger.info("Compiling encoder module...")
                asr_model.encoder = compile_model(asr_model.encoder)

        # Warmup run
        logger.info("Warmup run...")
        try:
            _ = asr_model.transcribe([prepared_file], timestamps=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("Warmup complete.")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

        logger.info(f"Starting benchmark ({NUM_RUNS} runs)...")
        execution_times = []
        final_output = None

        for i in range(NUM_RUNS):
            logger.info(f"--- Run {i + 1}/{NUM_RUNS} ---")

            # Synchronize CUDA before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            try:
                # Transcribe with timestamps
                output = asr_model.transcribe([prepared_file], timestamps=True)
                final_output = output[0]

                # Get text from hypothesis
                if hasattr(final_output, 'text') and final_output.text:
                    text_preview = final_output.text[:100]
                    logger.info(f"Transcription: {text_preview}...")
            except Exception as e:
                logger.error(f"Inference error: {e}")
                traceback.print_exc()
                continue

            # Synchronize CUDA after inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000
            execution_times.append(elapsed_ms)

            logger.info(f"Run {i + 1} time: {elapsed_ms:.2f} ms")

        # Print transcription result
        if final_output:
            logger.info("\n" + "=" * 60)
            logger.info("TRANSCRIPTION RESULT")
            logger.info("=" * 60)

            text = getattr(final_output, 'text', '') or ''
            logger.info(f"\nFull text:\n{text}\n")

            # Print timestamps
            if hasattr(final_output, 'timestamp') and final_output.timestamp:
                timestamps = final_output.timestamp

                # Segment timestamps
                if 'segment' in timestamps and timestamps['segment']:
                    logger.info("Segment timestamps:")
                    logger.info("-" * 40)
                    for seg in timestamps['segment']:
                        start = seg.get('start', 0)
                        end = seg.get('end', 0)
                        text = seg.get('segment', '')
                        logger.info(f"[{start:.2f}s - {end:.2f}s] {text}")

                # Word timestamps
                if 'word' in timestamps and timestamps['word']:
                    logger.info("\nWord timestamps:")
                    logger.info("-" * 40)
                    for word in timestamps['word']:
                        start = word.get('start', 0)
                        end = word.get('end', 0)
                        text = word.get('word', '')
                        logger.info(f"[{start:.2f}s - {end:.2f}s] {text}")

            logger.info("=" * 60)

        # Benchmark statistics
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            std_time = np.std(execution_times)

            logger.info(f"\n=== Benchmark Results (Parakeet TDT + {BACKEND}) ===")
            logger.info(f"Model:        {MODEL_NAME}")
            logger.info(f"Backend:      {BACKEND}")
            logger.info(f"FP16:         {USE_FP16}")
            logger.info(f"Average time: {avg_time:.2f} ms")
            logger.info(f"Std dev:      {std_time:.2f} ms")
            logger.info(f"Min time:     {min_time:.2f} ms")
            logger.info(f"Max time:     {max_time:.2f} ms")
            logger.info("=" * 52)

    finally:
        # Cleanup temp file
        if os.path.exists(prepared_file):
            os.remove(prepared_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
