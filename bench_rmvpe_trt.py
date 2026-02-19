#!/usr/bin/env python3
"""
RMVPE benchmark with TensorRT optimization via torch.compile.
Use: source /home/k4/Projects/PeoplePoseEstimationTestsAug2025/.env/bin/activate
"""

import sys
import os
import ctypes

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set up CUDA 13 library path for TensorRT BEFORE importing torch
_cuda13_lib = "/home/k4/Projects/PeoplePoseEstimationTestsAug2025/.env/lib/python3.10/site-packages/nvidia/cu13/lib"
if os.path.exists(_cuda13_lib):
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _cuda13_lib not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{_cuda13_lib}:{current_ld_path}"
    try:
        ctypes.CDLL(os.path.join(_cuda13_lib, "libcudart.so.13"), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

import time
import logging
import warnings
import types
import numpy as np
import librosa
import torch

# Suppress excessive warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)

# Import RMVPE without triggering __init__.py (which pulls in dataset dependencies)
_rmvpe_root = "/home/k4/Projects/BirdsMilkDatasetPreprocessing/RMVPE"
sys.path.insert(0, _rmvpe_root)

src_pkg = types.ModuleType("src")
src_pkg.__path__ = [os.path.join(_rmvpe_root, "src")]
src_pkg.__package__ = "src"
sys.modules["src"] = src_pkg

from src.inference import RMVPE

# Enable TensorFloat32 for better performance
torch.set_float32_matmul_precision('high')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimization configuration
USE_FP16 = True
BACKEND = "tensorrt"


def compile_model(detector):
    """Compile the E2E0 backbone with torch.compile using TensorRT."""
    original_model = detector.model
    original_model.eval()

    logger.info(f"Compiling E2E0 model with torch.compile (backend={BACKEND})...")

    try:
        if BACKEND == "tensorrt":
            import torch_tensorrt
            detector.model = torch.compile(
                original_model,
                backend="torch_tensorrt",
                options={
                    "enabled_precisions": {torch.float16} if USE_FP16 else {torch.float32},
                    "min_block_size": 1,
                    "truncate_double": True,
                }
            )
        elif BACKEND == "inductor":
            detector.model = torch.compile(
                original_model,
                mode="max-autotune-no-cudagraphs",
                backend="inductor",
                fullgraph=False,
            )
        else:
            detector.model = torch.compile(
                original_model,
                mode="default",
                backend="inductor",
            )
        logger.info(f"torch.compile ({BACKEND}) successful.")

    except Exception as e:
        logger.warning(f"Compilation with {BACKEND} failed: {e}")
        logger.info("Falling back to inductor...")
        try:
            detector.model = torch.compile(
                original_model,
                mode="max-autotune-no-cudagraphs",
                backend="inductor",
                fullgraph=False,
            )
            logger.info("torch.compile (inductor fallback) successful.")
        except Exception as e2:
            logger.warning(f"Inductor fallback also failed: {e2}")
            logger.info("Using original model without optimization.")
            detector.model = original_model


def main():
    input_file = "input.mp3"
    model_path = "/home/k4/Projects/BirdsMilkDatasetPreprocessing/ckpts/rmvpe.pt"

    if not os.path.exists(input_file):
        logger.error(f"File {input_file} not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"FP16 mode: {USE_FP16}")
    logger.info(f"Backend: {BACKEND}")

    if device != "cuda":
        logger.warning("CUDA not available. Running without optimization.")

    # Initialize Model
    logger.info("Initializing RMVPE model...")
    try:
        detector = RMVPE(model_path, hop_length=160)
    except Exception as e:
        logger.error(f"Error loading RMVPE: {e}")
        return

    # Move model and mel extractor to device before compiling
    detector.model = detector.model.to(device)
    detector.mel_extractor = detector.mel_extractor.to(device)

    # Load and Preprocess Audio (I/O excluded from benchmark)
    logger.info("Loading audio into memory (resampling to 16000Hz)...")
    try:
        audio, sr = librosa.load(input_file, sr=16000)
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        return

    # Compile model
    if device == "cuda":
        compile_model(detector)

    # Warmup runs to trigger JIT compilation
    logger.info("Warmup runs (triggers compilation)...")
    try:
        for warmup_i in range(3):
            f0 = detector.infer_from_audio(audio, sample_rate=sr, device=device, thred=0.03)
            if device == "cuda":
                torch.cuda.synchronize()
            logger.info(f"Warmup {warmup_i + 1}/3 complete.")
        voiced = np.sum(f0 > 0)
        logger.info(f"Warmup done. {len(f0)} frames, {voiced} voiced.")
    except Exception as e:
        logger.error(f"Error during warmup: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("Starting benchmark (10 runs)...")
    execution_times = []

    for i in range(10):
        logger.info(f"--- Run {i + 1}/10 ---")

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        try:
            f0 = detector.infer_from_audio(audio, sample_rate=sr, device=device, thred=0.03)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        execution_times.append(elapsed_ms)

        logger.info(f"Run {i + 1} time: {elapsed_ms:.2f} ms")

    # Statistics
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        std_time = np.std(execution_times)

        logger.info(f"\n=== Benchmark Results (RMVPE + {BACKEND}) ===")
        logger.info(f"Backend:      {BACKEND}")
        logger.info(f"FP16:         {USE_FP16}")
        logger.info(f"Average time: {avg_time:.2f} ms")
        logger.info(f"Std dev:      {std_time:.2f} ms")
        logger.info(f"Minimum:      {min_time:.2f} ms")
        logger.info(f"Maximum:      {max_time:.2f} ms")
        logger.info("=" * 48)


if __name__ == "__main__":
    main()
