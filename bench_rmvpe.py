import sys
import os
import time
import logging
import numpy as np
import librosa
import torch

import importlib

# Import RMVPE without triggering __init__.py (which pulls in dataset dependencies)
_rmvpe_root = "/home/k4/Projects/BirdsMilkDatasetPreprocessing/RMVPE"
sys.path.insert(0, _rmvpe_root)

# Stub out the package __init__ to avoid importing dataset/loss modules
import types
src_pkg = types.ModuleType("src")
src_pkg.__path__ = [os.path.join(_rmvpe_root, "src")]
src_pkg.__package__ = "src"
sys.modules["src"] = src_pkg

from src.inference import RMVPE

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    input_file = "input.mp3"
    model_path = "/home/k4/Projects/BirdsMilkDatasetPreprocessing/ckpts/rmvpe.pt"

    if not os.path.exists(input_file):
        logger.error(f"File {input_file} not found.")
        return

    # Initialize Model
    logger.info("Initializing RMVPE model...")
    try:
        detector = RMVPE(model_path, hop_length=160)
    except Exception as e:
        logger.error(f"Error loading RMVPE: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load and Preprocess Audio (I/O excluded from benchmark)
    # RMVPE internally works at 16kHz
    logger.info("Loading audio into memory (resampling to 16000Hz)...")
    try:
        audio, sr = librosa.load(input_file, sr=16000)
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        return

    logger.info("Starting benchmark (10 runs)...")
    execution_times = []

    for i in range(10):
        logger.info(f"--- Run {i + 1}/10 ---")

        start_time = time.perf_counter()

        try:
            f0 = detector.infer_from_audio(audio, sample_rate=sr, device=device, thred=0.03)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return

        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        execution_times.append(elapsed_ms)

        logger.info(f"Run {i + 1} time: {elapsed_ms:.2f} ms")

        if i == 0:
            voiced = np.sum(f0 > 0)
            logger.info(f"Inference successful. {len(f0)} frames, {voiced} voiced.")

    # Statistics
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)

        logger.info("\n=== Benchmark Results (RMVPE) ===")
        logger.info(f"Average time: {avg_time:.2f} ms")
        logger.info(f"Minimum:      {min_time:.2f} ms")
        logger.info(f"Maximum:      {max_time:.2f} ms")
        logger.info("=================================")

if __name__ == "__main__":
    main()
