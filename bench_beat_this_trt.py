#!/usr/bin/env python3
"""
Beat This! benchmark with TensorRT optimization via torch.compile.
Use: source /home/k4/Projects/PeoplePoseEstimationTestsAug2025/.env/bin/activate
"""

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
import warnings
import torch
import numpy as np

# Suppress excessive warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)

# Import Beat This! components
from beat_this.inference import Audio2Beats
from beat_this.preprocessing import load_audio

# Enable TensorFloat32 for better performance
torch.set_float32_matmul_precision('high')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimization configuration
BATCH_SIZE = 1  # Configurable batch size
USE_FP16 = True  # Use FP16 for better performance
# Backend options: "tensorrt", "inductor", "cudagraphs"
BACKEND = "tensorrt"  # Use TensorRT via torch.compile


class Audio2BeatsTRT(Audio2Beats):
    """
    Extended Audio2Beats with torch.compile optimization.
    """

    def __init__(self, checkpoint_path="final0", device="cuda", dbn=False, batch_size=BATCH_SIZE):
        super().__init__(checkpoint_path=checkpoint_path, device=device, dbn=dbn)
        self.batch_size = batch_size
        self._compiled = False
        self._original_model = None

    def compile_model(self):
        """
        Compile the internal model with torch.compile using specified backend.
        """
        if self._compiled:
            logger.info("Model already compiled.")
            return

        # Store original model reference
        self._original_model = self.model
        self._original_model.eval()

        logger.info(f"Compiling model with torch.compile (backend={BACKEND})...")

        try:
            if BACKEND == "tensorrt":
                # Import torch_tensorrt to register the backend
                import torch_tensorrt
                self.model = torch.compile(
                    self._original_model,
                    backend="torch_tensorrt",
                    options={
                        "enabled_precisions": {torch.float16} if USE_FP16 else {torch.float32},
                        "min_block_size": 1,
                        "truncate_double": True,
                    }
                )
            elif BACKEND == "inductor":
                self.model = torch.compile(
                    self._original_model,
                    mode="max-autotune-no-cudagraphs",
                    backend="inductor",
                    fullgraph=False,
                )
            else:  # cudagraphs or default
                self.model = torch.compile(
                    self._original_model,
                    mode="default",
                    backend="inductor",
                )

            self._compiled = True
            logger.info(f"torch.compile ({BACKEND}) successful.")

        except Exception as e:
            logger.warning(f"Compilation with {BACKEND} failed: {e}")
            logger.info("Falling back to inductor...")
            try:
                self.model = torch.compile(
                    self._original_model,
                    mode="max-autotune-no-cudagraphs",
                    backend="inductor",
                    fullgraph=False,
                )
                self._compiled = True
                logger.info("torch.compile (inductor fallback) successful.")
            except Exception as e2:
                logger.warning(f"Inductor fallback also failed: {e2}")
                logger.info("Using original model without optimization.")
                self.model = self._original_model


def main():
    # Adjust paths as necessary
    input_file = "input.mp3"

    # Beat This! parameters
    checkpoint = "final0"  # Default model
    use_dbn = False        # Beat This! aims to work well without DBN
    batch_size = BATCH_SIZE

    # Check input
    if not os.path.exists(input_file):
        logger.error(f"File {input_file} not found.")
        return

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device used: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"FP16 mode: {USE_FP16}")
    logger.info(f"Backend: {BACKEND}")

    if device != 'cuda':
        logger.warning("CUDA not available. Running without optimization.")

    # Initialize Model with TensorRT support
    logger.info("Initializing Beat This! model...")
    try:
        model = Audio2BeatsTRT(
            checkpoint_path=checkpoint,
            device=device,
            dbn=use_dbn,
            batch_size=batch_size
        )
    except Exception as e:
        logger.error(f"Error loading Beat This!: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load Audio (I/O excluded from benchmark)
    logger.info("Loading audio into memory...")
    try:
        audio_tensor, sample_rate = load_audio(input_file)
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        return

    # Compile model
    if device == 'cuda':
        model.compile_model()

    # Warmup runs to trigger JIT compilation
    logger.info("Warmup runs (triggers compilation)...")
    try:
        with torch.no_grad():
            for warmup_i in range(3):
                beats, downbeats = model(audio_tensor, sample_rate)
                if device == 'cuda':
                    torch.cuda.synchronize()
                logger.info(f"Warmup {warmup_i + 1}/3 complete.")

        logger.info(f"Warmup complete. Found {len(beats)} beats and {len(downbeats)} downbeats.")
    except Exception as e:
        logger.error(f"Error during warmup: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("Starting benchmark (10 runs)...")
    execution_times = []

    for i in range(10):
        logger.info(f"--- Run {i + 1}/10 ---")

        # Synchronize CUDA before timing
        if device == 'cuda':
            torch.cuda.synchronize()

        # Start timer
        start_time = time.perf_counter()

        try:
            with torch.no_grad():
                # Run inference
                beats, downbeats = model(audio_tensor, sample_rate)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return

        # Synchronize CUDA after inference
        if device == 'cuda':
            torch.cuda.synchronize()

        # End timer
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

        logger.info(f"\n=== Benchmark Results (Beat This! + {BACKEND}) ===")
        logger.info(f"Batch size:   {batch_size}")
        logger.info(f"Backend:      {BACKEND}")
        logger.info(f"FP16:         {USE_FP16}")
        logger.info(f"Average time: {avg_time:.2f} ms")
        logger.info(f"Std dev:      {std_time:.2f} ms")
        logger.info(f"Minimum:      {min_time:.2f} ms")
        logger.info(f"Maximum:      {max_time:.2f} ms")
        logger.info("=" * 54)


if __name__ == "__main__":
    main()
