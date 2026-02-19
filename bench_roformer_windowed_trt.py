#!/usr/bin/env python3
"""
Windowed Roformer benchmark with TensorRT optimization via torch.compile.
Use: source /home/k4/Projects/BirdsMilkDatasetPreprocessing/.venv/bin/activate
"""

import os
import sys

# Set CUDA to only use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Add windowed-roformer to path for model/main imports
sys.path.insert(0, "./windowed-roformer")

import time
import logging
import warnings
import torch
import numpy as np

# Suppress excessive warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)

# Import Windowed Roformer components
from main import load_audio, save_audio
from model import MelBandRoformerWSA

# Enable TensorFloat32 for better performance
torch.set_float32_matmul_precision('high')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimization configuration
BATCH_SIZE = 4  # Configurable batch size for chunk processing
USE_FP16 = True  # Use FP16 for better performance
# Backend options: "tensorrt", "inductor", "cudagraphs"
# Note: TensorRT doesn't support complex64 (used in FFT), so inductor is recommended
BACKEND = "inductor"  # Use inductor (TensorRT fails on complex64 tensors)


def download_ckpt_if_missing(path, url):
    """Download checkpoint if missing."""
    if not os.path.exists(path):
        logger.info(f"Checkpoint not found. Downloading {path}...")
        try:
            torch.hub.download_url_to_file(url, path)
        except Exception as e:
            logger.error(f"Failed to download checkpoint: {e}")
            raise


def load_model_trt(checkpoint_path: str, device: str = 'cuda') -> MelBandRoformerWSA:
    """Load model and optionally compile with TensorRT."""
    logger.info(f"Loading model from {checkpoint_path}...")

    model = MelBandRoformerWSA()

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint, strict=True)

    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully!")
    return model


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


def demix_trt(
    model: MelBandRoformerWSA,
    mix: torch.Tensor,
    device: str = 'cuda',
    sample_rate: int = 44100,
    chunk_size: int = 8,
    batch_size: int = BATCH_SIZE,
) -> torch.Tensor:
    """Demix audio using TensorRT-optimized model."""
    mix = torch.tensor(mix, dtype=torch.float32)
    chunk_size_samples = chunk_size * sample_rate
    audio_samples = mix.shape[1]
    full_samples = int(np.ceil(audio_samples / chunk_size_samples) * chunk_size_samples)
    if full_samples > audio_samples:
        pad_size = full_samples - audio_samples
        mix = torch.nn.functional.pad(mix, (0, pad_size), mode="constant", value=0)

    chunks = mix.unfold(dimension=1, size=chunk_size_samples, step=chunk_size_samples)
    chunks = chunks.permute(1, 0, 2)
    clips_num = chunks.shape[0]

    pointer = 0
    outputs = []

    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available() and USE_FP16):
        with torch.inference_mode():
            while pointer < clips_num:
                batch_end = min(pointer + batch_size, clips_num)
                batch_chunks = chunks[pointer:batch_end].to(device)

                batch_output = model(batch_chunks)
                batch_output = batch_output.cpu()

                outputs.append(batch_output)
                pointer += batch_size

    outputs = torch.cat(outputs, dim=0)
    channels_num = outputs.shape[1]
    outputs = outputs.permute(1, 0, 2).reshape(channels_num, -1)
    outputs = outputs[:, :audio_samples]

    return outputs


def main():
    input_file = "/home/k4/Python/windowed-roformer/input.mp3"
    output_file = "/home/k4/Python/windowed-roformer/output_wsa_trt.wav"

    # Model and checkpoint configuration
    ckpt_name = "mbr-win10-sink8.ckpt"
    ckpt_url = "https://huggingface.co/smulelabs/windowed-roformer/resolve/main/mbr-win10-sink8.ckpt"

    # Inference parameters
    sample_rate = 44100
    chunk_size = 8  # Seconds per chunk
    batch_size = BATCH_SIZE

    # Check input file
    if not os.path.exists(input_file):
        logger.error(f"File {input_file} not found.")
        return

    # Download model
    try:
        download_ckpt_if_missing(ckpt_name, ckpt_url)
    except Exception:
        return

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device used: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"FP16 mode: {USE_FP16}")
    logger.info(f"Backend: {BACKEND}")

    if device != 'cuda':
        logger.warning("CUDA not available. Running without optimization.")

    # Load model
    logger.info("Initializing Windowed Roformer model...")
    try:
        model = load_model_trt(ckpt_name, device)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compile model with TensorRT
    if device == 'cuda':
        model = compile_model(model, device)

    # Load audio (I/O excluded from benchmark)
    logger.info("Loading audio into memory...")
    audio_tensor = load_audio(input_file, sample_rate)

    # Warmup runs to trigger JIT compilation
    logger.info("Warmup runs (triggers compilation)...")
    try:
        for warmup_i in range(3):
            separated_audio = demix_trt(
                model,
                audio_tensor,
                device=device,
                sample_rate=sample_rate,
                chunk_size=chunk_size,
                batch_size=batch_size,
            )
            if device == 'cuda':
                torch.cuda.synchronize()
            logger.info(f"Warmup {warmup_i + 1}/3 complete.")

        logger.info(f"Warmup complete. Output shape: {separated_audio.shape}")
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
            # Run inference
            separated_audio = demix_trt(
                model,
                audio_tensor,
                device=device,
                sample_rate=sample_rate,
                chunk_size=chunk_size,
                batch_size=batch_size,
            )
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

        # Save output (outside timing)
        save_audio(separated_audio, output_file, sample_rate)

    # Statistics
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        std_time = np.std(execution_times)

        logger.info(f"\n=== Benchmark Results (Windowed Roformer + {BACKEND}) ===")
        logger.info(f"Batch size:   {batch_size}")
        logger.info(f"Backend:      {BACKEND}")
        logger.info(f"FP16:         {USE_FP16}")
        logger.info(f"Average time: {avg_time:.2f} ms")
        logger.info(f"Std dev:      {std_time:.2f} ms")
        logger.info(f"Minimum:      {min_time:.2f} ms")
        logger.info(f"Maximum:      {max_time:.2f} ms")
        logger.info("=" * 58)


if __name__ == "__main__":
    main()
