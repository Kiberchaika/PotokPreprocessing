#!/usr/bin/env python3
"""
Benchmark script for Qwen3 ASR model.
Runs 10 inference passes and reports timing statistics.
"""
# pip install -U qwen-asr[vllm]

import os
# Set GPU before any imports
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import time
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

INPUT_FILE = "/home/k4/Python/windowed-roformer/input_vocal.wav"
INPUT_FILE = "/media/k4_nas/disk1/Datasets/Music_ASR_Test/Кино - 1988 - Группа крови/02 - Закрой за мной дверь, я ухожу_vocal.mp3"
NUM_RUNS = 10


def main():
    # Check input file
    if not os.path.exists(INPUT_FILE):
        logger.error(f"File not found: {INPUT_FILE}")
        return 1

    logger.info(f"Input file: {INPUT_FILE}")

    # Import after setting CUDA device
    import torch
    from qwen_asr import Qwen3ASRModel

    logger.info("Initializing Qwen3 ASR model...")

    try:
        # Initialize model with vLLM backend and forced aligner for timestamps
        model = Qwen3ASRModel.LLM(
            model="Qwen/Qwen3-ASR-1.7B",
            gpu_memory_utilization=0.9,
            max_inference_batch_size=8,
            max_new_tokens=2048,
            max_model_len=4096,
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        traceback.print_exc()
        return 1

    logger.info(f"Starting benchmark ({NUM_RUNS} runs)...")
    execution_times = []
    final_result = None

    for i in range(NUM_RUNS):
        logger.info(f"--- Run {i + 1}/{NUM_RUNS} ---")

        start_time = time.perf_counter()

        try:
            results = model.transcribe(
                audio=INPUT_FILE,
                language="Russian",
                return_time_stamps=False,
            )
            if results:
                final_result = results[0]
                text_preview = final_result.text[:100] if final_result.text else ""
                logger.info(f"Transcription: {text_preview}...")
        except Exception as e:
            logger.error(f"Inference error: {e}")
            traceback.print_exc()
            continue

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        execution_times.append(elapsed_ms)
        logger.info(f"Run {i + 1} time: {elapsed_ms:.2f} ms")

    # Print transcription result
    if final_result:
        logger.info("\n" + "=" * 60)
        logger.info("TRANSCRIPTION RESULT")
        logger.info("=" * 60)
        logger.info(f"\nFull text:\n{final_result.text}\n")

        # Print timestamps
        if hasattr(final_result, 'time_stamps') and final_result.time_stamps:
            logger.info("Word timestamps:")
            logger.info("-" * 40)
            for ts in final_result.time_stamps:
                start = getattr(ts, 'start_time', 0)
                end = getattr(ts, 'end_time', 0)
                text = getattr(ts, 'text', '')
                logger.info(f"[{start:.2f}s - {end:.2f}s] {text}")

        logger.info("=" * 60)

    # Benchmark statistics
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)

        logger.info("\n=== Benchmark Results (Qwen3 ASR) ===")
        logger.info(f"Average time: {avg_time:.2f} ms")
        logger.info(f"Min time:     {min_time:.2f} ms")
        logger.info(f"Max time:     {max_time:.2f} ms")
        logger.info("=====================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
