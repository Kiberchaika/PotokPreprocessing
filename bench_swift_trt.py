import os
import time
import logging
import numpy as np
import librosa
import onnxruntime

from swift_f0 import SwiftF0

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimization configuration
USE_FP16 = True

def main():
    input_file = "input.mp3"

    if not os.path.exists(input_file):
        logger.error(f"File {input_file} not found.")
        return

    # Initialize Model
    logger.info("Initializing SwiftF0 model...")
    try:
        detector = SwiftF0(
            confidence_threshold=0.9,
            fmin=46.875,
            fmax=2093.75,
        )
    except Exception as e:
        logger.error(f"Error loading SwiftF0: {e}")
        return

    # Replace CPU session with TensorRT session
    import swift_f0
    model_path = os.path.join(os.path.dirname(swift_f0.__file__), "model.onnx")

    trt_provider_options = {
        "trt_max_workspace_size": str(2 * 1024 * 1024 * 1024),  # 2GB
        "trt_fp16_enable": str(USE_FP16),
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": "/tmp/swift_f0_trt_cache",
    }
    os.makedirs("/tmp/swift_f0_trt_cache", exist_ok=True)

    providers = [
        ("TensorrtExecutionProvider", trt_provider_options),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    logger.info(f"Re-creating ONNX session with TensorRT (FP16={USE_FP16})...")
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    detector.pitch_session = onnxruntime.InferenceSession(
        model_path, session_options, providers=providers
    )
    detector.pitch_input_name = detector.pitch_session.get_inputs()[0].name
    active_providers = detector.pitch_session.get_providers()
    logger.info(f"Active providers: {active_providers}")

    # Load and Preprocess Audio (I/O excluded from benchmark)
    logger.info("Loading audio into memory (resampling to 16000Hz)...")
    try:
        audio, sr = librosa.load(input_file, sr=16000)
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        return

    # Warmup runs to trigger TensorRT engine build
    logger.info("Warmup runs (triggers TRT engine build)...")
    for warmup_i in range(3):
        result = detector.detect_from_array(audio, sample_rate=sr)
        logger.info(f"Warmup {warmup_i + 1}/3 complete.")
    logger.info(f"Warmup done. {len(result.pitch_hz)} frames, {np.sum(result.voicing)} voiced.")

    logger.info("Starting benchmark (10 runs)...")
    execution_times = []

    for i in range(10):
        logger.info(f"--- Run {i + 1}/10 ---")

        start_time = time.perf_counter()

        try:
            result = detector.detect_from_array(audio, sample_rate=sr)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return

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

        logger.info(f"\n=== Benchmark Results (SwiftF0 + TensorRT) ===")
        logger.info(f"FP16:         {USE_FP16}")
        logger.info(f"Average time: {avg_time:.2f} ms")
        logger.info(f"Std dev:      {std_time:.2f} ms")
        logger.info(f"Minimum:      {min_time:.2f} ms")
        logger.info(f"Maximum:      {max_time:.2f} ms")
        logger.info("=" * 48)

if __name__ == "__main__":
    main()
