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

    # Replace CPU session with CUDA session
    import swift_f0
    model_path = os.path.join(os.path.dirname(swift_f0.__file__), "model.onnx")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    logger.info(f"Re-creating ONNX session with providers: {providers}")
    session_options = onnxruntime.SessionOptions()
    session_options.inter_op_num_threads = 1
    session_options.intra_op_num_threads = 1
    detector.pitch_session = onnxruntime.InferenceSession(
        model_path, session_options, providers=providers
    )
    detector.pitch_input_name = detector.pitch_session.get_inputs()[0].name
    active_providers = detector.pitch_session.get_providers()
    logger.info(f"Active providers: {active_providers}")

    # Load and Preprocess Audio (I/O excluded from benchmark)
    # SwiftF0 internally resamples to 16kHz, load at 16kHz to keep it fair
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
            result = detector.detect_from_array(audio, sample_rate=sr)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return

        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        execution_times.append(elapsed_ms)

        logger.info(f"Run {i + 1} time: {elapsed_ms:.2f} ms")

        if i == 0:
            voiced = np.sum(result.voicing)
            logger.info(f"Inference successful. {len(result.pitch_hz)} frames, {voiced} voiced.")

    # Statistics
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)

        logger.info("\n=== Benchmark Results (SwiftF0) ===")
        logger.info(f"Average time: {avg_time:.2f} ms")
        logger.info(f"Minimum:      {min_time:.2f} ms")
        logger.info(f"Maximum:      {max_time:.2f} ms")
        logger.info("===================================")

if __name__ == "__main__":
    main()
