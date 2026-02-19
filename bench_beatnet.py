import os
import time
import logging
import numpy as np
import librosa
import torch

# Import BeatNet
from BeatNet.BeatNet import BeatNet

# Set CUDA to only use GPU 1 (matching your base script)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Adjust paths as necessary
    input_file = "input.mp3"
    
    # BeatNet parameters
    # Model 1: GTZAN (standard), 2: Ballroom, 3: Rock Corpus
    model_id = 1 
    
    # Check input
    if not os.path.exists(input_file):
        logger.error(f"File {input_file} not found.")
        return

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device used: {device}")

    # Initialize Model
    logger.info("Initializing BeatNet model...")
    try:
        # We use mode='offline' with inference_model='DBN' for standard offline processing
        estimator = BeatNet(
            model=model_id, 
            mode='offline', 
            inference_model='DBN', 
            plot=[], 
            thread=False, 
            device=device
        )
    except Exception as e:
        logger.error(f"Error loading BeatNet: {e}")
        return

    # Load and Preprocess Audio (I/O excluded from benchmark)
    logger.info("Loading audio into memory (resampling to 22050Hz)...")
    try:
        # BeatNet explicitly requires 22050Hz. We load it here to avoid 
        # counting disk I/O and resampling in the inference timer.
        audio, _ = librosa.load(input_file, sr=22050)
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        return

    logger.info("Starting benchmark (10 runs)...")
    execution_times = []

    for i in range(10):
        logger.info(f"--- Run {i + 1}/10 ---")
        
        # Start timer
        start_time = time.perf_counter()
        
        try:
            # Run inference
            # BeatNet accepts the numpy array directly in process()
            output = estimator.process(audio)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return

        # End timer
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
        execution_times.append(elapsed_ms)
        
        logger.info(f"Run {i + 1} time: {elapsed_ms:.2f} ms")
        
        # Optional: Print first few beats to ensure it worked
        if i == 0:
            logger.info(f"Inference successful. Found {len(output)} beats.")

    # Statistics
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        logger.info("\n=== Benchmark Results (BeatNet) ===")
        logger.info(f"Average time: {avg_time:.2f} ms")
        logger.info(f"Minimum:      {min_time:.2f} ms")
        logger.info(f"Maximum:      {max_time:.2f} ms")
        logger.info("===================================")

if __name__ == "__main__":
    main()