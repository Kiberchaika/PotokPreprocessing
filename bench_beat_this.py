import os
import time
import logging
import torch
import numpy as np

# Import Beat This! components
from beat_this.inference import Audio2Beats
from beat_this.preprocessing import load_audio

# Set CUDA to only use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Adjust paths as necessary
    input_file = "input.mp3"
    
    # Beat This! parameters
    checkpoint = "final0" # Default model
    use_dbn = False       # Beat This! aims to work well without DBN
    
    # Check input
    if not os.path.exists(input_file):
        logger.error(f"File {input_file} not found.")
        return

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device used: {device}")

    # Initialize Model
    logger.info("Initializing Beat This! model...")
    try:
        # Audio2Beats encapsulates the model, spectrogram generation, and post-processing
        model = Audio2Beats(
            checkpoint_path=checkpoint, 
            device=device, 
            dbn=use_dbn
        )
    except Exception as e:
        logger.error(f"Error loading Beat This!: {e}")
        return

    # Load Audio (I/O excluded from benchmark)
    logger.info("Loading audio into memory...")
    try:
        # load_audio returns (waveform, samplerate)
        audio_tensor, sample_rate = load_audio(input_file)
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
            # Audio2Beats takes the raw waveform and SR
            beats, downbeats = model(audio_tensor, sample_rate)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return

        # End timer
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
        execution_times.append(elapsed_ms)
        
        logger.info(f"Run {i + 1} time: {elapsed_ms:.2f} ms")

        # Optional validation
        if i == 0:
            logger.info(f"Inference successful. Found {len(beats)} beats and {len(downbeats)} downbeats.")

    # Statistics
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        logger.info("\n=== Benchmark Results (Beat This!) ===")
        logger.info(f"Average time: {avg_time:.2f} ms")
        logger.info(f"Minimum:      {min_time:.2f} ms")
        logger.info(f"Maximum:      {max_time:.2f} ms")
        logger.info("======================================")

if __name__ == "__main__":
    main()