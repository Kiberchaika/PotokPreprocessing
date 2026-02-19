#!/usr/bin/env python3
"""
WER (Word Error Rate) benchmark for NVIDIA Parakeet TDT ASR model with TensorRT optimization.
Finds paired .txt (lyrics) and *_vocal.mp3 files, transcribes audio and calculates WER.
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

import re
import logging
import tempfile
import traceback
import warnings
from pathlib import Path

import torch
import torchaudio

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

INPUT_DIR = "/media/k4_nas/disk1/Datasets/Music_ASR_Test/"
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
TARGET_SR = 16000

# Optimization configuration
BATCH_SIZE = 1  # Configurable batch size
USE_FP16 = True  # Use FP16 for better performance
# Backend options: "tensorrt", "inductor", "cudagraphs"
# Note: NeMo models may have complex ops, inductor is more compatible
BACKEND = "inductor"  # Use inductor (more compatible with NeMo models)


def normalize_text(text: str) -> str:
    """Normalize text for WER calculation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text


def calculate_wer(reference: str, hypothesis: str) -> tuple[float, int, int, int, int]:
    """
    Calculate Word Error Rate using dynamic programming.
    Returns: (wer, substitutions, deletions, insertions, total_words)
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0, 0, 0, len(hyp_words), 0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j - 1] + 1, d[i][j - 1] + 1, d[i - 1][j] + 1)

    i, j = len(ref_words), len(hyp_words)
    substitutions = deletions = insertions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            deletions += 1
            i -= 1

    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer, substitutions, deletions, insertions, len(ref_words)


def find_paired_files(directory: str) -> list[tuple[str, str]]:
    """Find all paired .txt and *_vocal.mp3 files."""
    pairs = []

    for root, _, files in os.walk(directory):
        txt_files = {f.removesuffix('.txt') for f in files if f.lower().endswith('.txt')}
        vocal_files = {f for f in files if f.lower().endswith('_vocal.mp3')}

        for vocal_file in vocal_files:
            base_name = vocal_file.removesuffix('_vocal.mp3')
            if base_name in txt_files:
                txt_path = os.path.join(root, base_name + '.txt')
                vocal_path = os.path.join(root, vocal_file)
                pairs.append((txt_path, vocal_path))

    return sorted(pairs)


def prepare_audio(input_path: str) -> str:
    """Convert audio to mono 16kHz WAV for Parakeet model."""
    waveform, sr = torchaudio.load(input_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        waveform = resampler(waveform)

    tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    torchaudio.save(tmp_file.name, waveform, TARGET_SR)
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
    logger.info(f"Backend: {BACKEND}")
    logger.info(f"FP16 mode: {USE_FP16}")
    logger.info(f"Scanning {INPUT_DIR} for paired files...")
    pairs = find_paired_files(INPUT_DIR)
    logger.info(f"Found {len(pairs)} paired files (txt + _vocal.mp3)\n")

    if not pairs:
        logger.error("No paired files found!")
        return 1

    import nemo.collections.asr as nemo_asr

    logger.info(f"Loading model: {MODEL_NAME}...")

    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        asr_model.eval()
        if torch.cuda.is_available():
            asr_model = asr_model.cuda()
        logger.info("Model loaded successfully!")

        # Compile encoder with torch.compile for optimization
        if torch.cuda.is_available() and hasattr(asr_model, 'encoder'):
            logger.info("Compiling encoder module...")
            asr_model.encoder = compile_model(asr_model.encoder)

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        traceback.print_exc()
        return 1

    # Warmup run
    logger.info("Warmup run...")
    try:
        first_vocal = pairs[0][1]
        prepared_file = prepare_audio(first_vocal)
        _ = asr_model.transcribe([prepared_file], timestamps=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        os.remove(prepared_file)
        logger.info("Warmup complete.")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")

    results = []
    total_ref_words = 0
    total_errors = 0

    for i, (txt_path, vocal_path) in enumerate(pairs, 1):
        rel_path = os.path.relpath(vocal_path, INPUT_DIR)
        logger.info(f"[{i}/{len(pairs)}] {rel_path}")

        prepared_file = None
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                reference = f.read().strip()

            if not reference:
                logger.warning("  Empty reference text, skipping")
                continue

            # Prepare audio (convert to mono 16kHz)
            prepared_file = prepare_audio(vocal_path)

            # Transcribe
            output = asr_model.transcribe([prepared_file], timestamps=False)

            if not output or not output[0]:
                logger.warning("  Empty transcription, skipping")
                continue

            # Get text from output
            if hasattr(output[0], 'text'):
                hypothesis = output[0].text
            else:
                hypothesis = str(output[0])

            if not hypothesis:
                logger.warning("  Empty transcription text, skipping")
                continue

            # Calculate WER
            wer, subs, dels, ins, ref_words = calculate_wer(reference, hypothesis)
            hyp_words = len(normalize_text(hypothesis).split())
            errors = subs + dels + ins

            total_ref_words += ref_words
            total_errors += errors

            results.append({
                'file': rel_path,
                'wer': wer,
                'ref_words': ref_words,
                'hyp_words': hyp_words,
                'substitutions': subs,
                'deletions': dels,
                'insertions': ins,
            })

            halluc_flag = " [HALLUCINATION]" if hyp_words > ref_words * 2 else ""
            logger.info(f"  WER: {wer*100:.1f}% (S:{subs} D:{dels} I:{ins}) [ref:{ref_words} hyp:{hyp_words}]{halluc_flag}")

        except Exception as e:
            logger.error(f"  Error: {e}")
            traceback.print_exc()

        finally:
            if prepared_file and os.path.exists(prepared_file):
                os.remove(prepared_file)

    # Print summary
    if results:
        logger.info("\n" + "=" * 70)
        logger.info(f"WER BENCHMARK RESULTS (Parakeet TDT + {BACKEND})")
        logger.info("=" * 70)

        overall_wer = total_errors / total_ref_words if total_ref_words > 0 else 0

        wer_values = [r['wer'] for r in results]
        avg_wer = sum(wer_values) / len(wer_values)
        min_wer = min(wer_values)
        max_wer = max(wer_values)

        logger.info(f"\nBackend: {BACKEND}")
        logger.info(f"FP16: {USE_FP16}")
        logger.info(f"\nFiles processed: {len(results)}")
        logger.info(f"Total reference words: {total_ref_words}")
        logger.info(f"Total errors: {total_errors}")
        logger.info(f"\nOverall WER: {overall_wer*100:.2f}%")
        logger.info(f"Average WER (per file): {avg_wer*100:.2f}%")
        logger.info(f"Min WER: {min_wer*100:.2f}%")
        logger.info(f"Max WER: {max_wer*100:.2f}%")

        logger.info("\n--- Top 5 Best ---")
        sorted_best = sorted(results, key=lambda x: x['wer'])[:5]
        for r in sorted_best:
            logger.info(f"  {r['wer']*100:5.1f}% - {r['file']}")

        logger.info("\n--- Top 5 Worst ---")
        sorted_worst = sorted(results, key=lambda x: x['wer'], reverse=True)[:5]
        for r in sorted_worst:
            ratio = r['hyp_words'] / r['ref_words'] if r['ref_words'] > 0 else 0
            halluc = " [HALLUC]" if ratio > 2 else ""
            logger.info(f"  {r['wer']*100:5.1f}% - {r['file']} (ref:{r['ref_words']} hyp:{r['hyp_words']}){halluc}")

        halluc_count = sum(1 for r in results if r['hyp_words'] > r['ref_words'] * 2)
        if halluc_count > 0:
            logger.info(f"\nWarning: {halluc_count} files had hallucinations (hyp > 2x ref)")

        logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
