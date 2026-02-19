#!/usr/bin/env python3
"""
WER (Word Error Rate) benchmark for Whisper Large v3 Turbo (openai/whisper-large-v3-turbo).
Finds paired .txt (lyrics) and *_vocal.mp3 files, transcribes audio and calculates WER.
Uses transformers pipeline with built-in long-form audio handling.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import re
import logging
import traceback
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

INPUT_DIR = "/media/k4_nas/disk1/Datasets/Music_ASR_Test/"
MODEL_NAME = "openai/whisper-large-v3-turbo"


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


def main():
    logger.info(f"Scanning {INPUT_DIR} for paired files...")
    pairs = find_paired_files(INPUT_DIR)
    logger.info(f"Found {len(pairs)} paired files (txt + _vocal.mp3)\n")

    if not pairs:
        logger.error("No paired files found!")
        return 1

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    logger.info(f"Loading model: {MODEL_NAME}...")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(MODEL_NAME)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        traceback.print_exc()
        return 1

    results = []
    total_ref_words = 0
    total_errors = 0

    for i, (txt_path, vocal_path) in enumerate(pairs, 1):
        rel_path = os.path.relpath(vocal_path, INPUT_DIR)
        logger.info(f"[{i}/{len(pairs)}] {rel_path}")

        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                reference = f.read().strip()

            if not reference:
                logger.warning("  Empty reference text, skipping")
                continue

            # Transcribe with pipeline (return_timestamps=True for long audio)
            output = pipe(
                vocal_path,
                generate_kwargs={"language": "russian", "task": "transcribe"},
                return_timestamps=True,
            )

            hypothesis = output.get("text", "")

            if not hypothesis:
                logger.warning("  Empty transcription, skipping")
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

    # Print summary
    if results:
        logger.info("\n" + "=" * 70)
        logger.info("WER BENCHMARK RESULTS (Whisper Large v3 Turbo)")
        logger.info("=" * 70)

        overall_wer = total_errors / total_ref_words if total_ref_words > 0 else 0

        wer_values = [r['wer'] for r in results]
        avg_wer = sum(wer_values) / len(wer_values)
        min_wer = min(wer_values)
        max_wer = max(wer_values)

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
