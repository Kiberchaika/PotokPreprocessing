#!/usr/bin/env python3
"""Scan MP3 files and calculate total duration (excluding _vocal/_music files)."""

import glob
from pathlib import Path
from mutagen.mp3 import MP3
from tqdm import tqdm


def main():
    # Find all mp3 files
    base_dir = "/media/k4_nas/disk1/Datasets/Music_Part1/"
    #base_dir = "/media/k4_nas/disk2/Music_Part2/"
    all_mp3 = glob.glob(f"{base_dir}/**/*.mp3", recursive=True)

    # Filter out _vocal and _music files
    filtered = [
        f for f in all_mp3
        if not f.endswith("_vocal.mp3") and not f.endswith("_music.mp3")
    ]

    print(f"Found {len(all_mp3)} total MP3 files")
    print(f"After filtering _vocal/_music: {len(filtered)} files")
    print()

    total_duration = 0.0
    errors = []

    for filepath in tqdm(filtered, desc="Scanning", unit="file"):
        try:
            audio = MP3(filepath)
            total_duration += audio.info.length
        except Exception as e:
            errors.append((filepath, str(e)))

    # Convert to hours, minutes, seconds
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)

    print()
    print(f"Total duration: {hours}h {minutes}m {seconds}s ({total_duration:.1f} seconds)")
    print(f"Files processed: {len(filtered) - len(errors)}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for path, err in errors[:10]:
            print(f"  {path}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
