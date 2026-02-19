#!/usr/bin/env python3
"""Split a dataset into N parts of roughly equal total file size.

Each artist folder is kept intact (never split across parts).
Parts are created as sibling directories: Music_Part1.01_Part01, _Part02, etc.

Usage:
    python split_dataset.py /home/k4/Datasets/Music_Part1.01 N [--dry-run] [--move]
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def get_dir_size(path: Path) -> int:
    """Total size of all files under path, in bytes."""
    total = 0
    for entry in os.scandir(path):
        if entry.is_file(follow_symlinks=False):
            total += entry.stat(follow_symlinks=False).st_size
        elif entry.is_dir(follow_symlinks=False):
            total += get_dir_size(Path(entry.path))
    return total


def human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def split_dataset(src: Path, n: int, dry_run: bool = False):
    # Collect artist folders with sizes
    entries = []
    for item in sorted(src.iterdir()):
        if item.is_dir():
            size = get_dir_size(item)
            entries.append((item.name, size))

    if not entries:
        print("No subdirectories found.")
        sys.exit(1)

    total_size = sum(s for _, s in entries)
    target_per_part = total_size / n

    print(f"Source:          {src}")
    print(f"Total folders:   {len(entries)}")
    print(f"Total size:      {human_size(total_size)}")
    print(f"Parts:           {n}")
    print(f"Target per part: {human_size(target_per_part)}")
    print()

    # Greedy bin-packing: iterate sorted folders, fill each bin up to target
    # Sort largest-first for better balance
    entries.sort(key=lambda x: x[1], reverse=True)

    bins: list[list[tuple[str, int]]] = [[] for _ in range(n)]
    bin_sizes = [0] * n

    for name, size in entries:
        # Put into the bin with the smallest current total
        min_idx = bin_sizes.index(min(bin_sizes))
        bins[min_idx].append((name, size))
        bin_sizes[min_idx] += size

    # Sort folders within each bin alphabetically
    for b in bins:
        b.sort(key=lambda x: x[0])

    # Print summary
    for i, (b, bsize) in enumerate(zip(bins, bin_sizes)):
        part_name = f"{src.name}_Part{i+1:02d}"
        print(f"  {part_name}: {len(b):>5} folders, {human_size(bsize):>10}")

    if dry_run:
        print("\n[DRY RUN] No files were moved/copied.")
        return

    # Create part directories and move folders
    print(f"\nMoving folders...")

    for i, b in enumerate(bins):
        part_dir = src.parent / f"{src.name}_Part{i+1:02d}"
        part_dir.mkdir(exist_ok=True)
        for name, _ in b:
            src_path = src / name
            dst_path = part_dir / name
            # Remove leftover symlinks from previous runs
            if dst_path.is_symlink():
                dst_path.unlink()
            if dst_path.exists():
                continue
            if not src_path.exists():
                continue
            shutil.move(str(src_path), str(dst_path))
        print(f"  {part_dir.name}: {len(b)} folders done")

    # Move loose files (not in subdirs) to Part01
    part1_dir = src.parent / f"{src.name}_Part01"
    part1_dir.mkdir(exist_ok=True)
    loose_files = [f for f in src.iterdir() if f.is_file()]
    if loose_files:
        print(f"\nMoving {len(loose_files)} loose files to {part1_dir.name}...")
        for f in loose_files:
            dst = part1_dir / f.name
            if not dst.exists():
                shutil.move(str(f), str(dst))

    # Remove source directory if empty (or only .blackbird left)
    remaining = [p for p in src.iterdir() if p.name != ".blackbird"]
    if not remaining:
        shutil.rmtree(str(src))
        print(f"Removed empty source directory: {src}")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into N parts by size")
    parser.add_argument("dataset", type=Path, help="Path to dataset directory")
    parser.add_argument("n", type=int, help="Number of parts to split into")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without doing anything")
    args = parser.parse_args()

    if not args.dataset.is_dir():
        print(f"Error: {args.dataset} is not a directory")
        sys.exit(1)
    if args.n < 2:
        print("Error: N must be at least 2")
        sys.exit(1)

    split_dataset(args.dataset, args.n, dry_run=args.dry_run)
 
'''
python3 split_dataset.py /home/k4/Datasets/Music_Part1.01 4 --dry-run   # preview
python3 split_dataset.py /home/k4/Datasets/Music_Part1.01 4             # execute
'''