#!/usr/bin/env python3
"""Rsync only Music_Part1.02 Part01+Part02 (first two bins of a 4-way artist split).

Uses filelist_part1.02.txt and the same greedy packing as split_dataset.py.
Does not upload Part03/Part04 (~half of slice 2 by size).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = "188.120.253.126"
KEY = ROOT / "dev-233158-kiberchaika.pem"
BASE = Path("/media/k4_nas/disk1/Datasets/Music_Part1")
FILELIST = ROOT / "filelist_part1.02.txt"
INDEX = BASE / ".blackbird" / "index.pickle"
N_BINS = 4
KEEP_PARTS = ("01", "02")


def human(n: int) -> str:
    x = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(x) < 1024:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{x:.2f} PB"


def load_sizes(paths: list[str]) -> dict[str, int]:
    sys.path.insert(0, str(Path("/home/k4/Projects/BirdsMilkDatasetPreprocessing/The_Blackbird_Dataset")))
    from blackbird.index import DatasetIndex

    print(f"Loading {INDEX} ...", flush=True)
    idx = DatasetIndex.load(INDEX)
    wanted = set(paths)
    sizes: dict[str, int] = {}
    for t in idx.tracks.values():
        for fpath, sz in t.file_sizes.items():
            rel = fpath[5:] if fpath.startswith("Main/") else fpath
            if rel in wanted:
                sizes[rel] = sz
    missing = [p for p in paths if p not in sizes]
    print(f"sizes from index: {len(sizes)}/{len(paths)}; stat {len(missing)} missing", flush=True)
    for rel in missing:
        fp = BASE / rel
        try:
            sizes[rel] = fp.stat().st_size
        except OSError:
            sizes[rel] = 0
    return sizes


def pack(paths: list[str], sizes: dict[str, int]) -> list[list[str]]:
    artist_files: dict[str, list[str]] = defaultdict(list)
    artist_size: dict[str, int] = defaultdict(int)
    for rel in paths:
        artist = rel.split("/", 1)[0]
        artist_files[artist].append(rel)
        artist_size[artist] += sizes.get(rel, 0)

    entries = sorted(artist_size.items(), key=lambda x: x[1], reverse=True)
    bins: list[list[str]] = [[] for _ in range(N_BINS)]
    bin_sizes = [0] * N_BINS
    bin_artists: list[list[str]] = [[] for _ in range(N_BINS)]
    for artist, sz in entries:
        i = bin_sizes.index(min(bin_sizes))
        bin_artists[i].append(artist)
        bins[i].extend(artist_files[artist])
        bin_sizes[i] += sz

    for i, (arts, bsz, files) in enumerate(zip(bin_artists, bin_sizes, bins)):
        files.sort()
        print(f"  Part{i+1:02d}: {len(arts):5} artists, {len(files):7} mp3, {human(bsz)}")
    return bins


def rsync_part(part: str, list_path: Path) -> None:
    dest = f"root@{HOST}:/home/k4/Datasets/Music_Part1.02_Part{part}/"
    cmd = [
        "rsync", "-avz", "--progress", "--ignore-existing",
        "--files-from", str(list_path),
        "-e", f"ssh -i {KEY} -o StrictHostKeyChecking=no",
        f"{BASE}/",
        dest,
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ssh(*remote: str) -> None:
    subprocess.run(
        ["ssh", "-i", str(KEY), "-o", "StrictHostKeyChecking=no", f"root@{HOST}", *remote],
        check=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--write-lists", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    paths = [ln.strip() for ln in FILELIST.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"filelist {FILELIST.name}: {len(paths)} mp3")
    sizes = load_sizes(paths)
    bins = pack(paths, sizes)

    lists: dict[str, Path] = {}
    for part, files in zip(KEEP_PARTS, bins[:2]):
        lp = ROOT / f"filelist_part1.02_Part{part}.txt"
        lp.write_text("\n".join(files) + "\n", encoding="utf-8")
        lists[part] = lp
        print(f"wrote {lp.name} ({len(files)} files)")

    if args.write_lists:
        return

    for part in KEEP_PARTS:
        ssh(f"mkdir -p /home/k4/Datasets/Music_Part1.02_Part{part}")
        if args.dry_run:
            print(f"DRY RUN skip rsync Part{part}")
            continue
        rsync_part(part, lists[part])


if __name__ == "__main__":
    main()
