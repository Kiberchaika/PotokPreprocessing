#!/usr/bin/env python3
"""Build rsync --files-from lists for consecutive Music_Part1 TB slices.

Slice 1 = first --size-tb of mp3 (sorted by relative path) → filelist_part1.01.txt
Slice 2 = the next window, no overlap with slice 1            → filelist_part1.02.txt

Fast path (default): read NAS .blackbird/index.pickle — seconds, not hours.
Disk walk: --scan-disk [--jobs 32] if the index might be stale.

  python make_filelists_for_transfer.py --slice 1
  python make_filelists_for_transfer.py --slice 2
  python make_filelists_for_transfer.py --slice 1 --scan-disk --jobs 32
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_PATH = Path("/media/k4_nas/disk1/Datasets/Music_Part1")
DEFAULT_SIZE_TB = 1.5
DEFAULT_INDEX = BASE_PATH / ".blackbird" / "index.pickle"
SKIP_NAMES = {".blackbird", ".", ".."}
FILE_EXTS = {".mp3", ".json", ".opus", ".pt", ".bak", ".pickle", ".jpg", ".png", ".txt"}
BLACKBIRD_SRC = Path("/home/k4/Projects/BirdsMilkDatasetPreprocessing/The_Blackbird_Dataset")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


def _tb(n: int) -> str:
    return f"{n / 1024**4:.3f} TB"


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(iterable, **kwargs)


def mp3s_from_index(index_path: Path) -> list[tuple[str, int]]:
    """Relative paths under Music_Part1 (strip location prefix Main/)."""
    if BLACKBIRD_SRC.is_dir() and str(BLACKBIRD_SRC) not in sys.path:
        sys.path.insert(0, str(BLACKBIRD_SRC))
    from blackbird.index import DatasetIndex

    print(f"Loading index {index_path} ...", flush=True)
    idx = DatasetIndex.load(index_path)
    print(f"Index updated {getattr(idx, 'last_updated', '?')}", flush=True)
    files: list[tuple[str, int]] = []
    for track in idx.tracks.values():
        mp3 = track.files.get("mp3")
        if not mp3:
            continue
        rel = mp3
        if rel.startswith("Main/"):
            rel = rel[len("Main/") :]
        elif rel.startswith("Part2/"):
            # not on this NAS folder
            continue
        size = track.file_sizes.get(mp3, 0)
        files.append((rel, size))
    print("Sorting paths...", flush=True)
    files.sort(key=lambda x: x[0])
    return files


def _walk_artist_mp3s(artist_path: str, base: str) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for root, _dirs, filenames in os.walk(artist_path, followlinks=False):
        for name in filenames:
            if not name.lower().endswith(".mp3"):
                continue
            full = os.path.join(root, name)
            try:
                size = os.stat(full, follow_symlinks=False).st_size
            except OSError:
                continue
            found.append((os.path.relpath(full, base), size))
    return found


def mp3s_from_disk(base: Path, jobs: int) -> list[tuple[str, int]]:
    """listdir names only (no per-entry STAT), then parallel os.walk."""
    from concurrent.futures import wait, FIRST_COMPLETED

    print("Listing names (no is_dir/stat per entry)...", flush=True)
    try:
        names = os.listdir(base)
    except OSError as e:
        raise SystemExit(f"Cannot list {base}: {e}") from e
    print(f"Got {len(names):,} top-level names. Classifying (no NAS calls)...", flush=True)

    artist_dirs: list[str] = []
    loose: list[tuple[str, int]] = []
    base_s = str(base)
    for name in _progress(names, desc="Classify", unit="name"):
        if name in SKIP_NAMES:
            continue
        ext = os.path.splitext(name)[1].lower()
        # No stat: a top-level "foo.mp3" folder would be skipped (none expected).
        if ext in FILE_EXTS:
            continue
        artist_dirs.append(os.path.join(base_s, name))
    artist_dirs.sort()
    workers = max(1, min(jobs, 8))
    if jobs > 8:
        print(f"Capping --jobs {jobs} → {workers} (CIFS stalls with high parallelism)", flush=True)
    print(
        f"Walking {len(artist_dirs):,} folders, {workers} threads. "
        f"First completions can take minutes on NAS...",
        flush=True,
    )

    files = list(loose)
    n_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_walk_artist_mp3s, d, base_s) for d in artist_dirs}
        total = len(pending)
        done_n = 0
        bar = None
        if tqdm is not None:
            bar = tqdm(total=total, desc="Scanning", unit="artist", dynamic_ncols=True, file=sys.stderr)
        while pending:
            finished, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            if not finished:
                print(
                    f"heartbeat: {done_n}/{total} artists finished, waiting on CIFS...",
                    flush=True,
                )
                continue
            for fut in finished:
                chunk = fut.result()
                files.extend(chunk)
                n_bytes += sum(s for _, s in chunk)
                done_n += 1
                if tqdm is not None:
                    bar.update(1)
                    bar.set_postfix_str(f"{len(files):,} mp3  {_tb(n_bytes)}", refresh=False)
                elif done_n == 1 or done_n % 20 == 0:
                    print(
                        f"Scanning {done_n}/{total} artists  {len(files):,} mp3  {_tb(n_bytes)}",
                        flush=True,
                    )
        if tqdm is not None:
            bar.close()

    print("Sorting paths...", flush=True)
    files.sort(key=lambda x: x[0])
    return files


def window(files: list[tuple[str, int]], slice_n: int, max_bytes: int) -> tuple[list[str], int, int]:
    """Pack files into successive max_bytes bins (same rule as original slice 1)."""
    current = 1
    used = 0
    skipped = 0
    selected: list[str] = []
    selected_bytes = 0

    for rel_path, size in files:
        if used > 0 and used + size > max_bytes:
            current += 1
            used = 0
        if current < slice_n:
            skipped += size
            used += size
            continue
        if current > slice_n:
            break
        selected.append(rel_path)
        selected_bytes += size
        used += size

    return selected, selected_bytes, skipped


def main() -> None:
    p = argparse.ArgumentParser(description="Build Music_Part1 rsync filelists by TB slice")
    p.add_argument("--slice", type=int, default=1, help="1-based slice number (1=.01, 2=.02, …)")
    p.add_argument("--size-tb", type=float, default=DEFAULT_SIZE_TB)
    p.add_argument("--base", type=Path, default=BASE_PATH)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help="Blackbird index.pickle (default). Ignored with --scan-disk.",
    )
    p.add_argument(
        "--scan-disk",
        action="store_true",
        help="Walk NAS instead of using the index (slow even when parallelized).",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Parallel artist walks for --scan-disk (capped at 8; CIFS dies at 32)",
    )
    args = p.parse_args()

    if args.slice < 1:
        raise SystemExit("--slice must be >= 1")

    label = f"{args.slice:02d}"
    out = args.output or Path(f"filelist_part1.{label}.txt")
    max_bytes = int(args.size_tb * 1024**4)

    if args.scan_disk:
        print(f"Disk scan: {args.base}", flush=True)
        files = mp3s_from_disk(args.base, args.jobs)
    else:
        if not args.index.is_file():
            raise SystemExit(
                f"No index at {args.index}. Pass --scan-disk or a valid --index."
            )
        files = mp3s_from_index(args.index)

    total_all = sum(s for _, s in files)
    print(f"Total mp3 files found: {len(files)}")
    print(f"Total size of all files: {total_all / 1024**4:.3f} TB")

    selected, selected_bytes, skipped = window(files, args.slice, max_bytes)
    remaining_after = total_all - skipped - selected_bytes

    print(f"\nSlice {args.slice} ({args.size_tb:g} TB window)")
    print(f"Skipped before window: {skipped / 1024**4:.3f} TB")
    print(f"Files selected: {len(selected)} / {len(files)}")
    print(f"Selected size:  {selected_bytes / 1024**4:.3f} TB / {args.size_tb:g} TB")
    print(f"Remaining after this slice: {remaining_after / 1024**4:.3f} TB")

    if not selected:
        raise SystemExit("No files in this slice — Music_Part1 is smaller than the skip window")

    out.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(f"\nFilelist saved: {out.resolve()}")
    dest = f"/home/k4/Datasets/Music_Part1.{label}"
    print("\nrsync:")
    print(
        f'rsync -avz --progress --files-from={out.name} \\\n'
        f'  -e "ssh -i dev-233158-kiberchaika.pem" \\\n'
        f'  {args.base}/ \\\n'
        f'  root@188.120.253.126:{dest}'
    )


if __name__ == "__main__":
    main()
