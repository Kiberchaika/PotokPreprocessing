#!/usr/bin/env python3
"""Rsync Music_Part1.01 filelist entries that are missing on the VDS.

A track counts as present if the mp3 or a sibling
(_voc.opus / _music.opus / _beats.json / _lyrics.json / .json) exists
in any Music_Part1.01_Part0N index.

Missing mp3s are sent to the Part where that artist already lives.
Artists that are not on the VDS at all go to --fallback-part (default 04).

  python upload_missing_part1_01.py --write-lists
  python upload_missing_part1_01.py --dry-run
  python upload_missing_part1_01.py          # rsync + remote reindex
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = "188.120.253.126"
KEY = ROOT / "dev-233158-kiberchaika.pem"
BASE = Path("/media/k4_nas/disk1/Datasets/Music_Part1")
FILELIST = ROOT / "filelist_part1.01.txt"
SIBLINGS = ("_voc.opus", "_music.opus", "_beats.json", "_lyrics.json", ".json")
REMOTE_PART = "/home/k4/Datasets/Music_Part1.01_Part{part}"


def ssh_cmd(*remote: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(KEY),
        "-o",
        "StrictHostKeyChecking=no",
        f"root@{HOST}",
        *remote,
    ]


def fetch_vds_files_and_artists() -> tuple[set[str], dict[str, str]]:
    script = r"""
from pathlib import Path
import pickle, json
files=set(); artist_part={}
parts=["01","02","03","04"]
for part in parts:
    root=Path(f"/home/k4/Datasets/Music_Part1.01_Part{part}")
    with open(root/".blackbird"/"index.pickle","rb") as f:
        idx=pickle.load(f)
    for t in idx.tracks.values():
        for p in t.files.values():
            rel=p[5:] if p.startswith("Main/") else p
            files.add(rel)
            a=rel.split("/")[0]
            artist_part.setdefault(a, part)
Path("/tmp/vds_part1_01_allfiles.txt").write_text("\n".join(sorted(files))+"\n", encoding="utf-8")
Path("/tmp/vds_artist_to_part.json").write_text(json.dumps(artist_part, ensure_ascii=False), encoding="utf-8")
print(len(files), len(artist_part))
"""
    subprocess.run(ssh_cmd("source /home/k4/.venv/bin/activate && python3 -"), input=script, text=True, check=True)
    dest = Path("/tmp")
    subprocess.run(
        [
            "scp",
            "-q",
            "-i",
            str(KEY),
            "-o",
            "StrictHostKeyChecking=no",
            f"root@{HOST}:/tmp/vds_part1_01_allfiles.txt",
            f"root@{HOST}:/tmp/vds_artist_to_part.json",
            str(dest),
        ],
        check=True,
    )
    vds = {ln.strip() for ln in (dest / "vds_part1_01_allfiles.txt").read_text(encoding="utf-8").splitlines() if ln.strip()}
    amap = json.loads((dest / "vds_artist_to_part.json").read_text(encoding="utf-8"))
    return vds, amap


def present_on_vds(mp3: str, vds_files: set[str]) -> bool:
    if mp3 in vds_files:
        return True
    if not mp3.lower().endswith(".mp3"):
        return False
    stem = mp3[:-4]
    return any(f"{stem}{suf}" in vds_files for suf in SIBLINGS)


def classify(filelist: Path, vds_files: set[str], artist_part: dict[str, str], fallback: str) -> dict[str, list[str]]:
    by_part: dict[str, list[str]] = defaultdict(list)
    for line in filelist.read_text(encoding="utf-8").splitlines():
        mp3 = line.strip()
        if not mp3 or present_on_vds(mp3, vds_files):
            continue
        artist = mp3.split("/")[0]
        part = artist_part.get(artist, fallback)
        by_part[part].append(mp3)
    for part in by_part:
        by_part[part].sort()
    return dict(by_part)


def write_lists(by_part: dict[str, list[str]], out_dir: Path) -> Path:
    combined: list[str] = []
    for part in sorted(by_part):
        paths = by_part[part]
        combined.extend(paths)
        p = out_dir / f"filelist_part1.01_missing_Part{part}.txt"
        p.write_text("\n".join(paths) + "\n", encoding="utf-8")
        print(f"  Part{part}: {len(paths):>6} files -> {p.name}")
    all_path = out_dir / "filelist_part1.01_missing.txt"
    all_path.write_text("\n".join(combined) + "\n", encoding="utf-8")
    print(f"  total : {len(combined):>6} files -> {all_path.name}")
    return all_path


def rsync_part(part: str, list_path: Path) -> None:
    dest = f"root@{HOST}:{REMOTE_PART.format(part=part)}/"
    # -a does not delete dest files. --ignore-existing never overwrites
    # opus/json/mp3 that are already on the VDS.
    cmd = [
        "rsync",
        "-avz",
        "--progress",
        "--ignore-existing",
        "--files-from",
        str(list_path),
        "-e",
        f"ssh -i {KEY} -o StrictHostKeyChecking=no",
        f"{BASE}/",
        dest,
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def reindex_parts(parts: list[str]) -> None:
    remote = "source /home/k4/.venv/bin/activate; " + "; ".join(
        f"blackbird reindex /home/k4/Datasets/Music_Part1.01_Part{p}" for p in parts
    )
    print("reindex:", remote)
    subprocess.run(ssh_cmd(remote), check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--filelist", type=Path, default=FILELIST)
    p.add_argument("--fallback-part", default="04", help="Part for artists not yet on VDS")
    p.add_argument("--write-lists", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-reindex", action="store_true")
    p.add_argument("--use-cached-vds", action="store_true", help="Reuse /tmp/vds_part1_01_allfiles.txt")
    args = p.parse_args()

    if not args.filelist.is_file():
        raise SystemExit(f"Missing {args.filelist}")
    if not KEY.is_file():
        raise SystemExit(f"Missing SSH key {KEY}")

    if args.use_cached_vds and Path("/tmp/vds_part1_01_allfiles.txt").is_file():
        vds = {
            ln.strip()
            for ln in Path("/tmp/vds_part1_01_allfiles.txt").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
        amap = json.loads(Path("/tmp/vds_artist_to_part.json").read_text(encoding="utf-8"))
    else:
        print("Fetching VDS index file lists...")
        vds, amap = fetch_vds_files_and_artists()

    by_part = classify(args.filelist, vds, amap, args.fallback_part)
    n = sum(len(v) for v in by_part.values())
    print(f"Missing on VDS: {n} mp3s from {args.filelist.name}")
    write_lists(by_part, ROOT)

    if args.write_lists or n == 0:
        return

    for part, paths in sorted(by_part.items()):
        lst = ROOT / f"filelist_part1.01_missing_Part{part}.txt"
        print(f"\n=== rsync {len(paths)} files -> Part{part} ===")
        if args.dry_run:
            print(f"DRY RUN rsync --files-from={lst.name} -> {REMOTE_PART.format(part=part)}/")
            continue
        rsync_part(part, lst)

    if args.dry_run or args.skip_reindex:
        return
    reindex_parts(sorted(by_part))


if __name__ == "__main__":
    main()
