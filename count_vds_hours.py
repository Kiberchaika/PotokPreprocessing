#!/usr/bin/env python3
"""Sum original-track duration on the Blackbird VDS across WebDAV ports.

Does not download MP3/Opus. Each port's `.blackbird/index.pickle` already has
vocal-opus sizes. A sample of tiny `_beats.json` files (last beat ≈ track
length) calibrates bytes→seconds; that ratio is applied to every track.

Use --exact to download beats/lyrics for every track (slow).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

from blackbird.index import DatasetIndex, TrackInfo

logger = logging.getLogger("count_vds_hours")

DEFAULT_HOST = "188.120.253.126"
DEFAULT_PORTS = [8091, 8092, 8093, 8094, 8095, 8096]
DEFAULT_USER = "blackbird"
DEFAULT_PASS = "dataset"
VOCAL_BPS = 96_000

_tls = threading.local()


def format_hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m:02d}m {s:05.2f}s"


def format_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{n} B"


def parse_ports(raw: Optional[Sequence[str]]) -> List[int]:
    if not raw:
        return list(DEFAULT_PORTS)
    ports: List[int] = []
    for item in raw:
        for part in str(item).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            ports.append(int(part))
    return ports or list(DEFAULT_PORTS)


def remote_relpath(symbolic: str) -> str:
    parts = symbolic.split("/", 1)
    return parts[1] if len(parts) == 2 else symbolic


def make_session(auth: HTTPBasicAuth) -> requests.Session:
    session = requests.Session()
    session.auth = auth
    session.trust_env = False
    session.headers["User-Agent"] = "count_vds_hours/1.0"
    return session


def thread_session(auth: HTTPBasicAuth) -> requests.Session:
    sess = getattr(_tls, "session", None)
    if sess is None:
        sess = make_session(auth)
        _tls.session = sess
    return sess


def dav_get(
    session: requests.Session,
    base: str,
    remote_path: str,
    dest: Path,
    timeout: int = 180,
) -> None:
    url = f"{base}/{quote(remote_path.lstrip('/'), safe='/')}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    fh.write(chunk)


def dav_get_json(
    session: requests.Session,
    base: str,
    remote_path: str,
    timeout: float,
) -> Dict[str, Any]:
    url = f"{base}/{quote(remote_path.lstrip('/'), safe='/')}"
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError("json is not an object")
    return payload


def max_timestamp(values: Any) -> Optional[float]:
    best: Optional[float] = None
    if not isinstance(values, list):
        return None
    for item in values:
        try:
            val = float(item)
        except (TypeError, ValueError):
            continue
        if best is None or val > best:
            best = val
    return best


def duration_from_beats(payload: Dict[str, Any]) -> Optional[float]:
    a = max_timestamp(payload.get("beats"))
    b = max_timestamp(payload.get("downbeats"))
    vals = [x for x in (a, b) if x is not None]
    return max(vals) if vals else None


def duration_from_lyrics(payload: Dict[str, Any]) -> Optional[float]:
    words = payload.get("words") or []
    best: Optional[float] = None
    for w in words:
        try:
            val = float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if best is None or val > best:
            best = val
    return best


def vocal_bytes(track: TrackInfo) -> int:
    path = track.files.get("vocal")
    if not path:
        return 0
    return int(track.file_sizes.get(path, 0) or 0)


def mp3_bytes(track: TrackInfo) -> int:
    path = track.files.get("mp3")
    if not path:
        return 0
    return int(track.file_sizes.get(path, 0) or 0)


def load_index(host: str, port: int, auth: HTTPBasicAuth, work_dir: Path) -> DatasetIndex:
    base = f"http://{host}:{port}"
    cache = work_dir / f"index_{port}.pickle"
    logger.info("Port %s: downloading index ...", port)
    dav_get(make_session(auth), base, ".blackbird/index.pickle", cache)
    idx = DatasetIndex.load(cache)
    logger.info(
        "Port %s: %d tracks, index %s, updated %s",
        port, len(idx.tracks), format_bytes(cache.stat().st_size),
        getattr(idx, "last_updated", "?"),
    )
    return idx


def fetch_sidecars_duration(
    host: str,
    port: int,
    auth: HTTPBasicAuth,
    track: TrackInfo,
    timeout: float,
) -> Optional[float]:
    base = f"http://{host}:{port}"
    session = thread_session(auth)
    if "beats" in track.files:
        try:
            payload = dav_get_json(session, base, remote_relpath(track.files["beats"]), timeout)
            dur = duration_from_beats(payload)
            if dur and dur > 0:
                return dur
        except Exception:
            pass
    if "lyrics" in track.files:
        try:
            payload = dav_get_json(session, base, remote_relpath(track.files["lyrics"]), timeout)
            dur = duration_from_lyrics(payload)
            if dur and dur > 0:
                return dur
        except Exception:
            pass
    return None


def sample_rates(
    host: str,
    port: int,
    auth: HTTPBasicAuth,
    tracks: Sequence[TrackInfo],
    sample_n: int,
    workers: int,
    timeout: float,
) -> Dict[str, Any]:
    """Calibrate seconds-per-byte for vocal opus and source mp3 from a beats sample."""
    with_beats = [t for t in tracks if "beats" in t.files]
    voc_cands = [t for t in with_beats if vocal_bytes(t) > 0]
    mp3_cands = [t for t in with_beats if mp3_bytes(t) > 0]
    rng = random.Random(port)
    chosen: Dict[str, TrackInfo] = {}
    for pool in (voc_cands, mp3_cands):
        if not pool:
            continue
        n = min(sample_n, len(pool))
        for t in rng.sample(pool, n):
            chosen[t.track_path] = t
    if not chosen:
        logger.warning("Port %s: no beats sample, falling back to 96k/256k estimates", port)
        return {
            "voc_spb": 8.0 / VOCAL_BPS,
            "mp3_spb": 8.0 / 256_000,
            "sample_ok": 0,
            "voc_pairs": 0,
            "mp3_pairs": 0,
            "mean_dur": 0.0,
        }

    voc_pairs: List[Tuple[int, float]] = []
    mp3_pairs: List[Tuple[int, float]] = []
    sample = list(chosen.values())
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(fetch_sidecars_duration, host, port, auth, t, timeout): t
            for t in sample
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"calibrate {port}", unit="trk"):
            track = futs[fut]
            dur = fut.result()
            if not dur or dur <= 0:
                continue
            vb = vocal_bytes(track)
            mb = mp3_bytes(track)
            if vb > 0:
                voc_pairs.append((vb, dur))
            if mb > 0:
                mp3_pairs.append((mb, dur))

    def spb(pairs: List[Tuple[int, float]], fallback: float) -> float:
        if not pairs:
            return fallback
        return sum(d for _, d in pairs) / sum(s for s, _ in pairs)

    all_durs = [d for _, d in voc_pairs] or [d for _, d in mp3_pairs]
    return {
        "voc_spb": spb(voc_pairs, 8.0 / VOCAL_BPS),
        "mp3_spb": spb(mp3_pairs, 8.0 / 256_000),
        "sample_ok": max(len(voc_pairs), len(mp3_pairs)),
        "voc_pairs": len(voc_pairs),
        "mp3_pairs": len(mp3_pairs),
        "mean_dur": (sum(all_durs) / len(all_durs)) if all_durs else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Count total audio hours on the VDS Blackbird WebDAV ports",
    )
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument(
        "--ports", nargs="*", default=None, metavar="PORT",
        help="Ports to scan (default: %s)" % ",".join(str(x) for x in DEFAULT_PORTS),
    )
    p.add_argument("--username", default=DEFAULT_USER)
    p.add_argument("--password", default=DEFAULT_PASS)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument(
        "--sample", type=int, default=250,
        help="Tracks per port used to calibrate vocal-bytes → seconds (default: 250)",
    )
    p.add_argument(
        "--exact", action="store_true",
        help="Download beats/lyrics for every track (slow, most accurate)",
    )
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def summarize_port(
    port: int,
    index: DatasetIndex,
    seconds: float,
    parsed: int,
    method: str,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    tracks = list(index.tracks.values())
    info = {
        "tracks": len(tracks),
        "parsed": parsed,
        "seconds": seconds,
        "hours": seconds / 3600 if seconds else 0.0,
        "artists": len(index.album_by_artist),
        "albums": len(index.track_by_album),
        "with_mp3": sum(1 for t in tracks if "mp3" in t.files),
        "with_vocal": sum(1 for t in tracks if "vocal" in t.files),
        "with_beats": sum(1 for t in tracks if "beats" in t.files),
        "method": method,
        "index_updated": str(getattr(index, "last_updated", "")),
        **extra,
    }
    print()
    print(f"Port {port}")
    print(f"  tracks:     {info['tracks']}  parsed={parsed}")
    print(f"  mp3/voc/beats: {info['with_mp3']}/{info['with_vocal']}/{info['with_beats']}")
    print(f"  duration:   {format_hms(seconds)}  ({info['hours']:.2f} h)")
    print(f"  method:     {method}")
    if extra:
        bits = ", ".join(f"{k}={v}" for k, v in extra.items() if k != "errors")
        if bits:
            print(f"  {bits}")
    return info


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    ports = parse_ports(args.ports)
    auth = HTTPBasicAuth(args.username, args.password)
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="vds_hours_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, args.workers)

    per_port: Dict[int, Dict[str, Any]] = {}
    grand_sec = 0.0
    grand_tracks = 0
    grand_ok = 0

    for port in ports:
        try:
            index = load_index(args.host, port, auth, work_dir)
        except Exception as exc:
            logger.error("Port %s: cannot load index: %s", port, exc)
            per_port[port] = {"error": str(exc), "seconds": 0.0, "tracks": 0, "parsed": 0}
            continue

        tracks = list(index.tracks.values())
        grand_tracks += len(tracks)

        if args.exact:
            durations: Dict[str, float] = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {
                    pool.submit(
                        fetch_sidecars_duration, args.host, port, auth, t, args.timeout,
                    ): t
                    for t in tracks
                }
                for fut in tqdm(as_completed(futs), total=len(futs), desc=f"port {port}", unit="trk"):
                    track = futs[fut]
                    dur = fut.result()
                    if dur is None:
                        vb = vocal_bytes(track)
                        mb = mp3_bytes(track)
                        if vb > 0:
                            dur = vb * 8.0 / VOCAL_BPS
                        elif mb > 0:
                            dur = mb * 8.0 / 256_000
                    if dur:
                        durations[track.track_path] = dur
            total = sum(durations.values())
            info = summarize_port(
                port, index, total, len(durations), "exact_sidecars", {},
            )
        else:
            total = 0.0
            parsed = 0
            used_voc = 0
            used_mp3 = 0
            missing = 0
            rates = sample_rates(
                args.host, port, auth, tracks, args.sample, workers, args.timeout,
            )
            voc_spb = rates["voc_spb"]
            mp3_spb = rates["mp3_spb"]
            for track in tracks:
                vb = vocal_bytes(track)
                mb = mp3_bytes(track)
                if vb > 0:
                    total += vb * voc_spb
                    parsed += 1
                    used_voc += 1
                elif mb > 0:
                    total += mb * mp3_spb
                    parsed += 1
                    used_mp3 += 1
                else:
                    missing += 1
            voc_kbps = (8.0 / voc_spb) / 1000.0 if voc_spb else 0.0
            mp3_kbps = (8.0 / mp3_spb) / 1000.0 if mp3_spb else 0.0
            info = summarize_port(
                port, index, total, parsed, "calibrated_size",
                {
                    "sample_ok": rates["sample_ok"],
                    "voc_pairs": rates["voc_pairs"],
                    "mp3_pairs": rates["mp3_pairs"],
                    "implied_voc_kbps": round(voc_kbps, 2),
                    "implied_mp3_kbps": round(mp3_kbps, 2),
                    "sample_mean_sec": round(rates["mean_dur"], 2),
                    "from_vocal": used_voc,
                    "from_mp3": used_mp3,
                    "no_audio": missing,
                },
            )

        grand_sec += info["seconds"]
        grand_ok += info["parsed"]
        per_port[port] = info

    print()
    print("=" * 60)
    print(f"Host:    {args.host}")
    print(f"Ports:   {', '.join(str(p) for p in ports)}")
    print(f"Tracks:  {grand_ok}/{grand_tracks} with duration")
    print(f"TOTAL:   {format_hms(grand_sec)}")
    print(f"Hours:   {grand_sec / 3600:.4f}")
    print()
    print("By port:")
    for port in ports:
        info = per_port.get(port) or {}
        if info.get("error"):
            print(f"  {port}: ERROR {info['error']}")
            continue
        print(
            f"  {port}:  {format_hms(info['seconds']):>16s}  "
            f"{info['hours']:8.2f} h  tracks={info['parsed']}/{info['tracks']}"
        )

    if args.json_out:
        report = {
            "host": args.host,
            "ports": ports,
            "exact": bool(args.exact),
            "tracks": grand_tracks,
            "parsed": grand_ok,
            "total_seconds": grand_sec,
            "total_hours": grand_sec / 3600,
            "by_port": {str(k): v for k, v in per_port.items()},
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote %s", args.json_out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
