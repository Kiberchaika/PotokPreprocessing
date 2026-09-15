#!/usr/bin/env python3
"""Sum uploaded clip duration in the HF bucket without downloading MP3s.

Each clip from extract_asr_clips.py is stored as a pair:
  {port}/{artist}/{album}/{stem}_{idx}.mp3
  {port}/{artist}/{album}/{stem}_{idx}.json

JSON sidecars already contain source.duration. This script lists the bucket
and HTTP Range-reads only the tail of each JSON (a few KB), never the MP3s.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

logger = logging.getLogger("count_hf_bucket_hours")

ENV_FILE = Path(__file__).resolve().parent / ".env"
DEFAULT_HF_BUCKET = "https://huggingface.co/buckets/Mach1Corp/voice-corpus-internal/blackbird"
DEFAULT_HF_PREFIX = ""
HF_ENDPOINT = "https://huggingface.co"
DURATION_RE = re.compile(r'"duration"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
_tls = threading.local()


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def parse_hf_bucket_url(url: str) -> Tuple[str, str]:
    text = url.strip().rstrip("/")
    for prefix in (
        "https://huggingface.co/buckets/",
        "http://huggingface.co/buckets/",
        "hf://buckets/",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    parts = [p for p in text.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Bad HF bucket id: {url}")
    bucket_id = f"{parts[0]}/{parts[1]}"
    extra = "/".join(parts[2:])
    return bucket_id, extra


def _import_list_bucket_tree():
    user_site = Path.home() / ".local/lib/python3.10/site-packages"
    if user_site.is_dir():
        sys.path.insert(0, str(user_site))
        for name in list(sys.modules):
            if name == "huggingface_hub" or name.startswith("huggingface_hub."):
                del sys.modules[name]
    from huggingface_hub import list_bucket_tree
    return list_bucket_tree


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


def duration_from_text(text: str) -> Optional[float]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        src = payload.get("source") or {}
        if isinstance(src, dict) and src.get("duration") is not None:
            try:
                dur = float(src["duration"])
                if dur > 0:
                    return dur
            except (TypeError, ValueError):
                pass
    matches = DURATION_RE.findall(text)
    if matches:
        try:
            dur = float(matches[-1])
            if dur > 0:
                return dur
        except ValueError:
            return None
    return None


def group_key(path: str, depth: int) -> str:
    parts = path.split("/")
    if depth <= 0 or len(parts) <= depth:
        return path.rsplit("/", 1)[0] if "/" in path else "(root)"
    return "/".join(parts[:depth])


def print_breakdown(title: str, totals: Dict[str, float], counts: Dict[str, int]) -> None:
    if not totals:
        return
    print()
    print(title)
    keys = sorted(totals, key=lambda k: totals[k], reverse=True)
    for key in keys:
        print(
            f"  {key:40s}  {format_hms(totals[key]):>16s}  "
            f"{totals[key] / 3600:8.2f} h  clips={counts[key]}"
        )


class RateGate:
    """Global request pacing plus a shared cooldown after HTTP 429."""

    def __init__(self, rps: float):
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self.lock = threading.Lock()
        self.next_ok = 0.0
        self.cooldown_until = 0.0

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                ready = max(self.next_ok, self.cooldown_until)
                delay = ready - now
                if delay <= 0:
                    self.next_ok = now + self.min_interval
                    return
            time.sleep(min(delay, 0.2))

    def on_429(self, retry_after: float) -> None:
        wait = max(1.0, retry_after)
        with self.lock:
            until = time.monotonic() + wait
            if until > self.cooldown_until:
                self.cooldown_until = until
                logger.warning("HTTP 429 — pause all workers for %.1fs", wait)


def parse_retry_after(resp: requests.Response, default: float) -> float:
    raw = resp.headers.get("Retry-After")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def thread_session() -> requests.Session:
    sess = getattr(_tls, "session", None)
    if sess is None:
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.5,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
        sess = requests.Session()
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _tls.session = sess
    return sess


def fetch_duration(
    bucket_id: str,
    remote_path: str,
    token: str,
    timeout: float,
    tail_bytes: int,
    gate: RateGate,
    max_attempts: int,
    retry_after_default: float,
) -> Tuple[str, Optional[float], Optional[str]]:
    url = f"{HF_ENDPOINT}/buckets/{bucket_id}/resolve/{quote(remote_path, safe='/')}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Range": f"bytes=-{max(512, tail_bytes)}",
    }
    last_err = "unknown"
    for attempt in range(1, max_attempts + 1):
        gate.wait()
        try:
            resp = thread_session().get(url, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                last_err = "429 Too Many Requests"
                gate.on_429(parse_retry_after(resp, retry_after_default * attempt))
                continue
            resp.raise_for_status()
            dur = duration_from_text(resp.text)
            if dur is None:
                return remote_path, None, "no duration in JSON tail"
            return remote_path, dur, None
        except Exception as exc:
            last_err = str(exc)
            time.sleep(min(30.0, retry_after_default * attempt))
    return remote_path, None, last_err


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Count hours of MP3 in the HF bucket using JSON sidecars (no MP3 download)",
    )
    p.add_argument("--env-file", type=Path, default=ENV_FILE)
    p.add_argument("--hf-bucket", default=DEFAULT_HF_BUCKET)
    p.add_argument("--hf-prefix", default=DEFAULT_HF_PREFIX)
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel JSON Range requests (default: 8)")
    p.add_argument("--rps", type=float, default=10.0,
                   help="Global request rate limit (default: 10/s)")
    p.add_argument("--max-attempts", type=int, default=12,
                   help="Retries per file after 429/errors (default: 12)")
    p.add_argument("--retry-after", type=float, default=20.0,
                   help="Default cooldown seconds on 429 if Retry-After is missing")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument(
        "--tail-bytes", type=int, default=4096,
        help="How many bytes to read from the end of each JSON (default: 4096)",
    )
    p.add_argument(
        "--group-depth", type=int, default=2,
        help="Path prefix depth for breakdown (2=blackbird/port)",
    )
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    load_env_file(args.env_file)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit(f"HF_TOKEN not found in environment or {args.env_file}")

    bucket_id, url_prefix = parse_hf_bucket_url(args.hf_bucket)
    prefix = args.hf_prefix.strip("/")
    if url_prefix:
        prefix = "/".join(p for p in (url_prefix, prefix) if p)

    list_tree = _import_list_bucket_tree()
    logger.info("Listing bucket %s prefix=%s ...", bucket_id, prefix or "/")

    mp3: Dict[str, int] = {}
    json_files: Dict[str, int] = {}
    for item in list_tree(
        bucket_id,
        prefix=prefix or None,
        recursive=True,
        token=token,
    ):
        if getattr(item, "type", None) != "file":
            continue
        path = (getattr(item, "path", None) or "").strip("/")
        if not path:
            continue
        size = int(getattr(item, "size", 0) or 0)
        lower = path.lower()
        if lower.endswith(".mp3"):
            mp3[path] = size
        elif lower.endswith(".json"):
            json_files[path] = size

    logger.info("Listed %d mp3, %d json", len(mp3), len(json_files))

    paired: List[str] = []
    mp3_without_json = 0
    json_without_mp3 = 0
    for mp3_path in mp3:
        sidecar = mp3_path[:-4] + ".json"
        if sidecar in json_files:
            paired.append(sidecar)
        else:
            mp3_without_json += 1
    for js_path in json_files:
        if js_path[:-5] + ".mp3" not in mp3:
            json_without_mp3 += 1

    durations: Dict[str, float] = {}
    errors: List[Tuple[str, str]] = []
    workers = max(1, args.workers)
    gate = RateGate(args.rps)
    logger.info(
        "Reading JSON tails (%d bytes, workers=%d, rps=%.1f) ...",
        args.tail_bytes, workers, args.rps,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [
            pool.submit(
                fetch_duration, bucket_id, path, token, args.timeout,
                args.tail_bytes, gate, args.max_attempts, args.retry_after,
            )
            for path in paired
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="JSON tails", unit="file"):
            path, dur, err = fut.result()
            if dur is not None:
                durations[path] = dur
            else:
                errors.append((path, err or "unknown"))

    total_sec = sum(durations.values())
    mp3_bytes = sum(mp3.values())
    json_bytes = sum(json_files.values())

    by_group: Dict[str, float] = defaultdict(float)
    by_group_n: Dict[str, int] = defaultdict(int)
    for js_path, dur in durations.items():
        key = group_key(js_path, args.group_depth)
        by_group[key] += dur
        by_group_n[key] += 1

    print()
    print(f"Bucket:  {bucket_id}")
    print(f"Prefix:  {prefix or '/'}")
    print(f"MP3 files:           {len(mp3)}")
    print(f"JSON sidecars:       {len(json_files)}")
    print(f"Paired clips:        {len(paired)}")
    print(f"Duration parsed:     {len(durations)}")
    print(f"MP3 without JSON:    {mp3_without_json}")
    print(f"JSON without MP3:    {json_without_mp3}")
    print(f"JSON parse errors:   {len(errors)}")
    print(f"MP3 size:            {format_bytes(mp3_bytes)} ({mp3_bytes} bytes)")
    print(f"JSON size:           {format_bytes(json_bytes)} ({json_bytes} bytes)")
    print()
    print(f"Total duration:      {format_hms(total_sec)}")
    print(f"Total hours:         {total_sec / 3600:.4f}")
    if durations:
        print(f"Mean clip length:    {total_sec / len(durations):.2f} s")
    if errors and durations:
        known_bytes = 0
        for js_path in durations:
            mp3_path = js_path[:-5] + ".mp3"
            known_bytes += mp3.get(mp3_path, 0)
        missing_bytes = 0
        for js_path, _ in errors:
            mp3_path = js_path[:-5] + ".mp3"
            missing_bytes += mp3.get(mp3_path, 0)
        if known_bytes > 0 and missing_bytes > 0:
            bps = known_bytes / total_sec
            extra = missing_bytes / bps
            print(
                f"Est. missing clips:  {format_hms(extra)}  "
                f"(from MP3 size of {len(errors)} unread JSON tails)"
            )
            print(f"Est. total hours:    {(total_sec + extra) / 3600:.4f}")

    print_breakdown(f"By path prefix (depth={args.group_depth}):", dict(by_group), dict(by_group_n))

    if errors:
        print()
        print(f"First errors ({min(10, len(errors))} of {len(errors)}):")
        for path, err in errors[:10]:
            print(f"  {path}: {err}")

    if args.json_out:
        report = {
            "bucket_id": bucket_id,
            "prefix": prefix,
            "mp3_files": len(mp3),
            "json_files": len(json_files),
            "paired": len(paired),
            "duration_parsed": len(durations),
            "mp3_without_json": mp3_without_json,
            "json_without_mp3": json_without_mp3,
            "errors": len(errors),
            "mp3_bytes": mp3_bytes,
            "json_bytes": json_bytes,
            "total_seconds": total_sec,
            "total_hours": total_sec / 3600,
            "by_group": {
                k: {"seconds": by_group[k], "hours": by_group[k] / 3600, "clips": by_group_n[k]}
                for k in sorted(by_group)
            },
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote %s", args.json_out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
