#!/home/k4/Projects/BirdsMilkDatasetPreprocessing/.venv/bin/python
"""Cut ASR-backed vocal clips from remote Blackbird WebDAV datasets.

1. Prefetch *_voc.opus + *_lyrics.json from WebDAV so downloads run ahead of GPU work.
2. Drop inter-word pauses longer than 3 s; keep shorter gaps. Remap ASR
   word timestamps onto that compact timeline.
3. Per album: CAM++ of the first 5 s of each track; pick the majority voice.
4. Split compact audio by CAM++ vs segment start; skip clips that do not
   match the album voice (cosine > 0.8). Do not delete existing files.
5. Upload mp3 + json to the Hugging Face bucket in the background
   (hf://buckets/Mach1Corp/voice-corpus-internal/blackbird/...).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import requests
import torch
import torchaudio
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

from blackbird.index import DatasetIndex, TrackInfo

logger = logging.getLogger("extract_asr_clips")

DEFAULT_HOST = "188.120.253.126"
DEFAULT_PORTS = [8091, 8092, 8093, 8094, 8095, 8096]
DEFAULT_USER = "blackbird"
DEFAULT_PASS = "dataset"
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CAMPPLUS_CKPT = _SCRIPT_DIR / "checkpoints" / "campplus_cn_common.bin"
SEED_VC_ROOT = _SCRIPT_DIR / "third_party" / "seed-vc"

CAM_HOP = 5.0
CAM_SIM = 0.8
CAM_SR = 16_000
CAM_MIN_WINDOW = 1.0
MAX_PAUSE = 3.0
ENV_FILE = Path(__file__).resolve().parent / ".env"
DEFAULT_HF_BUCKET = "https://huggingface.co/buckets/Mach1Corp/voice-corpus-internal/blackbird"
DEFAULT_HF_PREFIX = ""

SAFE_CHARS = re.compile(r"[^\w.\-]+", re.UNICODE)


# ---------------------------------------------------------------------------
# ASR + pause stripping
# ---------------------------------------------------------------------------

def load_words(lyrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = lyrics.get("words") or []
    words: List[Dict[str, Any]] = []
    for w in raw:
        try:
            start = float(w["start"])
            end = float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        token = w.get("word") or w.get("text") or ""
        token = str(token).strip()
        if not token or end <= start:
            continue
        words.append({"word": token, "start": start, "end": end})
    words.sort(key=lambda x: (x["start"], x["end"]))
    return words


def load_audio_mono(path: Path) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0).contiguous(), int(sr)


def strip_pauses(
    wav: torch.Tensor,
    sr: int,
    words: Sequence[Dict[str, Any]],
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Keep gaps ≤ MAX_PAUSE, cut gaps > MAX_PAUSE. Remap word times."""
    pieces: List[torch.Tensor] = []
    compact: List[Dict[str, Any]] = []
    cursor = 0.0
    last_end = 0
    started = False
    n = wav.numel()
    max_gap = int(round(MAX_PAUSE * sr))

    for w in words:
        i0 = max(0, min(n, int(round(w["start"] * sr))))
        i1 = max(0, min(n, int(round(w["end"] * sr))))
        if i1 <= i0:
            continue

        if not started:
            if 0 < i0 <= max_gap:
                pieces.append(wav[:i0])
                cursor += i0 / sr
            started = True
        else:
            gap = i0 - last_end
            if gap > 0:
                if gap <= max_gap:
                    pieces.append(wav[last_end:i0])
                    cursor += gap / sr
            elif gap < 0:
                i0 = last_end
                if i1 <= i0:
                    continue

        pieces.append(wav[i0:i1])
        dur = (i1 - i0) / sr
        compact.append({
            "word": w["word"],
            "start": cursor,
            "end": cursor + dur,
            "orig_start": w["start"],
            "orig_end": w["end"],
        })
        cursor += dur
        last_end = i1

    if not pieces:
        return wav.new_zeros(0), []
    return torch.cat(pieces), compact


def words_in_range(words: Sequence[Dict[str, Any]], t0: float, t1: float) -> List[Dict[str, Any]]:
    out = []
    for w in words:
        if w["end"] <= t0 or w["start"] >= t1:
            continue
        out.append({
            "word": w["word"],
            "start": round(max(w["start"], t0) - t0, 4),
            "end": round(min(w["end"], t1) - t0, 4),
        })
    return out


def fragment_from_range(
    words: Sequence[Dict[str, Any]],
    t0: float,
    t1: float,
) -> Optional[Dict[str, Any]]:
    rel = words_in_range(words, t0, t1)
    if not rel:
        return None
    return {
        "start": t0,
        "end": t1,
        "duration": t1 - t0,
        "text": " ".join(w["word"] for w in rel),
        "words": rel,
    }


# ---------------------------------------------------------------------------
# CAM++
# ---------------------------------------------------------------------------

class CamPlusEncoder:
    """CAMPPlus speaker embeddings (3D-Speaker / FunASR campplus_cn_common)."""

    def __init__(self, ckpt: Path, device: str):
        if str(SEED_VC_ROOT) not in sys.path:
            sys.path.insert(0, str(SEED_VC_ROOT))
        from modules.campplus.DTDNN import CAMPPlus  # type: ignore

        self.device = torch.device(device)
        self.model = CAMPPlus(feat_dim=80, embedding_size=192)
        state = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def embed(self, wav_16k: torch.Tensor, start: float, end: float) -> torch.Tensor:
        i0 = max(0, int(start * CAM_SR))
        i1 = min(int(wav_16k.numel()), int(end * CAM_SR))
        chunk = wav_16k[i0:i1]
        if chunk.numel() < int(CAM_MIN_WINDOW * CAM_SR):
            pad = int(CAM_MIN_WINDOW * CAM_SR) - chunk.numel()
            chunk = torch.nn.functional.pad(chunk, (0, max(0, pad)))
        feat = torchaudio.compliance.kaldi.fbank(
            chunk.unsqueeze(0),
            num_mel_bins=80,
            dither=0,
            sample_frequency=CAM_SR,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        emb = self.model(feat.unsqueeze(0).to(self.device))
        return torch.nn.functional.normalize(emb.squeeze(0).float(), dim=0).cpu()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum().clamp(-1.0, 1.0))


def pick_album_voice(
    named_embs: Sequence[Tuple[str, torch.Tensor]],
    min_sim: float,
) -> Optional[torch.Tensor]:
    """Choose the first-5s embedding that matches the most other tracks."""
    if not named_embs:
        return None
    n = len(named_embs)
    best_i = 0
    best_key = (-1, -1.0)
    for i, (_, ei) in enumerate(named_embs):
        sims = [cosine_sim(ei, ej) for j, (_, ej) in enumerate(named_embs) if j != i]
        count = sum(1 for s in sims if s > min_sim)
        mean = sum(sims) / len(sims) if sims else 1.0
        if (count, mean) > best_key:
            best_key = (count, mean)
            best_i = i
    name, emb = named_embs[best_i]
    logger.info(
        "Album voice ← %s  (%d/%d tracks sim>%.2f, mean=%.3f)",
        name, best_key[0] + 1, n, min_sim, best_key[1],
    )
    return emb


def resample_16k(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == CAM_SR:
        return wav
    return torchaudio.transforms.Resample(sr, CAM_SR)(wav.unsqueeze(0)).squeeze(0)


def split_by_cam(
    wav_16k: torch.Tensor,
    encoder: CamPlusEncoder,
    hop: float,
    min_sim: float,
) -> List[Tuple[float, float]]:
    """Split compact audio: every `hop` s vs first `hop` s of the current segment.

    A window stays in the segment only if cosine > min_sim.
    """
    duration = wav_16k.numel() / CAM_SR
    if duration <= 0:
        return []

    cache: Dict[float, torch.Tensor] = {}

    def window(t: float) -> torch.Tensor:
        key = round(t, 4)
        if key not in cache:
            cache[key] = encoder.embed(wav_16k, t, min(t + hop, duration))
        return cache[key]

    bounds: List[Tuple[float, float]] = []
    t0 = 0.0
    while t0 < duration - 1e-6:
        remaining = duration - t0
        if remaining < hop:
            bounds.append((t0, duration))
            break
        ref = window(t0)
        t = t0 + hop
        while t + hop <= duration + 1e-6:
            sim = cosine_sim(ref, window(t))
            logger.debug(
                "CAM vs start t0=%.2f win=%.2f–%.2f sim=%.3f",
                t0, t, t + hop, sim,
            )
            if not (sim > min_sim):
                break
            t += hop
        else:
            t = duration
        if t <= t0 + 1e-6:
            t = min(t0 + hop, duration)
        bounds.append((t0, t))
        t0 = t
    return bounds


# ---------------------------------------------------------------------------
# WebDAV / HTTP
# ---------------------------------------------------------------------------

def make_http_session(auth: HTTPBasicAuth) -> requests.Session:
    session = requests.Session()
    session.auth = auth
    session.trust_env = False
    session.headers["User-Agent"] = "extract_asr_clips/1.0"
    return session


_tls = threading.local()


def thread_session(auth: HTTPBasicAuth) -> requests.Session:
    sess = getattr(_tls, "session", None)
    if sess is None:
        sess = make_http_session(auth)
        _tls.session = sess
    return sess


def remote_relpath(symbolic: str) -> str:
    parts = symbolic.split("/", 1)
    return parts[1] if len(parts) == 2 else symbolic


def dav_get(
    session: requests.Session,
    base: str,
    remote_path: str,
    dest: Path,
    timeout: int = 120,
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
    timeout: int = 60,
) -> Dict[str, Any]:
    url = f"{base}/{quote(remote_path.lstrip('/'), safe='/')}"
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def load_remote_index(
    session: requests.Session,
    base: str,
    cache_path: Path,
) -> DatasetIndex:
    dav_get(session, base, ".blackbird/index.pickle", cache_path)
    return DatasetIndex.load(cache_path)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_mp3(wav: torch.Tensor, sr: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    audio = wav.unsqueeze(0).cpu() if wav.dim() == 1 else wav.cpu()
    try:
        torchaudio.save(str(dest), audio, sr, format="mp3")
        return
    except Exception:
        pass
    tmp = dest.with_suffix(".wav")
    torchaudio.save(str(tmp), audio, sr)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp), "-c:a", "libmp3lame", "-q:a", "2", str(dest)],
        capture_output=True, text=True,
    )
    tmp.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:] or "ffmpeg mp3 encode failed")


def safe_name(text: str, max_len: int = 80) -> str:
    cleaned = SAFE_CHARS.sub("_", text).strip("._")
    return (cleaned or "track")[:max_len]


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
    if not ports:
        return list(DEFAULT_PORTS)
    return ports


def parse_hf_bucket_url(url: str) -> Tuple[str, str]:
    """Return (bucket_id, prefix) from org/bucket[/prefix] or a Hub buckets URL."""
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


def _import_hf_buckets():
    """Prefer user-site huggingface_hub (≥1.24) over the venv copy."""
    user_site = Path.home() / ".local/lib/python3.10/site-packages"
    if user_site.is_dir():
        sys.path.insert(0, str(user_site))
        for name in list(sys.modules):
            if name == "huggingface_hub" or name.startswith("huggingface_hub."):
                del sys.modules[name]
    from huggingface_hub import batch_bucket_files, list_bucket_tree
    return batch_bucket_files, list_bucket_tree


class HfBucketUploader:
    """Queue local files and upload them to an HF bucket on a background thread."""

    def __init__(self, bucket_id: str, prefix: str, token: str, workers: int = 2):
        self.bucket_id = bucket_id
        self.prefix = prefix.strip("/")
        self.token = token
        self._batch, self._list_tree = _import_hf_buckets()
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._album_present: Dict[str, bool] = {}
        self._listed_ports: set[int] = set()
        self._fail = 0
        self._ok = 0
        self._threads = []
        for i in range(max(1, workers)):
            t = threading.Thread(target=self._worker, name=f"hf-upload-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        logger.info(
            "HF bucket %s  prefix=%s  background workers=%d",
            self.bucket_id, self.prefix or "/", workers,
        )

    def remote_path(self, *parts: str) -> str:
        chunks = [self.prefix] if self.prefix else []
        chunks.extend(str(p).strip("/") for p in parts if p)
        return "/".join(chunks)

    def album_dir(self, port: int, artist: str, album: str) -> str:
        return self.remote_path(str(port), safe_name(artist), safe_name(album))

    def preload_existing_albums(self, port: int) -> int:
        """One recursive listing of {prefix}/{port}/; cache which album dirs exist."""
        if port in self._listed_ports:
            return sum(1 for k, v in self._album_present.items() if v and k.startswith(self.remote_path(str(port)) + "/"))
        root = self.remote_path(str(port))
        album_depth = len(root.split("/")) + 2  # port / artist / album
        found = 0
        logger.info("HF: listing existing albums under %s ...", root)
        try:
            for item in self._list_tree(
                self.bucket_id,
                prefix=root,
                recursive=True,
                token=self.token,
            ):
                path = (getattr(item, "path", None) or "").strip("/")
                if not path:
                    continue
                parts = path.split("/")
                if len(parts) < album_depth:
                    continue
                album_prefix = "/".join(parts[:album_depth])
                if not self._album_present.get(album_prefix):
                    self._album_present[album_prefix] = True
                    found += 1
        except Exception as exc:
            logger.warning("HF list failed for %s: %s — will treat albums as missing", root, exc)
            return 0
        self._listed_ports.add(port)
        logger.info("HF: %d albums already present under %s", found, root)
        return found

    def album_exists(self, port: int, artist: str, album: str) -> bool:
        """True if this album folder already has any files in the bucket."""
        prefix = self.album_dir(port, artist, album)
        if prefix in self._album_present:
            return self._album_present[prefix]
        if port in self._listed_ports:
            return False
        exists = False
        try:
            for item in self._list_tree(
                self.bucket_id,
                prefix=prefix,
                recursive=False,
                token=self.token,
            ):
                if getattr(item, "path", None):
                    exists = True
                    break
        except Exception as exc:
            logger.warning("HF list failed for %s: %s", prefix, exc)
        self._album_present[prefix] = exists
        return exists

    def submit(self, local_mp3: Path, local_json: Path, remote_mp3: str, remote_json: str) -> None:
        self._q.put((local_mp3, local_json, remote_mp3, remote_json))

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                local_mp3, local_json, remote_mp3, remote_json = item
                try:
                    self._batch(
                        self.bucket_id,
                        add=[
                            (str(local_mp3), remote_mp3),
                            (str(local_json), remote_json),
                        ],
                        token=self.token,
                    )
                    self._ok += 1
                    logger.info("uploaded %s", remote_mp3)
                except Exception as exc:
                    self._fail += 1
                    logger.error("HF upload failed %s: %s", remote_mp3, exc)
                finally:
                    Path(local_mp3).unlink(missing_ok=True)
                    Path(local_json).unlink(missing_ok=True)
            finally:
                self._q.task_done()

    def close(self) -> None:
        logger.info("Waiting for HF uploads (%d queued) ...", self._q.qsize())
        self._q.join()
        for _ in self._threads:
            self._q.put(None)
        for t in self._threads:
            t.join(timeout=120)
        logger.info("HF uploads done: ok=%d fail=%d", self._ok, self._fail)


# ---------------------------------------------------------------------------
# Per-port processing
# ---------------------------------------------------------------------------

@dataclass
class DownloadedTrack:
    track: TrackInfo
    lyrics_remote: str
    vocal_remote: str
    lyrics: Optional[Dict[str, Any]] = None
    vocal_path: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class PreparedTrack:
    track: TrackInfo
    album: str
    compact: torch.Tensor
    compact_16k: torch.Tensor
    sr: int
    compact_words: List[Dict[str, Any]]
    intro_emb: torch.Tensor
    lyrics_remote: str
    vocal_remote: str


def tracks_with_asr(index: DatasetIndex) -> List[TrackInfo]:
    out = []
    for track in index.tracks.values():
        if "lyrics" in track.files and "vocal" in track.files:
            out.append(track)
    out.sort(key=lambda t: t.track_path)
    return out


def group_albums(tracks: Sequence[TrackInfo]) -> List[Tuple[Tuple[str, str], List[TrackInfo]]]:
    grouped: Dict[Tuple[str, str], List[TrackInfo]] = defaultdict(list)
    for track in tracks:
        grouped[(track.artist, track.album_path)].append(track)
    return sorted(grouped.items(), key=lambda kv: kv[0])


def download_track(
    base: str,
    auth: HTTPBasicAuth,
    port: int,
    track: TrackInfo,
    work_dir: Path,
) -> DownloadedTrack:
    lyrics_remote = remote_relpath(track.files["lyrics"])
    vocal_remote = remote_relpath(track.files["vocal"])
    out = DownloadedTrack(
        track=track,
        lyrics_remote=lyrics_remote,
        vocal_remote=vocal_remote,
    )
    session = thread_session(auth)
    try:
        out.lyrics = dav_get_json(session, base, lyrics_remote)
    except Exception as exc:
        out.error = f"lyrics: {exc}"
        logger.warning("[%s] lyrics download failed: %s", track.base_name, exc)
        return out
    if not load_words(out.lyrics):
        out.error = "no words"
        return out
    dest_dir = work_dir / "prefetch" / str(port) / uuid.uuid4().hex
    dest_dir.mkdir(parents=True, exist_ok=True)
    vocal_local = dest_dir / f"{safe_name(track.base_name)}.opus"
    try:
        dav_get(session, base, vocal_remote, vocal_local, timeout=600)
        out.vocal_path = vocal_local
    except Exception as exc:
        out.error = f"vocal: {exc}"
        logger.warning("[%s] vocal download failed: %s", track.base_name, exc)
        vocal_local.unlink(missing_ok=True)
    return out


def assemble_prepared(
    downloaded: DownloadedTrack,
    cam: CamPlusEncoder,
    cam_hop: float,
) -> Optional[PreparedTrack]:
    if downloaded.error or downloaded.lyrics is None or downloaded.vocal_path is None:
        return None
    words = load_words(downloaded.lyrics)
    if not words:
        return None
    try:
        wav, sr = load_audio_mono(downloaded.vocal_path)
    except Exception as exc:
        logger.warning("[%s] vocal load failed: %s", downloaded.track.base_name, exc)
        return None
    compact, compact_words = strip_pauses(wav, sr, words)
    if compact.numel() == 0 or not compact_words:
        return None
    compact_16k = resample_16k(compact, sr)
    dur = compact_16k.numel() / CAM_SR
    if dur < CAM_MIN_WINDOW:
        return None
    intro_emb = cam.embed(compact_16k, 0.0, min(cam_hop, dur))
    album = downloaded.track.album_path.split("/")[-1]
    return PreparedTrack(
        track=downloaded.track,
        album=album,
        compact=compact,
        compact_16k=compact_16k,
        sr=sr,
        compact_words=compact_words,
        intro_emb=intro_emb,
        lyrics_remote=downloaded.lyrics_remote,
        vocal_remote=downloaded.vocal_remote,
    )


def cleanup_download(downloaded: DownloadedTrack) -> None:
    if downloaded.vocal_path is None:
        return
    downloaded.vocal_path.unlink(missing_ok=True)
    parent = downloaded.vocal_path.parent
    try:
        parent.rmdir()
    except OSError:
        pass


class AlbumPrefetch:
    """Download upcoming albums in the background; processing pulls ready bundles."""

    def __init__(
        self,
        base: str,
        auth: HTTPBasicAuth,
        port: int,
        work_dir: Path,
        albums: Sequence[Tuple[Tuple[str, str], List[TrackInfo]]],
        download_workers: int,
        prefetch_albums: int,
    ):
        self._base = base
        self._auth = auth
        self._port = port
        self._work_dir = work_dir
        self._albums = list(albums)
        self._workers = max(1, download_workers)
        self._ready: queue.Queue = queue.Queue(maxsize=max(1, prefetch_albums))
        self._thread = threading.Thread(target=self._producer, name="dav-prefetch", daemon=True)
        self._thread.start()
        logger.info(
            "Prefetch: %d download workers, %d albums ahead",
            self._workers, max(1, prefetch_albums),
        )

    def _producer(self) -> None:
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            for key, album_tracks in self._albums:
                futs = [
                    pool.submit(
                        download_track, self._base, self._auth,
                        self._port, track, self._work_dir,
                    )
                    for track in album_tracks
                ]
                items = [f.result() for f in futs]
                self._ready.put((key, items))
        self._ready.put(None)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[Tuple[str, str], List[DownloadedTrack]]:
        item = self._ready.get()
        if item is None:
            raise StopIteration
        return item


def emit_track(
    prepared: PreparedTrack,
    host: str,
    port: int,
    work_dir: Path,
    dry_run: bool,
    clip_quota: Optional[int],
    cam: CamPlusEncoder,
    cam_hop: float,
    cam_sim: float,
    album_voice: torch.Tensor,
    uploader: Optional[HfBucketUploader],
) -> Tuple[int, int, int]:
    track = prepared.track
    bounds = split_by_cam(prepared.compact_16k, cam, cam_hop, cam_sim)
    artist = safe_name(track.artist)
    album = safe_name(prepared.album)
    stem = safe_name(track.base_name)
    kept = 0
    skipped_voice = 0
    skipped_exist = 0
    idx = 0

    for t0, t1 in bounds:
        frag = fragment_from_range(prepared.compact_words, t0, t1)
        if frag is None:
            continue
        clip_emb = cam.embed(prepared.compact_16k, t0, min(t0 + cam_hop, t1))
        sim = cosine_sim(clip_emb, album_voice)
        if not (sim > cam_sim):
            skipped_voice += 1
            logger.debug(
                "SKIP voice %s %.2f–%.2f sim=%.3f",
                track.base_name, t0, t1, sim,
            )
            continue

        idx += 1
        if clip_quota is not None and kept >= clip_quota:
            break
        rel_mp3 = f"{port}/{artist}/{album}/{stem}_{idx:03d}.mp3"
        rel_json = f"{port}/{artist}/{album}/{stem}_{idx:03d}.json"
        remote_mp3 = uploader.remote_path(rel_mp3) if uploader else rel_mp3
        remote_json = uploader.remote_path(rel_json) if uploader else rel_json
        if dry_run:
            logger.info(
                "DRY %s  %.2f–%.2f (%.1fs, album_sim=%.3f)  %s",
                track.base_name, frag["start"], frag["end"],
                frag["duration"], sim, frag["text"][:80],
            )
            kept += 1
            continue
        payload = {
            "text": frag["text"],
            "words": frag["words"],
            "source": {
                "host": host,
                "port": port,
                "artist": track.artist,
                "album": prepared.album,
                "track": track.base_name,
                "vocal": prepared.vocal_remote,
                "lyrics": prepared.lyrics_remote,
                "compact_start": round(frag["start"], 4),
                "compact_end": round(frag["end"], 4),
                "duration": round(frag["duration"], 4),
                "cam_hop": cam_hop,
                "cam_sim": cam_sim,
                "album_voice_sim": round(sim, 4),
                "hf_path": remote_mp3,
            },
        }
        if uploader is None:
            continue
        staging = work_dir / "hf_staging" / uuid.uuid4().hex
        staging.mkdir(parents=True, exist_ok=True)
        local_mp3 = staging / f"{stem}_{idx:03d}.mp3"
        local_json = staging / f"{stem}_{idx:03d}.json"
        i0 = int(round(frag["start"] * prepared.sr))
        i1 = int(round(frag["end"] * prepared.sr))
        try:
            save_mp3(prepared.compact[i0:i1], prepared.sr, local_mp3)
            local_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("[%s] encode failed clip %d: %s", track.base_name, idx, exc)
            local_mp3.unlink(missing_ok=True)
            local_json.unlink(missing_ok=True)
            continue
        uploader.submit(local_mp3, local_json, remote_mp3, remote_json)
        kept += 1

    logger.info(
        "[%s] speech=%.1fs  keep=%d skip_voice=%d",
        track.base_name, prepared.compact.numel() / prepared.sr, kept, skipped_voice,
    )
    return kept, skipped_voice, skipped_exist


def process_album(
    host: str,
    port: int,
    downloaded: Sequence[DownloadedTrack],
    work_dir: Path,
    dry_run: bool,
    clip_quota: Optional[int],
    cam: CamPlusEncoder,
    cam_hop: float,
    cam_sim: float,
    uploader: Optional[HfBucketUploader],
) -> Tuple[int, int, int]:
    prepared: List[PreparedTrack] = []
    try:
        for item in downloaded:
            built = assemble_prepared(item, cam, cam_hop)
            if built is not None:
                prepared.append(built)
    finally:
        for item in downloaded:
            cleanup_download(item)

    if not prepared:
        return 0, 0, 0

    album_name = prepared[0].album
    artist = prepared[0].track.artist
    logger.info("Album %s / %s  — %d tracks with speech", artist, album_name, len(prepared))
    album_voice = pick_album_voice(
        [(p.track.base_name, p.intro_emb) for p in prepared],
        cam_sim,
    )
    if album_voice is None:
        return 0, 0, 0

    kept = skipped_voice = skipped_exist = 0
    for item in prepared:
        quota = None if clip_quota is None else max(0, clip_quota - kept)
        if quota == 0:
            break
        k, sv, se = emit_track(
            item, host, port, work_dir, dry_run, quota,
            cam, cam_hop, cam_sim, album_voice, uploader,
        )
        kept += k
        skipped_voice += sv
        skipped_exist += se
    return kept, skipped_voice, skipped_exist


def process_port(
    host: str,
    port: int,
    auth: HTTPBasicAuth,
    work_dir: Path,
    dry_run: bool,
    max_tracks: Optional[int],
    limit: Optional[int],
    cam: CamPlusEncoder,
    cam_hop: float,
    cam_sim: float,
    uploader: Optional[HfBucketUploader],
    download_workers: int,
    prefetch_albums: int,
) -> None:
    base = f"http://{host}:{port}"
    session = make_http_session(auth)

    cache = work_dir / f"index_{port}.pickle"
    logger.info("Port %s: downloading index ...", port)
    try:
        index = load_remote_index(session, base, cache)
    except Exception as exc:
        logger.error("Port %s: cannot load index: %s", port, exc)
        return
    session.close()

    tracks = tracks_with_asr(index)
    if max_tracks is not None:
        tracks = tracks[:max_tracks]
    albums = group_albums(tracks)
    logger.info(
        "Port %s: %d tracks with lyrics+vocal in %d albums",
        port, len(tracks), len(albums),
    )

    if uploader is not None:
        uploader.preload_existing_albums(port)

    pending: List[Tuple[Tuple[str, str], List[TrackInfo]]] = []
    skipped_albums = 0
    for (_artist, _album_path), album_tracks in albums:
        sample = album_tracks[0]
        if uploader is not None and uploader.album_exists(
            port, sample.artist, sample.album_path.split("/")[-1],
        ):
            skipped_albums += 1
            logger.debug(
                "SKIP album already on HF: %s / %s",
                sample.artist, sample.album_path.split("/")[-1],
            )
            continue
        pending.append(((_artist, _album_path), album_tracks))
    logger.info(
        "Port %s: %d albums to process, %d already on HF",
        port, len(pending), skipped_albums,
    )

    written = 0
    skipped_voice = 0
    skipped_exist = 0
    used_albums = 0
    if not pending:
        logger.info(
            "Port %s done: nothing to download (%d albums already on HF)",
            port, skipped_albums,
        )
        return

    prefetch = AlbumPrefetch(
        base, auth, port, work_dir, pending,
        download_workers=download_workers,
        prefetch_albums=prefetch_albums,
    )
    pbar = tqdm(total=len(pending), desc=f":{port}", unit="album")
    for (_artist, _album_path), downloaded in prefetch:
        quota = None if limit is None else max(0, limit - written)
        if quota == 0:
            break
        n, sv, se = process_album(
            host, port, downloaded,
            work_dir, dry_run, quota,
            cam, cam_hop, cam_sim, uploader,
        )
        written += n
        skipped_voice += sv
        skipped_exist += se
        if n:
            used_albums += 1
        pbar.update(1)
        pbar.set_postfix(
            clips=written, skip_v=skipped_voice,
            albums=used_albums, skip_alb=skipped_albums,
        )
        if limit is not None and written >= limit:
            logger.info("Reached --limit %d clips", limit)
            break
    for leftover in prefetch:
        for item in leftover[1]:
            cleanup_download(item)
    pbar.close()
    logger.info(
        "Port %s done: %d new clips, %d skipped (other voice), "
        "%d albums uploaded, %d albums already on HF",
        port, written, skipped_voice, used_albums, skipped_albums,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strip vocal pauses, remap ASR, split by CAM++, upload clips to HF bucket")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument(
        "--ports", nargs="*", default=None, metavar="PORT",
        help="Server ports to process (default: all %s). "
             "Example: --ports 8091 8093" % ",".join(str(p) for p in DEFAULT_PORTS),
    )
    p.add_argument("--username", default=DEFAULT_USER)
    p.add_argument("--password", default=DEFAULT_PASS)
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/extract_asr_clips"))
    p.add_argument("--env-file", type=Path, default=ENV_FILE)
    p.add_argument("--hf-bucket", default=DEFAULT_HF_BUCKET,
                   help="Bucket id or Hub URL")
    p.add_argument("--hf-prefix", default=DEFAULT_HF_PREFIX,
                   help="Extra path prefix inside the bucket")
    p.add_argument("--hf-workers", type=int, default=2,
                   help="Background HF bucket upload threads (default: 2)")
    p.add_argument("--download-workers", type=int, default=4,
                   help="Parallel WebDAV download threads (default: 4)")
    p.add_argument("--prefetch-albums", type=int, default=2,
                   help="How many albums to download ahead of processing (default: 2)")
    p.add_argument("--dry-run", action="store_true",
                   help="Strip pauses + CAM++ split, do not upload")
    p.add_argument("--max-tracks", type=int, default=None,
                   help="Cap tracks per port (debug)")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many new clips (across a port)")
    p.add_argument("--cam-ckpt", type=Path, default=DEFAULT_CAMPPLUS_CKPT)
    p.add_argument("--cam-hop", type=float, default=CAM_HOP,
                   help="Window length / hop in seconds (default: 5)")
    p.add_argument("--cam-sim", type=float, default=CAM_SIM,
                   help="Require cosine > this vs first 5s of the segment (default: 0.8)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and not args.dry_run:
        raise SystemExit(f"HF_TOKEN not found in environment or {args.env_file}")

    bucket_id, url_prefix = parse_hf_bucket_url(args.hf_bucket)
    prefix = args.hf_prefix.strip("/")
    if url_prefix:
        prefix = "/".join(p for p in (url_prefix, prefix) if p)

    ports = parse_ports(args.ports)
    logger.info("Ports: %s", ", ".join(str(p) for p in ports))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    auth = HTTPBasicAuth(args.username, args.password)

    if not args.cam_ckpt.is_file():
        raise SystemExit(f"CAM++ checkpoint not found: {args.cam_ckpt}")
    logger.info("Loading CAM++ from %s on %s", args.cam_ckpt, args.device)
    cam = CamPlusEncoder(args.cam_ckpt, args.device)

    uploader = None
    if token:
        uploader = HfBucketUploader(
            bucket_id, prefix, token, workers=args.hf_workers,
        )
    elif not args.dry_run:
        raise SystemExit(f"HF_TOKEN not found in environment or {args.env_file}")
    try:
        for port in ports:
            process_port(
                host=args.host,
                port=port,
                auth=auth,
                work_dir=args.work_dir,
                dry_run=args.dry_run,
                max_tracks=args.max_tracks,
                limit=args.limit,
                cam=cam,
                cam_hop=args.cam_hop,
                cam_sim=args.cam_sim,
                uploader=uploader,
                download_workers=args.download_workers,
                prefetch_albums=args.prefetch_albums,
            )
    finally:
        if uploader is not None:
            uploader.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
