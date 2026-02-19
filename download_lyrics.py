#!/usr/bin/env python3
"""
Lyrics downloader for music albums.
Downloads lyrics for each audio file (MP3, M4A) and saves as .txt with the same name.
"""

import os
import re
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import mutagen


AUDIO_EXTENSIONS = ('.mp3', '.m4a')


def get_metadata_from_file(audio_path: str) -> tuple[str | None, str | None]:
    """Extract artist and title from audio file tags."""
    try:
        audio = mutagen.File(audio_path, easy=True)
        if audio is None:
            return None, None
        artist = audio.get('artist', [None])[0]
        title = audio.get('title', [None])[0]
        return artist, title
    except Exception:
        return None, None


def parse_filename(mp3_path: str) -> tuple[str, str]:
    """Parse artist and title from filename and directory structure."""
    path = Path(mp3_path)
    filename = path.stem

    # Remove track number prefix (e.g., "01. ", "1 - ", "01 ", etc.)
    title = re.sub(r'^(\d+[\.\-\s]+)', '', filename).strip()

    # Try to get artist from parent directories
    # Expected structure: "Artist - Year - Album/..."
    parts = path.parts
    artist = None
    for part in parts:
        match = re.match(r'^(.+?)\s*-\s*\d{4}\s*-', part)
        if match:
            artist = match.group(1).strip()
            break

    return artist or "Unknown Artist", title


def fetch_lyrics_genius(artist: str, title: str, api_token: str) -> str | None:
    """Fetch lyrics from Genius API."""
    headers = {"Authorization": f"Bearer {api_token}"}
    search_url = "https://api.genius.com/search"

    query = f"{artist} {title}"
    params = {"q": query}

    try:
        response = requests.get(search_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        hits = data.get("response", {}).get("hits", [])
        if not hits:
            return None

        # Get the first result's URL
        song_url = hits[0]["result"]["url"]

        # Scrape lyrics from the page
        return scrape_genius_lyrics(song_url)
    except Exception as e:
        print(f"  Genius API error: {e}")
        return None


def scrape_genius_lyrics(url: str) -> str | None:
    """Scrape lyrics from Genius page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find lyrics containers
        lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
        if lyrics_divs:
            lyrics_parts = []
            for div in lyrics_divs:
                # Replace <br> with newlines
                for br in div.find_all('br'):
                    br.replace_with('\n')
                lyrics_parts.append(div.get_text())
            return '\n'.join(lyrics_parts).strip()

        return None
    except Exception as e:
        print(f"  Scraping error: {e}")
        return None


def fetch_lyrics_lrclib(artist: str, title: str) -> str | None:
    """Fetch lyrics from LRCLIB (free, no API key required)."""
    url = "https://lrclib.net/api/get"
    params = {
        "artist_name": artist,
        "track_name": title
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Prefer plain lyrics over synced
            return data.get("plainLyrics") or data.get("syncedLyrics")
        return None
    except Exception:
        return None


def fetch_lyrics_ovh(artist: str, title: str) -> str | None:
    """Fetch lyrics from lyrics.ovh API (free, no API key)."""
    url = f"https://api.lyrics.ovh/v1/{requests.utils.quote(artist)}/{requests.utils.quote(title)}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("lyrics")
        return None
    except Exception:
        return None


def fetch_lyrics(artist: str, title: str, genius_token: str | None = None) -> str | None:
    """Try multiple sources to fetch lyrics."""
    # Clean up artist and title
    artist_clean = re.sub(r'\s*\([^)]*\)', '', artist).strip()
    title_clean = re.sub(r'\s*\([^)]*\)', '', title).strip()
    title_clean = re.sub(r'\s*\[[^\]]*\]', '', title_clean).strip()

    # Try LRCLIB first (free, good for various languages)
    lyrics = fetch_lyrics_lrclib(artist_clean, title_clean)
    if lyrics:
        return lyrics

    # Try lyrics.ovh
    lyrics = fetch_lyrics_ovh(artist_clean, title_clean)
    if lyrics:
        return lyrics

    # Try Genius if token provided
    if genius_token:
        lyrics = fetch_lyrics_genius(artist_clean, title_clean, genius_token)
        if lyrics:
            return lyrics

    return None


def process_mp3(mp3_path: str, genius_token: str | None = None, overwrite: bool = False) -> tuple[str, bool, str]:
    """Process a single MP3 file and download its lyrics."""
    txt_path = Path(mp3_path).with_suffix('.txt')

    # Skip if lyrics already exist
    if txt_path.exists() and not overwrite:
        return mp3_path, True, "already exists"

    # Get metadata
    artist, title = get_metadata_from_file(mp3_path)

    # Fall back to filename parsing
    if not artist or not title:
        artist_fb, title_fb = parse_filename(mp3_path)
        artist = artist or artist_fb
        title = title or title_fb

    # Fetch lyrics
    lyrics = fetch_lyrics(artist, title, genius_token)

    if lyrics:
        txt_path.write_text(lyrics, encoding='utf-8')
        return mp3_path, True, f"saved ({artist} - {title})"
    else:
        return mp3_path, False, f"not found ({artist} - {title})"


def find_audio_files(directory: str) -> list[str]:
    """Find all audio files (MP3, M4A) in directory recursively."""
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(AUDIO_EXTENSIONS):
                audio_files.append(os.path.join(root, file))
    return sorted(audio_files)


def main():
    parser = argparse.ArgumentParser(description="Download lyrics for audio files (MP3, M4A)")
    parser.add_argument(
        "directory",
        nargs="?",
        default="/media/k4_nas/disk1/Datasets/Music_ASR_Test/",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--genius-token",
        help="Genius API token for better results"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing lyrics files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests (seconds)"
    )
    args = parser.parse_args()

    print(f"Scanning {args.directory} for audio files (MP3, M4A)...")
    audio_files = find_audio_files(args.directory)
    print(f"Found {len(audio_files)} audio files\n")

    if not audio_files:
        return

    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, audio_path in enumerate(audio_files, 1):
        rel_path = os.path.relpath(audio_path, args.directory)
        print(f"[{i}/{len(audio_files)}] {rel_path}")

        path, success, message = process_mp3(audio_path, args.genius_token, args.overwrite)

        if "already exists" in message:
            skip_count += 1
            print(f"  ⏭ Skipped: {message}")
        elif success:
            success_count += 1
            print(f"  ✓ {message}")
        else:
            fail_count += 1
            print(f"  ✗ {message}")

        # Rate limiting
        if i < len(audio_files):
            time.sleep(args.delay)

    print(f"\n{'='*50}")
    print(f"Done! Success: {success_count}, Failed: {fail_count}, Skipped: {skip_count}")


if __name__ == "__main__":
    main()
