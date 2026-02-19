'''

rsync -avz --progress --files-from=filelist_part1.01.txt -e "ssh -i dev-233158-kiberchaika.pem" /media/k4_nas/disk1/Datasets/Music_Part1/ root@188.120.253.126:/home/k4/Datasets/Music_Part1.01

'''

#!/usr/bin/env python3
import os
import sys

BASE_PATH = "/media/k4_nas/disk1/Datasets/Music_Part1"
OUTPUT_FILE = "filelist_part1.01.txt"
MAX_SIZE = 1.5 * 1024**4  # 1.5 TB in bytes

def main():
    print(f"Scanning: {BASE_PATH}")
    
    # Collect all mp3 files with sizes
    files = []
    for root, dirs, filenames in os.walk(BASE_PATH):
        for f in filenames:
            if f.lower().endswith(".mp3"):
                full_path = os.path.join(root, f)
                try:
                    size = os.path.getsize(full_path)
                    rel_path = os.path.relpath(full_path, BASE_PATH)
                    files.append((rel_path, size))
                except OSError as e:
                    print(f"Error reading: {full_path} — {e}")

    # Sort by relative path
    files.sort(key=lambda x: x[0])

    print(f"Total mp3 files found: {len(files)}")
    total_all = sum(s for _, s in files)
    print(f"Total size of all files: {total_all / 1024**4:.3f} TB")

    # Add files until 1.5 TB
    selected = []
    cumulative = 0
    for rel_path, size in files:
        if cumulative + size > MAX_SIZE:
            break
        selected.append(rel_path)
        cumulative += size

    print(f"\nFiles selected: {len(selected)} / {len(files)}")
    print(f"Selected size:  {cumulative / 1024**4:.3f} TB / {MAX_SIZE / 1024**4:.1f} TB")
    print(f"Remaining:      {len(files) - len(selected)} files, "
          f"{(total_all - cumulative) / 1024**4:.3f} TB")

    # Write filelist
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rel_path in selected:
            f.write(rel_path + "\n")

    print(f"\nFilelist saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()