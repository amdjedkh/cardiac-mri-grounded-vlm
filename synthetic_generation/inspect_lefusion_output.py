"""
inspect_lefusion_output.py

Checks what's actually in the LeFusion output folder on the volume, with
timestamps -- to tell whether the last run's 47 listed files were all
genuinely just-generated (meaning max_cases didn't limit things the way
expected), or whether most of them are leftovers from the earlier
full-directory run that hit the 30-minute timeout partway through.

Usage:
    modal run inspect_lefusion_output.py
"""

import modal

app = modal.App("lefusion-inspect")
volume = modal.Volume.from_name("cardiac-data", create_if_missing=False)
VOLUME_PATH = "/cardiac-data"


@app.function(volumes={VOLUME_PATH: volume})
def inspect():
    import os
    import time

    out_img = f"{VOLUME_PATH}/LeFusion_output/Image/"
    out_mask = f"{VOLUME_PATH}/LeFusion_output/Mask/"

    for label, path in [("Image", out_img), ("Mask", out_mask)]:
        print(f"\n=== {label} folder: {path} ===")
        if not os.path.exists(path):
            print("does not exist")
            continue
        files = os.listdir(path)
        print(f"Total files: {len(files)}")
        entries = []
        for f in files:
            full = os.path.join(path, f)
            mtime = os.path.getmtime(full)
            entries.append((mtime, f))
        entries.sort()
        print("\nOldest 5 (by last-modified time):")
        for mtime, f in entries[:5]:
            print(f"  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}  {f}")
        print("\nNewest 5 (by last-modified time):")
        for mtime, f in entries[-5:]:
            print(f"  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}  {f}")


@app.local_entrypoint()
def main():
    inspect.remote()
