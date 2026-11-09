"""
aha_coverage.py

Which AHA segments do the 54 real train lesions and the 200 inserts reach?

Per slice: ring level from the slice position (verification/assign_aha17_segment.py:
thirds of the stack, slice 0 = apex), each wall pixel gets the segment of its
angle around the cavity centroid (image top = anterior, image right = septal,
same convention as the report checks). Segments 1-6 basal, 7-12 mid,
13-16 apical. The apex cap (segment 17) is not modelled by that pipeline.

  touch  : the lesion has >= 1 pixel (infarct or MVO) in the segment
  cover  : >= 25% of the segment's wall pixels are lesion

Orientation is author-inspected and not yet confirmed by a cardiologist.
Writes aha_coverage.csv and aha_coverage.png to EXPERIMENT_OUT_DIR.

    python experiments/aha_coverage.py
"""

import os
import csv
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
from common import config, md, load_nii
from assign_aha17_segment import (SECTOR_CENTERS_4, SECTOR_CENTERS_6, SEGMENT_NUMBERS,
                                  determine_ring_level)

COVER_MIN = 0.25
NAMES = {v: f"{lvl} {sec}" for (lvl, sec), v in SEGMENT_NUMBERS.items()}


def segment_counts(mask):
    """(lesion pixels, wall pixels) per segment, arrays of length 17 (index = segment number)."""
    les, wall = np.zeros(17), np.zeros(17)
    n = mask.shape[2]
    for k in range(n):
        g = md.SliceGeom(mask[:, :, k])
        if not g.valid:
            continue
        level = determine_ring_level(k, n)
        centers = SECTOR_CENTERS_4 if level == "apical" else SECTOR_CENTERS_6
        names = list(centers)
        bull = (np.degrees(g.theta) % 360 - 270) % 360
        d = np.abs(bull[:, None] - np.array([centers[x] for x in names])[None, :])
        sector = np.argmin(np.minimum(d, 360 - d), axis=1)
        seg = np.array([SEGMENT_NUMBERS[(level, names[i])] for i in sector])
        is_les = np.isin(g.label, (md.INF, md.MVO))
        np.add.at(wall, seg, 1)
        np.add.at(les, seg, is_les)
    return les, wall


def lesion_sets(mask):
    les, wall = segment_counts(mask)
    touch = {s for s in range(1, 17) if les[s] >= 1}
    cover = {s for s in range(1, 17) if wall[s] and les[s] / wall[s] >= COVER_MIN}
    return touch, cover


def main():
    real = []
    for cid in common.real_train_cases():
        _, mask, _ = md.load_case(config.EMIDEC_ROOT, cid)
        real.append(lesion_sets(mask))
    ins = []
    for job in common.JOBS:
        for p in sorted(glob.glob(f"{config.FULL_INSERT_DIR}/input_{job}/labels/*.nii.gz")):
            ins.append(lesion_sets(load_nii(p).astype(int)))

    real_cover_union = set().union(*[c for _, c in real])
    real_touch_union = set().union(*[t for t, _ in real])
    rows = []
    for s in range(1, 17):
        rows.append({"segment": s, "name": NAMES[s],
                     "real_touch": sum(s in t for t, _ in real), "real_cover": sum(s in c for _, c in real),
                     "insert_touch": sum(s in t for t, _ in ins), "insert_cover": sum(s in c for _, c in ins)})
    os.makedirs(config.EXPERIMENT_OUT_DIR, exist_ok=True)
    with open(os.path.join(config.EXPERIMENT_OUT_DIR, "aha_coverage.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"lesions: real {len(real)}, inserts {len(ins)}")
    print(f"{'seg':>3} {'name':28s} {'real touch':>10} {'real cover':>10} {'ins touch':>10} {'ins cover':>10}")
    for r in rows:
        print(f"{r['segment']:>3} {r['name']:28s} {r['real_touch']:>10} {r['real_cover']:>10} {r['insert_touch']:>10} {r['insert_cover']:>10}")
    new_cover = sum(bool(c - real_cover_union) for _, c in ins)
    new_touch = sum(bool(t - real_touch_union) for t, _ in ins)
    print(f"segments covered by no real lesion: {sorted(set(range(1, 17)) - real_cover_union)}; touched by none: {sorted(set(range(1, 17)) - real_touch_union)}")
    print(f"inserts covering >=1 segment no real lesion covers: {new_cover} of {len(ins)}; touching one no real lesion touches: {new_touch}")
    print(f"segments never touched by any real lesion or insert: {sorted(set(range(1, 17)) - real_touch_union - set().union(*[t for t, _ in ins]))}")

    x = np.arange(16)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(x - 0.2, [100 * r["real_touch"] / len(real) for r in rows], 0.4, label=f"real train (n={len(real)})", color="#4C72B0")
    ax.bar(x + 0.2, [100 * r["insert_touch"] / len(ins) for r in rows], 0.4, label=f"inserts (n={len(ins)})", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['segment']}\n{r['name'].replace('apical ', '').replace('basal ', '').replace('mid ', '')}" for r in rows], fontsize=7)
    ax.set_ylabel("% of lesions with >= 1 lesion pixel in the segment")
    ax.set_title("AHA segment coverage (1-6 basal, 7-12 mid, 13-16 apical; orientation not cardiologist-confirmed)", fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.EXPERIMENT_OUT_DIR, "aha_coverage.png"), dpi=110)


if __name__ == "__main__":
    main()
