"""
unique_shapes.py

Counts distinct lesion shapes among the 54 real train lesions and the 200
LeFusion inserts. Definition (exact):

  Footprint: trim empty end slices, then unroll the wall into a grid of
  5 relative slice positions x 24 angle bins. A cell is 1 when >= 25% of the
  wall pixels in it are lesion (label 3 or 4).
  Same shape: footprint IoU >= THR after the best circular shift along the
  angle axis (so a rotated copy is the same shape). 'shape+location' is the
  same test WITHOUT the shift, so rotation counts as different.
  Unique shapes: leader clustering in file order; a lesion is new if its IoU
  to every earlier leader is < THR (order dependent, so the count is an
  estimate, not a canonical number).

Reads EMIDEC_ROOT and FULL_INSERT_DIR from config.py.

    python experiments/unique_shapes.py
"""

import glob
import os

import numpy as np

import common
from common import config, md, load_nii

Z_BINS, A_BINS, CELL_MIN = 5, 24, 0.25


def footprint(mask):
    valid = [k for k in range(mask.shape[2]) if md.SliceGeom(mask[:, :, k]).valid]
    if not valid:
        return None
    mask = mask[:, :, valid[0]:valid[-1] + 1]
    n = mask.shape[2]
    num, den = np.zeros((Z_BINS, A_BINS)), np.zeros((Z_BINS, A_BINS))
    for k in range(n):
        g = md.SliceGeom(mask[:, :, k])
        if not g.valid:
            continue
        zb = min(int(round(k / max(n - 1, 1) * (Z_BINS - 1))), Z_BINS - 1)
        ab = np.minimum(((g.theta + np.pi) / (2 * np.pi) * A_BINS).astype(int), A_BINS - 1)
        les = np.isin(g.label, (md.INF, md.MVO))
        np.add.at(num[zb], ab, les)
        np.add.at(den[zb], ab, 1)
    fp = np.where(den > 0, num / np.maximum(den, 1), 0) >= CELL_MIN
    return fp if fp.any() else None


def iou(a, b, shift):
    best = 0.0
    for s in (range(A_BINS) if shift else [0]):
        bb = np.roll(b, s, axis=1)
        u = (a | bb).sum()
        best = max(best, (a & bb).sum() / u if u else 0)
    return best


def count_unique(fps, thr, shift):
    leaders = []
    for f in fps:
        if not any(iou(f, l, shift) >= thr for l in leaders):
            leaders.append(f)
    return len(leaders)


def report(name, fps):
    out = {f"shape@{t}": count_unique(fps, t, True) for t in (0.5, 0.6, 0.7)}
    out["shape+location@0.6"] = count_unique(fps, 0.6, False)
    print(f"{name:34s} lesions={len(fps):4d}  " + "  ".join(f"{k}={v}" for k, v in out.items()))
    return out


def main():
    real = []
    for cid in common.real_train_cases():
        _, mask, _ = md.load_case(config.EMIDEC_ROOT, cid)
        fp = footprint(mask)
        if fp is not None:
            real.append(fp)
    ins = []
    for job in common.JOBS:
        for p in sorted(glob.glob(f"{config.FULL_INSERT_DIR}/input_{job}/labels/*.nii.gz")):
            fp = footprint(load_nii(p).astype(int))
            if fp is not None:
                ins.append(fp)
    report("real train", real)
    report("200 inserts alone", ins)
    report("real train + 200 inserts", real + ins)


if __name__ == "__main__":
    main()
