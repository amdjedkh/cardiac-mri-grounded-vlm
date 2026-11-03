"""
make_orientation_sample.py

Draws orientation_sample.png: the middle slice (n // 2) of 20 random cases
(pathological and normal) drawn from the split's TRAIN list with
config.SPLIT_SEED, raw image with cavity and myocardium overlaid (the whole
wall, labels 2/3/4, shown as myocardium). Array orientation is the one
assign_aha17_segment.py uses: axis0 = rows (row 0 at top), axis1 = columns,
no flip or transpose.

Usage:
    python verification/make_orientation_sample.py /path/to/emidec [out.png]
"""
import os
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "splits"))
import config
import colors
from split_utils import load_split

N = 20


def main(root, out):
    split = load_split()
    pool = sorted(split["train"])                 # test-set patients are never in here
    assert not set(pool) & set(split["test"])
    rng = np.random.RandomState(config.SPLIT_SEED)
    chosen = sorted(rng.choice(pool, size=N, replace=False).tolist())
    print("cases:", ", ".join(chosen))
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for ax, cid in zip(axes.ravel(), chosen):
        img = np.asarray(nib.load(os.path.join(root, cid, "Images", f"{cid}.nii.gz")).dataobj).astype(float)
        msk = np.asarray(nib.load(os.path.join(root, cid, "Contours", f"{cid}.nii.gz")).dataobj).round().astype(int)
        k = img.shape[2] // 2
        sl = img[:, :, k]
        lo, hi = np.percentile(sl, [1, 99])
        gray = np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)
        m2 = np.where(np.isin(msk[:, :, k], (3, 4)), 2, msk[:, :, k])
        ax.imshow(colors.apply_overlay(gray, m2, regions=["lv_cavity", "myocardium"]))
        h, w = gray.shape
        ax.set_xticks(range(0, w, 50)); ax.set_yticks(range(0, h, 50))
        ax.tick_params(labelsize=7, length=2)
        ax.set_xlabel("column (axis1)", fontsize=7, labelpad=1)
        ax.set_ylabel("row (axis0)", fontsize=7, labelpad=1)
        # assumed orientation (assign_aha17_segment.py): RV / septal side = image right, anterior = top
        ax.annotate("RV side", xy=(0.995, 0.5), xycoords="axes fraction", xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", rotation=90, fontsize=9, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#B03030", ec="none", alpha=0.85))
        ax.annotate("anterior", xy=(0.5, 0.995), xycoords="axes fraction", xytext=(0, -4), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color="white", bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.6))
        ax.annotate("lateral", xy=(0.005, 0.5), xycoords="axes fraction", xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", rotation=90, fontsize=8, color="white", bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.6))
        group = "normal" if cid.startswith("Case_N") else "pathological"
        ax.set_title(f"{cid}  ({group})  slice {k} of 0-{img.shape[2] - 1}", fontsize=11)
    fig.suptitle("Orientation sample: middle slice, array [axis0 = rows, axis1 = columns], row 0 at top, no flip "
                 "(as in assign_aha17_segment.py).\nBlue = cavity, green = myocardium. RV side / anterior / lateral are the ASSUMED "
                 "orientation, author-inspected, not confirmed by a cardiologist.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=80)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "orientation_sample.png")
