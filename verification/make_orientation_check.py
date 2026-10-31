"""
make_orientation_check.py

Draws orientation_check.png: for each case, slices 0, middle and last, each as
the raw LGE image next to the same image with cavity and myocardium overlaid.
Uses the array orientation assign_aha17_segment.py uses: array [axis0, axis1]
shown with axis0 as rows (row 0 at the top) and axis1 as columns. No flip or
transpose. Edge labels say where the array edges are on screen. The myocardium
overlay covers labels 2, 3 and 4 (the whole wall) so the full ring is visible.

Usage:
    python verification/make_orientation_check.py /path/to/emidec [out.png]
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
import colors
from split_utils import test_ids

# Case_P030 does not exist in EMIDEC (ids go P029, P031); P029 is the nearest non-test case.
CASES = ["Case_P019", "Case_P004", "Case_P055", "Case_P010", "Case_P029"]


def edge_labels(ax):
    kw = dict(transform=ax.transAxes, fontsize=9, color="black", ha="center", va="center")
    ax.text(0.5, 1.035, "top", **kw)
    ax.text(0.5, -0.035, "bottom", **kw)
    ax.text(-0.05, 0.5, "left", rotation=90, **kw)
    ax.text(1.05, 0.5, "right", rotation=270, **kw)


def main(root, out):
    bad = set(CASES) & test_ids()
    assert not bad, f"test-set patients in CASES: {bad}"
    fig, axes = plt.subplots(len(CASES), 6, figsize=(30, 5.2 * len(CASES)),
                             gridspec_kw={"wspace": 0.28, "hspace": 0.35})
    for r, cid in enumerate(CASES):
        img = np.asarray(nib.load(os.path.join(root, cid, "Images", f"{cid}.nii.gz")).dataobj).astype(float)
        msk = np.asarray(nib.load(os.path.join(root, cid, "Contours", f"{cid}.nii.gz")).dataobj).round().astype(int)
        n = img.shape[2]
        for c, k in enumerate([0, n // 2, n - 1]):
            sl = img[:, :, k]
            lo, hi = np.percentile(sl, [1, 99])
            gray = np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)
            m2 = msk[:, :, k]
            cav_area = int((m2 == 1).sum())
            wall_view = np.where(np.isin(m2, (3, 4)), 2, m2)  # whole wall shown as myocardium
            panels = [("raw", gray), ("cavity + myocardium", colors.apply_overlay(gray, wall_view, regions=["lv_cavity", "myocardium"]))]
            for j, (name, arr) in enumerate(panels):
                ax = axes[r][c * 2 + j]
                ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None, vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                edge_labels(ax)
                ax.set_xlabel(f"slice {k} of 0-{n - 1}   cavity area {cav_area} px", fontsize=8, labelpad=14)
                if r == 0:
                    ax.set_title(name, fontsize=9, pad=14)
                if c == 0 and j == 0:
                    ax.set_ylabel(cid, fontsize=10, labelpad=22)
    fig.suptitle("Orientation check: array [axis0 = rows, axis1 = columns], row 0 at top, no flip (as in assign_aha17_segment.py)", fontsize=11)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.04)
    fig.savefig(out, dpi=75)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "orientation_check.png")
