"""
render_case_bullseye.py

Directly builds what Karen asked for: an AHA17 bullseye colored with REAL
computed data from a case's actual segmentation mask (like panel D in her
reference image), not a generic textbook reference.

For every slice in the case, this determines the AHA17 segment(s) present
and computes what fraction of each segment's myocardial area is infarct
(and separately, MVO), across the WHOLE volume, then renders a standard
17-segment bullseye colored by that real per-segment involvement.

Reuses the exact angle math and sector assignment already tested in
verify_slice_claims.py and assign_aha17_segment.py.

Usage:
    python render_case_bullseye.py Case_P019
"""

import sys
import os
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verify_slice_claims import DEFAULT_LABEL_MAP
from assign_aha17_segment import (
    to_bullseye_angle, nearest_sector, determine_ring_level,
    SECTOR_CENTERS_6, SECTOR_CENTERS_4, SEGMENT_NUMBERS,
)


def compute_case_segment_involvement(case_id: str, root: str = ".", label_map: dict = None):
    """
    Loops over every slice in the case's real mask, assigns every infarct/MVO
    pixel to its AHA17 segment, and returns per-segment: infarct area, MVO
    area, and total myocardium area (infarct + MVO + normal), so involvement
    can be expressed as a percentage of that segment's own myocardium.
    """
    lm = label_map or DEFAULT_LABEL_MAP
    mask_path = os.path.join(root, case_id, "Contours", f"{case_id}.nii.gz")
    img = nib.load(mask_path)
    data = np.asarray(img.get_fdata()).round().astype(int)
    n_slices = data.shape[2]

    # per-segment accumulators: {segment_number: {"infarct": px, "mvo": px, "myo": px}}
    segments = {}

    for s in range(n_slices):
        mask_2d = data[:, :, s]
        cavity_ys, cavity_xs = np.where(mask_2d == lm["lv_cavity"])
        if len(cavity_ys) == 0:
            continue  # no LV cavity on this slice (e.g. beyond the heart), skip
        center = (cavity_ys.mean(), cavity_xs.mean())
        level = determine_ring_level(s, n_slices)

        for label_name, key in [("myocardium_normal", "myo"), ("infarct", "infarct"), ("mvo", "mvo")]:
            ys, xs = np.where(mask_2d == lm[label_name])
            if len(ys) == 0:
                continue
            angles = np.degrees(np.arctan2(ys - center[0], xs - center[1])) % 360
            for a in angles:
                bullseye = to_bullseye_angle(a)
                if level == "apical":
                    sector = nearest_sector(bullseye, SECTOR_CENTERS_4)
                else:
                    sector = nearest_sector(bullseye, SECTOR_CENTERS_6)
                seg_num = SEGMENT_NUMBERS.get((level, sector))
                if seg_num is None:
                    continue
                segments.setdefault(seg_num, {"infarct": 0, "mvo": 0, "myo": 0, "level": level, "sector": sector})
                segments[seg_num][key] += 1
                if key in ("infarct", "mvo"):
                    segments[seg_num]["myo"] += 1  # infarct/mvo tissue is still myocardium

    return segments, n_slices


def render_bullseye_from_data(segments: dict, case_id: str, out_path: str):
    fig = plt.figure(figsize=(8.5, 9.2), facecolor="white")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 3)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    ring_bounds = {"basal": (2, 3), "mid": (1, 2), "apical": (0.4, 1)}
    ring_nseg = {"basal": 6, "mid": 6, "apical": 4}
    ring_offset = {"basal": -30, "mid": -30, "apical": -45}

    def seg_geometry(level, sector, n_seg):
        centers = SECTOR_CENTERS_6 if n_seg == 6 else SECTOR_CENTERS_4
        idx = list(centers.keys()).index(sector)
        width = 2 * np.pi / n_seg
        theta = idx * width + np.radians(ring_offset[level])
        return theta, width

    cmap = plt.cm.Reds

    for seg_num in range(1, 18):
        if seg_num == 17:
            level, sector = "apex", "apex"
        else:
            level, sector = next(((lv, sec) for (lv, sec), n in SEGMENT_NUMBERS.items() if n == seg_num), (None, None))
        if level is None:
            continue

        data = segments.get(seg_num)
        if data and data["myo"] > 0:
            pct_involved = (data["infarct"] + data["mvo"]) / data["myo"] * 100
        else:
            pct_involved = 0.0
        color = cmap(0.15 + 0.75 * min(pct_involved / 100, 1.0)) if pct_involved > 0 else "#E8E8E8"

        if seg_num == 17:
            ax.bar(0, 0.4, width=2 * np.pi, bottom=0, color=color, edgecolor="white", linewidth=1.5)
            ax.text(0, 0.0, f"17\n{pct_involved:.0f}%", ha="center", va="center",
                     fontsize=8, fontweight="bold",
                     color="white" if pct_involved > 40 else "#1A1A1A")
            continue

        n_seg = ring_nseg[level]
        theta, width = seg_geometry(level, sector, n_seg)
        r_in, r_out = ring_bounds[level]
        ax.bar(theta, r_out - r_in, width=width, bottom=r_in, color=color,
               edgecolor="white", linewidth=1.5)
        label_r = (r_in + r_out) / 2
        ax.text(theta, label_r, f"{seg_num}\n{pct_involved:.0f}%", ha="center", va="center",
                 fontsize=8, fontweight="bold",
                 color="white" if pct_involved > 40 else "#1A1A1A")

    ax.set_title(f"{case_id} \u2014 AHA17 bullseye, real infarct + MVO involvement per segment\n"
                 f"(computed from the actual mask, % of each segment's myocardium)",
                 fontsize=11.5, fontweight="bold", pad=28)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08, shrink=0.7)
    cbar.set_label("% of segment myocardium that is infarct or MVO", fontsize=9.5)

    plt.savefig(out_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render_case_bullseye.py <Case_ID>")
        sys.exit(1)

    case_id = sys.argv[1]
    segments, n_slices = compute_case_segment_involvement(case_id)

    print(f"\n=== {case_id}: real AHA17 segment involvement, {n_slices} slices ===\n")
    for seg_num in sorted(segments.keys()):
        d = segments[seg_num]
        pct = (d["infarct"] + d["mvo"]) / d["myo"] * 100 if d["myo"] > 0 else 0
        if pct > 0:
            print(f"segment {seg_num} ({d['level']} {d['sector']}): {pct:.1f}% involved "
                  f"(infarct px={d['infarct']}, mvo px={d['mvo']})")

    out_path = f"{case_id}_aha17_bullseye_real.png"
    render_bullseye_from_data(segments, case_id, out_path)
    print(f"\nSaved to {out_path}")
    print("\nASSUMPTION FLAG (same as assign_aha17_segment.py): image-top = anterior "
          "(now visually confirmed via sternum), slice 0 = base (not yet independently confirmed).")
