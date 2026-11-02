"""
montage.py

Montage PNGs of the insert run, read from contrast_inserts_200.csv:
  montage_random12.png   12 random volumes (seeded with MASK_DIVERSITY_SEED)
  montage_faintest8.png  the 8 with the lowest contrast_all
  montage_brightest8.png the 8 with the highest contrast_all

Each volume is a pair of panels: the LeFusion output with its mask overlaid
(slice with the most lesion voxels), and the source heart the lesion was
taken from (its slice with the most lesion voxels), same colours and the same
[-1, 1] grey scale.

    python experiments/montage.py
"""

import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import colors
from common import config, md, bs, INFARCT, MVO

PER_ROW = 2   # volumes per montage row (4 panels)


def heart_window(mask2d_any, margin=6):
    ys, xs = np.nonzero(mask2d_any)
    return max(ys.min() - margin, 0), ys.max() + margin, max(xs.min() - margin, 0), xs.max() + margin


def best_slice(mask):
    return int(np.argmax(np.isin(mask, (INFARCT, MVO)).sum(axis=(0, 1))))


def draw(ax, img, mask, title):
    k = best_slice(mask)
    y0, y1, x0, x1 = heart_window((mask > 0).any(axis=2))
    ax.imshow(img[y0:y1, x0:x1, k], cmap="gray", vmin=-1, vmax=1)
    ax.imshow(colors.rgba_overlay(mask[:, :, k])[y0:y1, x0:x1])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7)


def source_heart(lesion_from):
    img, mask, _ = md.load_case(config.EMIDEC_ROOT, lesion_from)
    box = bs.crop_box(mask)
    return common.minmax(bs.crop(img, box)), bs.crop(mask, box)


def make_png(path, names, rows, vols, title):
    nrow = (len(names) + PER_ROW - 1) // PER_ROW
    fig, axes = plt.subplots(nrow, 2 * PER_ROW, figsize=(2.6 * 2 * PER_ROW, 2.7 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, name in enumerate(names):
        r, v = rows[name], vols[name]
        a, b = axes[i // PER_ROW][2 * (i % PER_ROW)], axes[i // PER_ROW][2 * (i % PER_ROW) + 1]
        a.axis("on"); b.axis("on")
        draw(a, v["out_img"], v["out_mask"], f"{name.replace('.nii.gz', '')}\ncontrast {float(r['contrast_all']):+.3f}")
        simg, smask = source_heart(r["lesion_from"])
        draw(b, simg, smask, f"source {r['lesion_from']}")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close()


def main():
    out = config.EXPERIMENT_OUT_DIR
    with open(os.path.join(out, "contrast_inserts_200.csv")) as f:
        rows = {r["file"]: r for r in csv.DictReader(f)}
    order = sorted(rows, key=lambda n: float(rows[n]["contrast_all"]))
    rng = np.random.RandomState(config.MASK_DIVERSITY_SEED)
    groups = {"random12": [order[i] for i in sorted(rng.choice(len(order), 12, replace=False))],
              "faintest8": order[:8], "brightest8": order[::-1][:8]}
    wanted = {n for g in groups.values() for n in g}
    vols = {v["file"]: v for v in common.iter_volumes() if v["file"] in wanted}
    for key, names in groups.items():
        make_png(os.path.join(out, f"montage_{key}.png"), names, rows, vols,
                 f"{key}: output with mask (left) | source heart of the lesion (right)")
        print("wrote", os.path.join(out, f"montage_{key}.png"))


if __name__ == "__main__":
    main()
