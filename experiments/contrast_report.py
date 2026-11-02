"""
contrast_report.py

Lesion contrast of the 200 LeFusion inserts and of the 54 real train
pathological cases, on LeFusion's [-1, 1] scale.

  contrast_all           mean(infarct + MVO) - mean(healthy wall)
  contrast_infarct_only  mean(infarct)       - mean(healthy wall)
  contrast_mvo_only      mean(MVO)           - mean(healthy wall)
  outside_mean_abs_change  (inserts only) mean |output - input| over input
                           voxels that are not lesion

Real cases are cropped with the same recipe as the staged inputs
(bounding box of labels > 0 + 5 px) and min-max rescaled to [-1, 1].
Nothing is filtered or deleted; the thresholds only count.

Reads  FULL_INSERT_DIR (incl. MANIFEST.json) and EMIDEC_ROOT from config.py.
Writes contrast_inserts_200.csv, contrast_real_train.csv, contrast_summary.txt
to EXPERIMENT_OUT_DIR.

    python experiments/contrast_report.py
"""

import os
import csv
import json

import numpy as np

import common
from common import config, md, bs, INFARCT, MVO

THRESHOLDS = (0.05, 0.099)   # 0.099 = weakest real train lesion
PIXEL_MM2 = 1.5625 ** 2
COLUMNS = ["file", "heart", "lesion_from", "kind", "infarct_area_px", "mvo_area_px", "infarct_area_mm2",
           "mvo_area_mm2", "size_tertile", "contrast_all", "contrast_infarct_only", "contrast_mvo_only",
           "outside_mean_abs_change"]


def row_from(name, heart, lesion_from, img, mask, outside=None):
    inf, mvo = int((mask == INFARCT).sum()), int((mask == MVO).sum())
    return {"file": name, "heart": heart, "lesion_from": lesion_from,
            "kind": "infarct+MVO" if mvo else "infarct only",
            "infarct_area_px": inf, "mvo_area_px": mvo,
            "infarct_area_mm2": round(inf * PIXEL_MM2, 1), "mvo_area_mm2": round(mvo * PIXEL_MM2, 1),
            "contrast_all": common.contrast(img, mask, (INFARCT, MVO)),
            "contrast_infarct_only": common.contrast(img, mask, (INFARCT,)),
            "contrast_mvo_only": common.contrast(img, mask, (MVO,)),
            "outside_mean_abs_change": outside}


def add_tertiles(rows):
    size = np.array([r["infarct_area_px"] + r["mvo_area_px"] for r in rows])
    lo, hi = np.quantile(size, [1 / 3, 2 / 3])
    for r, s in zip(rows, size):
        r["size_tertile"] = "small" if s <= lo else ("medium" if s <= hi else "large")


def insert_rows():
    manifest_path = os.path.join(config.FULL_INSERT_DIR, "MANIFEST.json")
    man = {m["file"]: m for m in json.load(open(manifest_path))["files"]} if os.path.exists(manifest_path) else {}
    rows = []
    for v in common.iter_volumes():
        name = v["file"]
        off = common.locate(v["lab_in"], v["out_mask"])
        if off is None:
            raise SystemExit(f"{name}: output mask != padded input mask; run validate_full.py")
        in_n = common.place(common.minmax(v["img_in"]), v["out_img"].shape, off)
        valid = common.place(np.ones(v["img_in"].shape, bool), v["out_img"].shape, off)
        lesion = np.isin(v["out_mask"], (INFARCT, MVO))
        outside = float(np.abs(v["out_img"] - in_n)[valid & ~lesion].mean())
        m = man.get(name, {})
        rows.append(row_from(name, m.get("heart", name.split("__")[0]), m.get("lesion_from", ""),
                             v["out_img"], v["out_mask"], round(outside, 4)))
    add_tertiles(rows)
    return rows


def real_rows():
    rows, skipped = [], []
    for cid in common.real_train_cases():
        img, mask, _ = md.load_case(config.EMIDEC_ROOT, cid)
        try:
            box = bs.crop_box(mask)
        except ValueError as e:
            skipped.append((cid, str(e)))
            continue
        img_c, mask_c = bs.crop(img, box), bs.crop(mask, box)
        if not np.isin(mask_c, (INFARCT, MVO)).any():
            skipped.append((cid, "no infarct or MVO"))
            continue
        rows.append(row_from(cid, cid, cid, common.minmax(img_c), mask_c))
    add_tertiles(rows)
    return rows, skipped


def stats(rows, key):
    v = np.array([r[key] for r in rows if r[key] is not None and not np.isnan(r[key])])
    if not len(v):
        return "n=0"
    return (f"n={len(v):3d}  mean {v.mean():+.3f}  median {np.median(v):+.3f}  "
            f"min {v.min():+.3f}  max {v.max():+.3f}")


def summary(title, rows, lines):
    lines.append(f"== {title} (n={len(rows)})")
    lines.append(f"  all lesions        contrast_all           {stats(rows, 'contrast_all')}")
    lines.append(f"  infarct voxels     contrast_infarct_only  {stats(rows, 'contrast_infarct_only')}")
    lines.append(f"  MVO voxels         contrast_mvo_only      {stats(rows, 'contrast_mvo_only')}")
    for kind in ("infarct only", "infarct+MVO"):
        sub = [r for r in rows if r["kind"] == kind]
        lines.append(f"  kind {kind:13s} contrast_all           {stats(sub, 'contrast_all')}")
    for t in ("small", "medium", "large"):
        sub = [r for r in rows if r["size_tertile"] == t]
        sizes = [r["infarct_area_px"] + r["mvo_area_px"] for r in sub]
        lines.append(f"  size {t:7s} ({min(sizes)}-{max(sizes)} px) contrast_all {stats(sub, 'contrast_all')}")
    for th in THRESHOLDS:
        n = sum(r["contrast_all"] < th for r in rows)
        lines.append(f"  contrast_all < {th}: {n} of {len(rows)}")
    lines.append("")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in COLUMNS})


def main():
    out = config.EXPERIMENT_OUT_DIR
    os.makedirs(out, exist_ok=True)
    ins, (real, skipped) = insert_rows(), real_rows()
    write_csv(os.path.join(out, "contrast_inserts_200.csv"), ins)
    write_csv(os.path.join(out, "contrast_real_train.csv"), real)
    lines = []
    summary("200 inserts", ins, lines)
    summary("real train pathological", real, lines)
    ocm = np.array([r["outside_mean_abs_change"] for r in ins])
    lines.append(f"outside-lesion mean |out - in| per volume: mean {ocm.mean():.4f}  median {np.median(ocm):.4f}  "
                 f"min {ocm.min():.4f}  max {ocm.max():.4f}")
    if skipped:
        lines.append(f"real cases skipped: {skipped}")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(out, "contrast_summary.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
