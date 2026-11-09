"""
contiguity_check.py

Contiguity (and MVO-touch) of the real train lesions, full size vs cropped the
way the staged inputs are (bbox of labels > 0 + 5 px, then centre pad / crop to
72 x 72 x 10), and of the 200 inserts, against mask_diversity's real reference
(mask_diversity_reference.json).

That reference was computed by compute_reference() on the FULL-SIZE native
masks of ALL 67 pathological cases (train and test). A bounding-box crop keeps
the same pixel grid, so the score should not change; this script checks it, and
re-applies the accept/reject range test to every insert with (a) the shipped
reference and (b) a reference rebuilt from the 54 train lesions at 72 x 72 x 10.

    python experiments/contiguity_check.py
"""

import glob

import numpy as np

import common
from common import config, md, bs, load_nii

SHAPE = config.CROP_SHAPE if hasattr(config, "CROP_SHAPE") else (72, 72, 10)


def main():
    shipped = md.load_reference()
    full, cropped = {}, {}
    for cid in common.real_train_cases():
        _, mask, _ = md.load_case(config.EMIDEC_ROOT, cid)
        mf = md.lesion_metrics(mask)
        mc = md.lesion_metrics(common.to_shape(bs.crop(mask, bs.crop_box(mask)), SHAPE, 0))
        full[cid], cropped[cid] = mf, mc
    diff = max(abs(full[c]["contiguity"] - cropped[c]["contiguity"]) for c in full if full[c]["contiguity"] is not None)
    tdiff = max(abs(full[c]["mvo_touch"] - cropped[c]["mvo_touch"]) for c in full if full[c]["mvo_touch"] is not None)

    ins = {}
    for job in common.JOBS:
        for p in sorted(glob.glob(f"{config.FULL_INSERT_DIR}/input_{job}/labels/*.nii.gz")):
            ins[p.split("/")[-1]] = md.lesion_metrics(common.to_shape(load_nii(p).astype(int), SHAPE, 0))

    def rng(d, key):
        v = [m[key] for m in d.values() if m[key] is not None]
        return min(v), max(v), len(v)

    print("contiguity, real train lesions (54)")
    for name, d in (("full size", full), ("72x72x10", cropped)):
        lo, hi, n = rng(d, "contiguity")
        print(f"  {name:10s} min {lo:.3f} max {hi:.3f} (n={n})")
    print(f"  max |full - cropped| per case: contiguity {diff:.6f}, mvo_touch {tdiff:.6f}")
    lo, hi, n = rng(ins, "contiguity")
    print(f"contiguity, 200 inserts: min {lo:.3f} max {hi:.3f} (n={n})")
    print(f"shipped reference (67 cases, full size): {shipped['contiguity']['min']:.3f} - {shipped['contiguity']['max']:.3f}; "
          f"mvo_touch min {shipped['mvo_touch']['min']:.3f}")

    ref_train = {"contiguity": {"min": rng(cropped, "contiguity")[0], "max": rng(cropped, "contiguity")[1]},
                 "mvo_touch": {"min": rng(cropped, "mvo_touch")[0]}}

    def decisions(ref):
        out = {}
        for name, m in ins.items():
            bad = m["contiguity"] is None or not ref["contiguity"]["min"] <= m["contiguity"] <= ref["contiguity"]["max"]
            if m["mvo_px"] > 0:
                bad = bad or m["mvo_touch"] is None or m["mvo_touch"] < ref["mvo_touch"]["min"]
            out[name] = bad
        return out

    a, b = decisions(shipped), decisions(ref_train)
    print(f"inserts rejected by the contiguity / MVO-touch rules: shipped reference {sum(a.values())}, "
          f"train-only 72x72x10 reference {sum(b.values())} "
          f"(train-only range {ref_train['contiguity']['min']:.3f} - {ref_train['contiguity']['max']:.3f}, "
          f"mvo_touch min {ref_train['mvo_touch']['min']:.3f})")
    changed = sorted(n for n in a if a[n] != b[n])
    print(f"decisions that change: {len(changed)} {changed[:10]}")


if __name__ == "__main__":
    main()
