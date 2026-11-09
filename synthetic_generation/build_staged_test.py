"""
build_staged_test.py

Builds a LeFusion input folder (images/ + labels/ + MANIFEST.json + staged_files.txt
+ review PNGs) entirely on this machine. Every volume is a real lesion taken from a
train pathological patient and put into a train NORMAL heart:

  --mode test   3 normal hearts x 2 lesions (the first placed as in its source heart,
                the second with a random angle / width / depth / slice shift).
  --mode full   every train normal heart x --per-heart lesions (default 8), all with
                a random transform. Library lesions are used in turn so each is
                used about equally often.

Pathological hearts are never augmented here: LeFusion only regenerates pixels
labelled 3/4, so a scar that the new mask calls healthy would stay in the image.
Normal hearts have no old scar, so image and mask stay consistent.

Crop recipe (checked against LeFusion's preprocessed files: image and label
arrays identical on 8 pathological and 4 normal cases): bounding box of all
labelled structure (labels > 0) plus 5 px on each side in-plane, all slices kept.
LeFusion then centre-pads/crops to 72 x 72 x 10 itself.

File names must contain an upper-case 'P': LeFusion's loader skips every file
without one (emidec_hist_in.py, get_image_files), which would drop normals.
source_patient() still maps 'Case_N089__insP01' to Case_N089.

Usage:
    python synthetic_generation/build_staged_test.py --emidec ROOT --out DIR --mode test
    python synthetic_generation/build_staged_test.py --emidec ROOT --out DIR --mode full --per-heart 8
"""

import os
import sys
import json
import argparse
from dataclasses import asdict

import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")
for sub in ("", "splits", "verification", "synthetic_generation"):
    sys.path.insert(0, os.path.join(REPO_ROOT, sub))
import config
import mask_diversity as md
from split_utils import load_split, source_patient

MARGIN_PX = 5
LEFUSION_AFFINE = np.diag([-1.5625, -1.5625, 10.0, 1.0])   # same spacing as LeFusion's preprocessed files

TEST_NORMALS = ["Case_N089", "Case_N084", "Case_N083"]
TEST_EXCLUDED_LIBRARY = ["Case_P019", "Case_P004"]          # kept out of the library only to reproduce the first test set


def crop_box(labels):
    nz = np.nonzero(labels > 0)
    y0, y1 = nz[0].min() - MARGIN_PX, nz[0].max() + MARGIN_PX
    x0, x1 = nz[1].min() - MARGIN_PX, nz[1].max() + MARGIN_PX
    if y0 < 0 or x0 < 0 or y1 >= labels.shape[0] or x1 >= labels.shape[1]:
        raise ValueError("crop window leaves the image; recipe not verified for this case")
    return y0, y1, x0, x1


def crop(arr, box):
    y0, y1, x0, x1 = box
    return arr[y0:y1 + 1, x0:x1 + 1]


def write_pair(out, name, img, lab):
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    nib.save(nib.Nifti1Image(img.astype(np.float32), LEFUSION_AFFINE), os.path.join(out, "images", name))
    nib.save(nib.Nifti1Image(lab.astype(np.uint8), LEFUSION_AFFINE), os.path.join(out, "labels", name))


def random_insert(nmask, lib, rng, ref, tries=25):
    """One accepted insertion with a RANDOM transform (md.insert_many uses the
    untransformed lesion for its first single result). Returns (result, rejected)."""
    rejected = []
    for _ in range(tries):
        p = md.sample_params(rng)
        new, leave, lost = md.insert_lesion(nmask, lib, p)
        m = md.lesion_metrics(new)
        why = md.evaluate(lib.metrics, m, leave, lost, ref, use_fraction=True)
        if not why:
            return {"params": asdict(p), "mask": new, "metrics": m}, rejected
        rejected.append(why)
    return None, rejected


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emidec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["test", "full"], default="test")
    ap.add_argument("--per-heart", type=int, default=8, help="full mode: lesions inserted per normal heart")
    a = ap.parse_args()

    split = load_split()
    train, test = set(split["train"]), set(split["test"])
    rng = np.random.RandomState(config.MASK_DIVERSITY_SEED)
    ref = md.load_reference(a.emidec)

    if a.mode == "test":
        normals, per_heart = TEST_NORMALS, 2
        pool = sorted(c for c in train if c.startswith("Case_P") and c not in TEST_EXCLUDED_LIBRARY)
    else:
        normals, per_heart = sorted(c for c in train if c.startswith("Case_N")), a.per_heart
        pool = sorted(c for c in train if c.startswith("Case_P"))
    for cid in normals + pool:
        assert cid in train and cid not in test, f"{cid} is not a train patient"
    order = [str(c) for c in rng.permutation(pool)]
    libs = {}
    manifest, review = [], []
    shortfall = {}
    used_all = set()   # test mode: no lesion is reused across hearts (as in the first test run)

    for hi, nid in enumerate(normals):
        nimg, nmask, _ = md.load_case(a.emidec, nid)
        box = crop_box(nmask)
        used = used_all if a.mode == "test" else set()
        start = (hi * per_heart) % len(order) if a.mode == "full" else 0
        candidates = order[start:] + order[:start]
        for j in range(per_heart):
            skipped = []
            for lib_id in [c for c in candidates if c not in used]:
                if lib_id not in libs:
                    _, lmask, _ = md.load_case(a.emidec, lib_id)
                    libs[lib_id] = md.LibraryLesion(lib_id, lmask)
                lib = libs[lib_id]
                if a.mode == "test":                  # same calls as the first test run, so it is reproducible:
                    # first lesion as in its source heart (n=1), second with a random transform (n=2, first kept)
                    acc, rej = md.insert_many(nmask, lib, 1 if j == 0 else 2, rng, ref)
                    res = acc[0] if acc else None
                else:
                    res, rej = random_insert(nmask, lib, rng, ref)
                if res:
                    break
                skipped.append(lib_id)
            else:
                print(f"  {nid}: only {j} of {per_heart} lesions could be placed (the rest were rejected for this heart)")
                shortfall[nid] = j
                break
            used.add(lib_id)
            name = f"{nid}__insP{j + 1:02d}.nii.gz"
            lab_c, img_c = crop(res["mask"], box), crop(nimg, box)
            write_pair(a.out, name, img_c, lab_c)
            manifest.append({"file": name, "heart": nid, "lesion_from": lib_id, "params": res["params"],
                             "metrics": res["metrics"], "skipped_before": skipped})
            review.append((name, nid, lib_id, nimg, nmask, res["mask"], lib))

    for m in manifest:
        assert source_patient(m["file"]) not in test and source_patient(m["lesion_from"]) not in test
    with open(os.path.join(a.out, "MANIFEST.json"), "w") as f:
        json.dump({"mode": a.mode, "crop_recipe": f"labels>0 bounding box + {MARGIN_PX}px, all slices",
                   "seed": config.MASK_DIVERSITY_SEED, "per_heart": per_heart, "shortfall": shortfall, "files": manifest}, f, indent=1)
    with open(os.path.join(a.out, "staged_files.txt"), "w") as f:
        f.write("\n".join(m["file"] for m in manifest) + "\n")

    # review PNGs: a seeded sample of volumes, lesion source | normal before | after
    pick = rng.choice(len(review), size=min(6, len(review)), replace=False)
    for k in sorted(pick):
        name, nid, lib_id, nimg, nmask, new_mask, lib = review[k]
        _, lmask, _ = md.load_case(a.emidec, lib_id)
        limg, _, _ = md.load_case(a.emidec, lib_id)
        panels = [("lesion source: " + lib_id, limg, lmask), ("before: " + nid, nimg, nmask), ("after", nimg, new_mask)]
        md.save_review_png(os.path.join(a.out, f"review_{name.replace('.nii.gz', '')}.png"), panels, name)

    print(f"{len(manifest)} volumes in {a.out}")
    for m in manifest:
        shape = nib.load(os.path.join(a.out, "images", m["file"])).shape
        print(f"{m['file']:28s} heart {m['heart']}  lesion {m['lesion_from']}  shape {shape}"
              + (f"  (skipped {m['skipped_before']})" if m["skipped_before"] else ""))


if __name__ == "__main__":
    main()
