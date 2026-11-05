"""
evaluate.py

Evaluation of the segmentation arms. Two subcommands.

  dice     predict one (arm, seed) with its checkpoint_final.pth and write per-case Dice
  summary  read the per-case CSVs and report mean +- std over seeds, per-seed values and the
           patient-level bootstrap for the primary metric

FROZEN SETTINGS (do not change after the first test use; the SHA-256 of this file is printed on every run)
  checkpoint   checkpoint_final.pth ONLY. The script stops if it is missing. checkpoint_best.pth and
               checkpoint_latest.pth are never loaded.
  prediction   nnUNetPredictor defaults (tile_step_size 0.5, mirroring on, no probabilities), fold 'all'
  crop         "labelbox": bounding box of labels > 0 + 5 px in plane, all slices, then centre pad/crop to
                           72 x 72 x 10 (the training recipe; uses the expert contour, so Dice is optimistic)
               "centre":   72 x 72 in-plane window at the centre of the full slice, no label information,
                           depth centre pad/crop to 10
  intensity    min-max to [-1, 1] over the cropped volume (nnU-Net itself does not normalise: channel 'noNorm')
  spacing      nominal 1.5625 x 1.5625 x 10 mm written to the NIfTI, as in training
  Dice         3D, over the 72 x 72 x 10 crop, classes 1 LV cavity, 2 myocardium, 3 infarct, 4 MVO; undefined
               (NaN, left out of means) when the class is absent from the expert mask. GT voxels lost by the
               crop are counted and reported.
  primary      infarct Dice on the pathological cases of the split (patients with infarct = true in
               splits/split_v1.json: 13 in the test set), labelbox crop, arm 2 minus arm 1, same seed paired.
               Per case: mean over seeds of (arm 2 - arm 1). Bootstrap: resample cases with replacement,
               10,000 times, numpy RandomState(0), 95% percentile interval of the mean difference.
  test set     case IDs come from splits/split_v1.json. Test images are opened ONLY with
               --split test --confirm "run on test".

    python experiments/nnunet/evaluate.py dice --arm 1 --seed 0 --split train --crop labelbox --cases 3
    python experiments/nnunet/evaluate.py summary --split train --crop labelbox
"""

import os
import sys
import csv
import json
import argparse
import hashlib
import tempfile

import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import common
from common import config, md, bs, load_split
from make_arm1_dataset import to_shape

DATASETS = {1: "Dataset501_EMIDECArm1", 2: "Dataset502_EMIDECArm2", 3: "Dataset501_EMIDECArm1",
            4: "Dataset504_EMIDECArm4", 5: "Dataset505_EMIDECArm5"}
TRAINERS = {1: "nnUNetTrainerEmidec", 2: "nnUNetTrainerEmidec", 3: "nnUNetTrainerEmidecStrongAug",
            4: "nnUNetTrainerEmidec", 5: "nnUNetTrainerEmidec"}
CLASSES = {"lv": 1, "myo": 2, "infarct": 3, "mvo": 4}
SHAPE = config.CROP_SHAPE
BOOT_N, BOOT_SEED = 10000, 0
TEST_CONFIRM = "run on test"


def file_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]


def prepare(case, crop):
    """Image (float32, [-1, 1]) and label (uint8) in the 72 x 72 x 10 evaluation space, plus GT voxels lost."""
    img, mask, _ = md.load_case(config.EMIDEC_ROOT, case)
    if crop == "labelbox":
        box = bs.crop_box(mask)
        img_c, mask_c = bs.crop(img, box), bs.crop(mask, box)
    elif crop == "centre":
        img_c = to_shape(img.astype(np.float32), (SHAPE[0], SHAPE[1], img.shape[2]), float(img.min()))
        mask_c = to_shape(mask.astype(np.uint8), (SHAPE[0], SHAPE[1], img.shape[2]), 0)
    else:
        raise ValueError(crop)
    img_n = to_shape(common.minmax(img_c).astype(np.float32), SHAPE, 0.0)
    lab = to_shape(mask_c.astype(np.uint8), SHAPE, 0)
    return img_n, lab, int((mask > 0).sum() - (lab > 0).sum())


def dice(pred, gt, k):
    g, p = gt == k, pred == k
    if not g.any():
        return float("nan")
    return float(2 * (g & p).sum() / (g.sum() + p.sum()))


def model_dir(arm, seed, results):
    return os.path.join(results, f"arm{arm}_seed{seed}", DATASETS[arm], f"{TRAINERS[arm]}__nnUNetPlans__2d")


def cmd_dice(a):
    split = load_split()
    if a.split == "test":
        if a.confirm != TEST_CONFIRM:
            sys.exit(f'refusing to open test cases: pass --split test --confirm "{TEST_CONFIRM}" only after the user says so')
        cases = list(split["test"])
    else:
        cases = sorted(split["train"])
        assert not set(cases) & set(split["test"])
    if a.cases:
        cases = cases[:a.cases] if not a.pathological else [c for c in cases if split["patients"][c]["infarct"]][:a.cases]
    mdir = model_dir(a.arm, a.seed, a.results)
    ckpt = os.path.join(mdir, "fold_all", "checkpoint_final.pth")
    if not os.path.isfile(ckpt):
        sys.exit(f"MISSING {ckpt}: refusing to evaluate (never falls back to checkpoint_best/latest)")
    print(f"script sha256 {file_sha()} | checkpoint {ckpt} | split {a.split} | crop {a.crop} | {len(cases)} cases")
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                           device=torch.device(a.device), verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(mdir, use_folds=("all",), checkpoint_name="checkpoint_final.pth")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        inp, outp = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(inp)
        gts, lost = {}, {}
        for c in cases:
            img, lab, lost[c] = prepare(c, a.crop)
            gts[c] = lab
            nib.save(nib.Nifti1Image(img, bs.LEFUSION_AFFINE), os.path.join(inp, f"{c}_0000.nii.gz"))
        pred.predict_from_files(inp, outp, save_probabilities=False, overwrite=True,
                                num_processes_preprocessing=1, num_processes_segmentation_export=1, num_parts=1, part_id=0)
        for c in cases:
            p = np.asarray(nib.load(os.path.join(outp, f"{c}.nii.gz")).dataobj).astype(np.uint8)
            row = {"split": a.split, "crop": a.crop, "arm": a.arm, "seed": a.seed, "case": c,
                   "pathological": int(split["patients"][c]["infarct"]), "gt_voxels_lost_by_crop": lost[c]}
            for name, k in CLASSES.items():
                row[f"dice_{name}"] = dice(p, gts[c], k)
                row[f"gt_{name}"] = int((gts[c] == k).sum())
                row[f"pred_{name}"] = int((p == k).sum())
            rows.append(row)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"dice_{a.split}_{a.crop}_arm{a.arm}_seed{a.seed}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote", path)
    for r in rows:
        print(f"  {r['case']:10s} " + " ".join(f"{n} {r['dice_'+n]:.3f}" for n in CLASSES) + f"  lost {r['gt_voxels_lost_by_crop']}")


def load_rows(a):
    out = {}
    for arm in a.arms:
        for seed in a.seeds:
            p = os.path.join(a.out, f"dice_{a.split}_{a.crop}_arm{arm}_seed{seed}.csv")
            if not os.path.isfile(p):
                sys.exit(f"missing {p}")
            out[(arm, seed)] = {r["case"]: r for r in csv.DictReader(open(p))}
    return out


def cmd_summary(a):
    R = load_rows(a)
    cases = sorted(set.intersection(*[set(v) for v in R.values()]))
    patho = [c for c in cases if next(iter(R.values()))[c]["pathological"] == "1"]
    tag = "TRAIN-CASE NUMBERS, NOT PERFORMANCE: " if a.split == "train" else ""
    print(f"{tag}split {a.split}, crop {a.crop}, {len(cases)} cases, {len(patho)} pathological | script sha256 {file_sha()}")

    def val(arm, seed, c, name):
        return float(R[(arm, seed)][c][f"dice_{name}"])

    for name in CLASSES:
        print(f"\n{name}: Dice per seed (mean over cases with the class in the expert mask), then mean +- std over seeds")
        for arm in a.arms:
            per = []
            for seed in a.seeds:
                v = [val(arm, seed, c, name) for c in (patho if name in ("infarct", "mvo") else cases)]
                v = [x for x in v if not np.isnan(x)]
                per.append(np.mean(v))
            print(f"  arm {arm}: " + "  ".join(f"s{s} {x:.4f}" for s, x in zip(a.seeds, per)) + f"  | {np.mean(per):.4f} +- {np.std(per, ddof=1):.4f}")
    if 1 in a.arms and 2 in a.arms:
        d = np.array([[val(2, s, c, "infarct") - val(1, s, c, "infarct") for s in a.seeds] for c in patho])
        print(f"\nPRIMARY: infarct Dice, arm 2 minus arm 1, {len(patho)} pathological cases, paired by seed")
        for j, s in enumerate(a.seeds):
            print(f"  seed {s}: mean difference {np.nanmean(d[:, j]):+.4f}")
        per_case = np.nanmean(d, axis=1)
        keep = per_case[~np.isnan(per_case)]
        rng = np.random.RandomState(BOOT_SEED)
        boots = np.array([rng.choice(keep, len(keep), replace=True).mean() for _ in range(BOOT_N)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  mean over cases (seed-averaged) {keep.mean():+.4f}, 95% bootstrap interval [{lo:+.4f}, {hi:+.4f}] "
              f"({BOOT_N} resamples of {len(keep)} cases, RandomState({BOOT_SEED}))")
        if a.split == "test":
            print("  improvement only if the interval excludes 0:", "YES" if lo > 0 or hi < 0 else "no")
        else:
            print("  (no verdict printed: these are training cases)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dice")
    d.add_argument("--arm", type=int, required=True)
    d.add_argument("--seed", type=int, required=True)
    d.add_argument("--split", choices=["train", "test"], required=True)
    d.add_argument("--crop", choices=["labelbox", "centre"], default="labelbox")
    d.add_argument("--cases", type=int, default=0, help="first N cases only (for tests)")
    d.add_argument("--pathological", action="store_true", help="with --cases: take pathological cases only")
    d.add_argument("--results", default=os.path.join(config.NNUNET_DIR, "volume_results"))
    d.add_argument("--out", default=os.path.join(config.EXPERIMENT_OUT_DIR, "segmentation"))
    d.add_argument("--device", default="cpu")
    d.add_argument("--confirm", default="")
    s = sub.add_parser("summary")
    s.add_argument("--split", choices=["train", "test"], required=True)
    s.add_argument("--crop", choices=["labelbox", "centre"], default="labelbox")
    s.add_argument("--arms", type=int, nargs="+", default=[1, 2, 3])
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    s.add_argument("--out", default=os.path.join(config.EXPERIMENT_OUT_DIR, "segmentation"))
    a = ap.parse_args()
    {"dice": cmd_dice, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
