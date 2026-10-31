"""
make_split.py

Builds the patient-level train/test split and writes splits/split_v1.json.

Every patient gets three flags read from the real mask:
    normal   -- EMIDEC normal group (case id starts with N)
    infarct  -- label 3 present
    mvo      -- label 4 present

Strata: normal / infarct without MVO / infarct with MVO. The test set takes
round(TEST_FRACTION * size) patients from each stratum, using a RandomState
seeded with config.SPLIT_SEED, so the same inputs always give the same split.

Usage (EMIDEC root = folder holding Case_* directories):
    python splits/make_split.py /path/to/emidec-dataset-1.0.1
"""

import os
import sys
import json
import glob
import numpy as np
import nibabel as nib

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)
import config

OUT_PATH = os.path.join(REPO_ROOT, "splits", "split_v1.json")
LABEL_INFARCT, LABEL_MVO = 3, 4


def patient_flags(emidec_root: str) -> dict:
    flags = {}
    for case_dir in sorted(glob.glob(os.path.join(emidec_root, "Case_[NP]*"))):
        case_id = os.path.basename(case_dir)
        mask_path = os.path.join(case_dir, "Contours", f"{case_id}.nii.gz")
        if not os.path.exists(mask_path):
            print(f"WARNING: no mask for {case_id}, skipped")
            continue
        mask = np.asarray(nib.load(mask_path).dataobj).round().astype(int)
        flags[case_id] = {
            "normal": case_id.startswith("Case_N"),
            "infarct": bool((mask == LABEL_INFARCT).any()),
            "mvo": bool((mask == LABEL_MVO).any()),
        }
    return flags


def stratum_of(f: dict) -> str:
    if f["normal"]:
        return "normal"
    return "infarct_mvo" if f["mvo"] else "infarct_no_mvo"


def build_split(flags: dict, seed: int, test_fraction: float) -> dict:
    strata = {}
    for case_id, f in flags.items():
        strata.setdefault(stratum_of(f), []).append(case_id)
    rng = np.random.RandomState(seed)
    test = []
    counts = {}
    for name in sorted(strata):
        members = sorted(strata[name])
        n_test = int(round(test_fraction * len(members)))
        picked = sorted(rng.choice(members, size=n_test, replace=False).tolist())
        test += picked
        counts[name] = {"total": len(members), "test": n_test, "train": len(members) - n_test}
    test = sorted(test)
    train = sorted(c for c in flags if c not in set(test))
    return {
        "version": "split_v1",
        "seed": seed,
        "test_fraction": test_fraction,
        "strata_counts": counts,
        "n_test": len(test),
        "n_train": len(train),
        "test": test,
        "train": train,
        "patients": {c: {**flags[c], "stratum": stratum_of(flags[c])} for c in sorted(flags)},
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    flags = patient_flags(sys.argv[1])
    split = build_split(flags, config.SPLIT_SEED, config.TEST_FRACTION)
    with open(OUT_PATH, "w") as f:
        json.dump(split, f, indent=2)
    print(f"{len(flags)} patients -> {split['n_train']} train, {split['n_test']} test (seed {split['seed']})")
    for name, c in split["strata_counts"].items():
        print(f"  {name:15s} total={c['total']:3d}  test={c['test']:2d}  train={c['train']:3d}")
    print("test set:", ", ".join(split["test"]))
    print(f"wrote {OUT_PATH}")
