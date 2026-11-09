"""
make_arm_datasets.py

Builds the nnU-Net raw datasets for the insert arms in NNUNET_DIR/nnUNet_raw
(after make_arm1_dataset.py has written Dataset501):

  Dataset502_EMIDECArm2   80 real + the 197 inserts marked `included` in experiments/insert_list_v2.txt
  Dataset504_EMIDECArm4   80 real + the 154 of them with contrast_all >= 0.099 (the weakest real lesion)
  Dataset505_EMIDECArm5   80 real + a seeded random 154 of the 197 (control for arm 4)
(arm 3 trains on Dataset501 with a different trainer; arm 1 on Dataset501.)

Insert volumes are LeFusion's OUTPUT image (already [-1, 1], 72 x 72 x 10) and output mask
(transposed from [D, H, W] to [H, W, D]); case ids are '<heart>__insPnn', which the trainer
recognises as synthetic by '__ins'. All insert sources are train patients (checked here).
The chosen file lists are written to experiments/nnunet/arm_lists.json.

    python experiments/nnunet/make_arm_datasets.py
"""

import os
import sys
import csv
import json
import shutil

import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import common
from common import config, bs, load_split, source_patient, load_nii

BRIGHT_MIN = 0.099
ARMS = {502: "Dataset502_EMIDECArm2", 504: "Dataset504_EMIDECArm4", 505: "Dataset505_EMIDECArm5"}


def main():
    split = load_split()
    train, test = set(split["train"]), set(split["test"])
    included = [l.split("\t")[0] for l in open(os.path.join(common.HERE, "insert_list_v2.txt")) if "\tincluded" in l]
    assert len(included) == 197
    manifest = {m["file"]: m for m in json.load(open(os.path.join(config.FULL_INSERT_DIR, "MANIFEST.json")))["files"]}
    for f in included:
        assert source_patient(f) in train and manifest[f]["lesion_from"] in train and source_patient(f) not in test
    contrast = {r["file"]: float(r["contrast_all"]) for r in csv.DictReader(
        open(os.path.join(config.EXPERIMENT_OUT_DIR, "contrast_inserts_200.csv")))}
    bright = [f for f in included if contrast[f] >= BRIGHT_MIN]
    rng = np.random.RandomState(config.MASK_DIVERSITY_SEED)
    random154 = sorted(rng.choice(included, size=len(bright), replace=False).tolist())
    lists = {502: included, 504: bright, 505: random154}
    assert len(bright) == 154

    job_of = {}
    for job in common.JOBS:
        for f in os.listdir(f"{config.FULL_INSERT_DIR}/output_{job}/Image"):
            job_of[f] = job
    real_raw = os.path.join(config.NNUNET_DIR, "nnUNet_raw", config.NNUNET_DATASET_NAME)
    for did, name in ARMS.items():
        raw = os.path.join(config.NNUNET_DIR, "nnUNet_raw", name)
        shutil.rmtree(raw, ignore_errors=True)
        for sub in ("imagesTr", "labelsTr"):
            os.makedirs(os.path.join(raw, sub))
            for f in os.listdir(os.path.join(real_raw, sub)):
                shutil.copy(os.path.join(real_raw, sub, f), os.path.join(raw, sub, f))
        for f in lists[did]:
            job = job_of[f]
            cid = f.replace(".nii.gz", "")
            img = load_nii(f"{config.FULL_INSERT_DIR}/output_{job}/Image/{f}").astype(np.float32)
            lab = np.transpose(load_nii(f"{config.FULL_INSERT_DIR}/output_{job}/Mask/{f}"), (1, 2, 0)).astype(np.uint8)
            assert img.shape == (72, 72, 10) == lab.shape and img.min() >= -1.001 and img.max() <= 1.001
            nib.save(nib.Nifti1Image(img, bs.LEFUSION_AFFINE), os.path.join(raw, "imagesTr", f"{cid}_0000.nii.gz"))
            nib.save(nib.Nifti1Image(lab, bs.LEFUSION_AFFINE), os.path.join(raw, "labelsTr", f"{cid}.nii.gz"))
        ds = json.load(open(os.path.join(real_raw, "dataset.json")))
        ds["numTraining"] = 80 + len(lists[did])
        json.dump(ds, open(os.path.join(raw, "dataset.json"), "w"), indent=1)
        print(f"{name}: 80 real + {len(lists[did])} inserts = {ds['numTraining']}")
    json.dump({"arm2_included_197": included, "arm4_bright_154": bright, "arm5_random_154": random154,
               "bright_min_contrast": BRIGHT_MIN, "random_seed": config.MASK_DIVERSITY_SEED},
              open(os.path.join(HERE, "arm_lists.json"), "w"), indent=1)
    print("arm4 vs arm5 overlap:", len(set(bright) & set(random154)), "of 154")


if __name__ == "__main__":
    main()
