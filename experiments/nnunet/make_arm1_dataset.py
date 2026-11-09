"""
make_arm1_dataset.py

Builds the nnU-Net v2 raw dataset for ARM 1 (real train volumes only) in
NNUNET_DIR/nnUNet_raw/Dataset501_EMIDECArm1. All 80 train patients, no validation split
(training uses nnU-Net fold 'all'). No test patient is written.

Every volume is in the shared space: bounding box of labels > 0 + 5 px in
plane, all slices, min-max rescale to [-1, 1] over the crop, then centre pad
(zeros) or crop to 72 x 72 x 10, written with the nominal spacing
1.5625 x 1.5625 x 10 mm (the same as LeFusion's files and the inserts).
Channel name is 'noNorm' so nnU-Net keeps the [-1, 1] values.

    python experiments/nnunet/make_arm1_dataset.py
"""

import os
import sys
import json

import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import common
from common import config, md, bs, load_split


def to_shape(arr, shape, fill):
    """Centre pad with `fill` / centre crop to `shape` (same arithmetic as LeFusion's CropOrPad)."""
    out = np.full(shape, fill, arr.dtype)
    src, dst = [], []
    for n, t in zip(arr.shape, shape):
        if n <= t:
            o = (t - n) // 2
            src.append(slice(0, n)); dst.append(slice(o, o + n))
        else:
            o = (n - t) // 2
            src.append(slice(o, o + t)); dst.append(slice(0, t))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def main():
    split = load_split()
    train, test = list(split["train"]), set(split["test"])
    assert not (set(train) & test) and len(train) == 80
    raw = os.path.join(config.NNUNET_DIR, "nnUNet_raw", config.NNUNET_DATASET_NAME)
    os.makedirs(os.path.join(raw, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(raw, "labelsTr"), exist_ok=True)
    sizes = []
    for cid in sorted(train):
        img, mask, _ = md.load_case(config.EMIDEC_ROOT, cid)
        box = bs.crop_box(mask)
        img_c, mask_c = bs.crop(img, box), bs.crop(mask, box)
        sizes.append(img_c.shape)
        img_n = to_shape(common.minmax(img_c).astype(np.float32), config.CROP_SHAPE, 0.0)
        lab = to_shape(mask_c.astype(np.uint8), config.CROP_SHAPE, 0)
        nib.save(nib.Nifti1Image(img_n, bs.LEFUSION_AFFINE), os.path.join(raw, "imagesTr", f"{cid}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(lab, bs.LEFUSION_AFFINE), os.path.join(raw, "labelsTr", f"{cid}.nii.gz"))
    json.dump({"channel_names": {"0": "noNorm"},
               "labels": {"background": 0, "LV_cavity": 1, "myocardium": 2, "infarct": 3, "MVO": 4},
               "numTraining": len(train), "file_ending": ".nii.gz"},
              open(os.path.join(raw, "dataset.json"), "w"), indent=1)
    h, w, d = zip(*sizes)
    print(f"{len(train)} volumes written to {raw}")
    print(f"crop sizes before padding: H {min(h)}-{max(h)}, W {min(w)}-{max(w)}, D {min(d)}-{max(d)}; all padded/cropped to {config.CROP_SHAPE}")


if __name__ == "__main__":
    main()
