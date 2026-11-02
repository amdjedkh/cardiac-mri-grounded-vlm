"""
common.py

Shared loading for the LeFusion insert-run checks (validate_full.py,
contrast_report.py, montage.py). Paths come from config.py (env vars
EMIDEC_ROOT, FULL_INSERT_DIR, EXPERIMENT_OUT_DIR).

Array conventions: staged inputs and real EMIDEC arrays are [H, W, D];
LeFusion output masks are [D, H, W] and are transposed to [H, W, D] here.
LeFusion rescales each input volume to [-1, 1] (min-max) and then centre
pads / crops to 72 x 72 x 10; the output image is on that same scale.
"""

import os
import sys
import glob

import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")
for sub in ("", "splits", "synthetic_generation"):
    sys.path.insert(0, os.path.join(REPO_ROOT, sub))
import config
import mask_diversity as md
import build_staged_test as bs
from split_utils import load_split, source_patient

JOBS = (1, 2, 3)
INFARCT, MVO, WALL = 3, 4, 2


def load_nii(path):
    return np.asarray(nib.load(path).dataobj)


def minmax(img):
    img = img.astype(float)
    return (img - img.min()) / (img.max() - img.min()) * 2 - 1


def find_offset(small, target_shape):
    """(dy, dx, dz) at which `small` sits inside a zero array of target_shape
    after LeFusion's centre padding; checked by the caller via equality."""
    h, w, d = small.shape
    H, W, D = target_shape
    return (H - h) // 2, (W - w) // 2, (D - d) // 2


def place(small, target_shape, offset):
    out = np.zeros(target_shape, small.dtype)
    dy, dx, dz = offset
    h, w, d = small.shape
    out[dy:dy + h, dx:dx + w, dz:dz + d] = small
    return out


def locate(lab_in, out_mask):
    """Offset where the padded input mask equals the output mask, or None.
    Tries the centred offset first, then every offset."""
    H, W, D = out_mask.shape
    h, w, d = lab_in.shape
    first = find_offset(lab_in, out_mask.shape)
    tried = [first] + [(dy, dx, dz) for dz in range(D - d + 1) for dy in range(H - h + 1)
                       for dx in range(W - w + 1) if (dy, dx, dz) != first]
    for off in tried:
        if np.array_equal(place(lab_in, out_mask.shape, off), out_mask):
            return off
    return None


def iter_volumes(root=None):
    """Yield one dict per volume of the insert run, in job order."""
    root = root or config.FULL_INSERT_DIR
    for job in JOBS:
        for path in sorted(glob.glob(f"{root}/output_{job}/Image/*.nii.gz")):
            name = os.path.basename(path)
            img_in = load_nii(f"{root}/input_{job}/images/{name}").astype(float)
            lab_in = load_nii(f"{root}/input_{job}/labels/{name}").astype(int)
            out_img = load_nii(path).astype(float)
            out_mask = np.transpose(load_nii(f"{root}/output_{job}/Mask/{name}").astype(int), (1, 2, 0))
            yield {"job": job, "file": name, "img_in": img_in, "lab_in": lab_in,
                   "out_img": out_img, "out_mask": out_mask}


def contrast(img, mask, lesion_labels):
    """Mean over lesion_labels minus mean over healthy wall (label 2); NaN if either is empty."""
    les, wall = np.isin(mask, lesion_labels), mask == WALL
    if not les.any() or not wall.any():
        return float("nan")
    return float(img[les].mean() - img[wall].mean())


def real_train_cases():
    split = load_split()
    return sorted(c for c in split["train"] if c.startswith("Case_P"))
