"""
compare_two_real_cases.py

Compares two DIFFERENT real patients' masks using the exact same method
(resample both to a common shape, compare pixel-for-pixel) used to check
real P004 against synthetic P004. This tells us how much two totally
unrelated hearts naturally overlap just from general anatomical similarity
-- the baseline needed to judge whether 73% (real P004 vs synthetic P004)
is meaningfully high, unremarkable, or low.

Usage:
    python compare_two_real_cases.py Case_P004 Case_P019
"""

import sys
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def load_full_mask(case_id: str, root: str = "."):
    path = os.path.join(root, case_id, "Contours", f"{case_id}.nii.gz")
    return np.asarray(nib.load(path).get_fdata()).round().astype(int)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_two_real_cases.py <Case_A> <Case_B>")
        print("Example: python compare_two_real_cases.py Case_P004 Case_P019")
        sys.exit(1)

    case_a, case_b = sys.argv[1], sys.argv[2]
    data_a = load_full_mask(case_a)
    data_b = load_full_mask(case_b)

    print(f"\n=== Comparing {case_a} vs {case_b} (two different real patients) ===\n")
    print(f"{case_a} native shape: {data_a.shape}")
    print(f"{case_b} native shape: {data_b.shape}")

    # resample both to the same common shape used for the synthetic
    # comparison (72, 72, 10), so this is directly comparable
    common_shape = (72, 72, 10)
    zoom_a = [c / n for c, n in zip(common_shape, data_a.shape)]
    zoom_b = [c / n for c, n in zip(common_shape, data_b.shape)]
    resampled_a = zoom(data_a, zoom_a, order=0)
    resampled_b = zoom(data_b, zoom_b, order=0)

    matching_fraction = float((resampled_a == resampled_b).mean())
    print(f"\nFraction matching pixels after resampling both to {common_shape}: {matching_fraction:.4f}")

    dist_a = {int(l): int((resampled_a == l).sum()) for l in np.unique(resampled_a)}
    dist_b = {int(l): int((resampled_b == l).sum()) for l in np.unique(resampled_b)}
    print(f"{case_a} label distribution: {dist_a}")
    print(f"{case_b} label distribution: {dist_b}")

    print(f"\nFor reference: real P004 vs synthetic P004 matched at 0.7324")
    print(f"This unrelated-patient baseline: {matching_fraction:.4f}")
