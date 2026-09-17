"""
scan_real_consistency.py

Checks slice-to-slice anatomical consistency for a real EMIDEC case --
establishes the real-world baseline to compare synthetic cases against.

Usage (run from your EMIDEC folder):
    python3 scan_real_consistency.py Case_P019
"""

import sys
import os
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slice_consistency_metrics import compute_slice_profile, compute_consistency_metrics


def load_real_mask(case_id: str, root: str = ".") -> np.ndarray:
    path = os.path.join(root, case_id, "Contours", f"{case_id}.nii.gz")
    return np.asarray(nib.load(path).get_fdata()).round().astype(int)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scan_real_consistency.py <Case_ID> [Case_ID2 ...]")
        sys.exit(1)

    case_ids = sys.argv[1:]
    for case_id in case_ids:
        try:
            mask = load_real_mask(case_id)
        except Exception as e:
            print(f"{case_id}: [ERROR] {e}")
            continue

        profile = compute_slice_profile(mask)
        metrics = compute_consistency_metrics(profile)

        print(f"\n=== {case_id} ({mask.shape[2]} total slices) ===")
        if metrics is None:
            print("  Not enough slices with a visible cavity to check.")
            continue
        for k, v in metrics.items():
            print(f"  {k}: {v}")
