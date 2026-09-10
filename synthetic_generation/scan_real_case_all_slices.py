"""
scan_real_case_all_slices.py

Matches check_all_slices.py (the synthetic-side scanner) but for real EMIDEC
cases -- runs contiguity across every slice with infarct, not just one
slice, so the real-vs-synthetic comparison is fully apples-to-apples
(whole-volume average vs whole-volume average, not one slice vs one slice).

Usage:
    python scan_real_case_all_slices.py Case_P019
"""

import sys
import nibabel as nib
import numpy as np

from verify_slice_claims import angular_coverage, contiguity_score, DEFAULT_LABEL_MAP, load_slice_mask


def scan_all_slices(case_id: str, root: str = "."):
    import os
    mask_path = os.path.join(root, case_id, "Contours", f"{case_id}.nii.gz")
    img = nib.load(mask_path)
    data = np.asarray(img.get_fdata()).round().astype(int)
    n_slices = data.shape[2]

    results = []
    for s in range(n_slices):
        mask_2d = data[:, :, s]
        cavity_ys, cavity_xs = np.where(mask_2d == DEFAULT_LABEL_MAP["lv_cavity"])
        if len(cavity_ys) == 0:
            continue
        center = (cavity_ys.mean(), cavity_xs.mean())
        infarct_present = (mask_2d == DEFAULT_LABEL_MAP["infarct"]).any()
        if not infarct_present:
            continue
        cscore = contiguity_score(mask_2d, DEFAULT_LABEL_MAP["infarct"], center)
        results.append({"slice": s, "infarct_contiguity": cscore["contiguity"]})

    return {"case_id": case_id, "total_slices": n_slices, "slices_with_infarct": results}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scan_real_case_all_slices.py <Case_ID>")
        sys.exit(1)

    case_id = sys.argv[1]
    result = scan_all_slices(case_id)

    print(f"\n=== {result['case_id']}: all slices with infarct (out of {result['total_slices']} total) ===\n")
    if not result["slices_with_infarct"]:
        print("No slices with infarct found at all.")
    else:
        for entry in result["slices_with_infarct"]:
            c = entry["infarct_contiguity"]
            c_str = f"{c:.3f}" if c is not None else "N/A (too small a span)"
            print(f"slice {entry['slice']}: contiguity = {c_str}")

        valid_scores = [e["infarct_contiguity"] for e in result["slices_with_infarct"] if e["infarct_contiguity"] is not None]
        if valid_scores:
            print(f"\nAverage across all slices with infarct: {sum(valid_scores)/len(valid_scores):.3f}")
            print(f"Range: {min(valid_scores):.3f} to {max(valid_scores):.3f}")
