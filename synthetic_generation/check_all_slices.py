"""
check_all_slices.py

Checks contiguity across EVERY slice of a synthetic case that has infarct,
not just one auto-picked slice -- to see whether a low score is a
consistent pattern or a one-slice anomaly.

Usage:
    python check_all_slices.py Case_P001.nii.gz
"""

import sys
import modal

if len(sys.argv) < 2:
    print("Usage: python check_all_slices.py <case_filename>")
    sys.exit(1)

case_id = sys.argv[1]
fn = modal.Function.from_name("lefusion-emidec-poc", "scan_all_slices_geometry")
result = fn.remote(case_id=case_id)

if "error" in result:
    print(f"Error: {result['error']}")
    sys.exit(1)

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
