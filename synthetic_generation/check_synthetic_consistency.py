"""
check_synthetic_consistency.py

Checks slice-to-slice anatomical consistency for a synthetic case, using
the exact same metric as scan_real_consistency.py -- directly comparable
numbers.

Usage:
    python3 check_synthetic_consistency.py Case_P001.nii.gz
"""

import sys
import modal

if len(sys.argv) < 2:
    print("Usage: python3 check_synthetic_consistency.py <case_filename>")
    sys.exit(1)

case_id = sys.argv[1]
fn = modal.Function.from_name("lefusion-emidec-poc", "check_synthetic_consistency")
result = fn.remote(case_id=case_id)

if "error" in result:
    print(f"Error: {result['error']}")
    sys.exit(1)

print(f"\n=== {result['case_id']} ({result['total_slices']} total slices) ===")
for k, v in result.items():
    if k not in ("case_id", "total_slices"):
        print(f"  {k}: {v}")
