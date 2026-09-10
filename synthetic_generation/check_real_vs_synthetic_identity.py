"""
check_real_vs_synthetic_identity.py

Directly checks whether a "synthetic" mask is actually just the real
conditioning mask copied through unchanged, pixel by pixel -- triggered by
real and "synthetic" P004 producing byte-for-byte identical contiguity
numbers, which a real generation process would not do by chance.

Usage:
    python check_real_vs_synthetic_identity.py Case_P004.nii.gz
"""

import sys
import modal

if len(sys.argv) < 2:
    print("Usage: python check_real_vs_synthetic_identity.py <case_filename>")
    sys.exit(1)

case_id = sys.argv[1]
fn = modal.Function.from_name("lefusion-emidec-poc", "compare_synthetic_to_real_mask")
result = fn.remote(case_id=case_id)

print(f"\n=== Real vs synthetic mask identity check: {case_id} ===\n")
for k, v in result.items():
    print(f"{k}: {v}")
