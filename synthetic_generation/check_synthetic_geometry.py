"""
check_synthetic_geometry.py

Runs the same contiguity/coverage math already validated on real EMIDEC
cases against a synthetic case, for direct comparison against the real
baselines:
    P019: infarct contiguity 0.823, MVO contiguity 0.817
    P004: infarct contiguity 0.861

Usage:
    python check_synthetic_geometry.py Case_P001.nii.gz
"""

import sys
import modal

if len(sys.argv) < 2:
    print("Usage: python check_synthetic_geometry.py <case_filename>")
    print("Example: python check_synthetic_geometry.py Case_P001.nii.gz")
    sys.exit(1)

case_id = sys.argv[1]

fn = modal.Function.from_name("lefusion-emidec-poc", "analyze_synthetic_geometry")
result = fn.remote(case_id=case_id)

if "error" in result:
    print(f"Error: {result['error']}")
    sys.exit(1)

print(f"\n=== Synthetic geometry check: {result['case_id']}, slice {result['slice_used']} ===\n")
for region in ["infarct", "mvo"]:
    r = result[region]
    if not r["present"]:
        print(f"{region}: not present on this slice")
        continue
    contiguity = r["contiguity"]
    if contiguity is None:
        print(f"{region}: present but too small a span to score")
        continue
    verdict = "smooth/solid" if contiguity >= 0.8 else "fragmented/scattered" if contiguity < 0.5 else "somewhat patchy"
    print(f"{region}: coverage={r['coverage_pct_of_circle']}% of ring, contiguity={contiguity} ({verdict})")

print("\nReal-case baselines for comparison:")
print("  P019: infarct contiguity 0.823, MVO contiguity 0.817")
print("  P004: infarct contiguity 0.861")
