"""
spot_check_reports.py

Checks specific cases' newly generated reports against geometry we already
independently established earlier in this project, rather than starting
from scratch. Known ground truth being reused here:

  Case_P019: infarct covers ~84.4% of the ring (near-circumferential),
             MVO IS present, positioned toward one edge (0.81), not central
  Case_P004: infarct covers ~29.7% of the ring (localized, NOT
             near-circumferential), MVO is NOT present on the checked slice

Usage (run from your EMIDEC folder):
    python3 spot_check_reports.py
"""

import os
import json
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "verification"))
try:
    from verify_slice_claims import analyze_slice, fact_check_claims, contiguity_score
except ImportError:
    print("Couldn't import verify_slice_claims.py -- make sure it's either in this")
    print("folder or in a 'verification' subfolder next to this script.")
    sys.exit(1)

CASES_TO_CHECK = ["Case_P019", "Case_P004", "Case_P055", "Case_P060"]


def get_slice_idx_from_report(report_path: str) -> int:
    with open(report_path) as f:
        data = json.load(f)
    slice_str = data.get("slice_used", "")
    match = re.search(r"slice_(\d+)", slice_str)
    if not match:
        return None
    return int(match.group(1))


def main():
    for case_id in CASES_TO_CHECK:
        report_path = os.path.join("outputs_region_grounded", f"{case_id}.json")
        if not os.path.exists(report_path):
            print(f"{case_id}: no report found, skipping")
            continue

        with open(report_path) as f:
            report_data = json.load(f)

        slice_idx = get_slice_idx_from_report(report_path)
        print(f"\n{'='*70}")
        print(f"{case_id}  (model: {report_data.get('model')}, slice: {slice_idx})")
        print('='*70)
        print("\n--- Generated report ---")
        print(report_data.get("output", "(no output found)"))

        if slice_idx is None:
            print("\n[Couldn't determine which slice this report used, skipping geometry check]")
            continue

        try:
            analysis = analyze_slice(case_id, slice_idx)
            print("\n--- Independently verified geometry (ground truth) ---")
            for line in fact_check_claims(analysis):
                print(f"  - {line}")
        except Exception as e:
            print(f"\n[Geometry check failed: {e}]")

    print(f"\n{'='*70}")
    print("Read each report above against its geometry check. Look specifically for:")
    print("  - Does the report's infarct extent description match the real % coverage?")
    print("  - Does the report correctly say MVO present/absent?")
    print("  - If MVO is present, does any positioning claim (central/edge) match?")


if __name__ == "__main__":
    main()
