"""
scan_all_cases.py

Scans the full local EMIDEC training set and builds one summary CSV covering
every case's measurements + key clinical fields, so you can pick the 5-10
representative cases from real numbers instead of guessing.

Place this in the same folder as measurements.py and patient_schema.py, then
run it from inside your EMIDEC root folder:

    python scan_all_cases.py

Output: emidec_case_summary.csv in the current directory.
"""

import os
import csv
import glob
from measurements import extract_measurements
from patient_schema import parse_emidec_clinical_txt


def find_cases(root="."):
    """Finds all Case_NXXX / Case_PXXX folders and their matching clinical .txt files."""
    case_dirs = sorted(glob.glob(os.path.join(root, "Case_[NP]*")))
    cases = []
    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)  # e.g. "Case_P001"
        group = "normal" if "_N" in case_id else "pathological"

        mask_path = os.path.join(case_dir, "Contours", f"{case_id}.nii.gz")
        # clinical txt files use a space instead of underscore: "Case P001.txt"
        clinical_txt_name = case_id.replace("_", " ") + ".txt"
        clinical_path = os.path.join(root, clinical_txt_name)

        if not os.path.exists(mask_path):
            print(f"WARNING: mask not found for {case_id}, skipping: {mask_path}")
            continue
        if not os.path.exists(clinical_path):
            print(f"WARNING: clinical file not found for {case_id}, skipping: {clinical_path}")
            continue

        cases.append({
            "case_id": case_id,
            "group": group,
            "mask_path": mask_path,
            "clinical_path": clinical_path,
        })
    return cases


def build_summary(root="."):
    cases = find_cases(root)
    rows = []
    for c in cases:
        try:
            m = extract_measurements(mask_path=c["mask_path"], patient_id=c["case_id"])
            cl = parse_emidec_clinical_txt(c["clinical_path"])
        except Exception as e:
            print(f"ERROR processing {c['case_id']}: {e}")
            continue

        rows.append({
            "case_id": c["case_id"],
            "group": c["group"],
            "infarct_present": m.infarct_present,
            "mvo_present": m.mvo_present,
            "infarct_volume_ml": m.infarct_volume_ml,
            "infarct_pct_of_myocardium": m.infarct_percentage_of_myocardium,
            "mvo_volume_ml": m.mvo_volume_ml,
            "mvo_pct_of_infarct": m.mvo_percentage_of_infarct,
            "n_infarct_slices": len(m.infarct_affected_slices),
            "n_mvo_slices": len(m.mvo_affected_slices),
            "total_slices": m.total_slices,
            "lv_cavity_volume_ml": m.lv_cavity_volume_ml,
            "myocardium_total_volume_ml": m.myocardium_total_volume_ml,
            "age": cl.age,
            "sex": cl.sex,
            "troponin": cl.troponin,
            "killip_max": cl.killip_max,
            "lvef_echo_percent": cl.lvef_echo_percent,
            "ecg_stemi": cl.ecg_stemi,
        })
    return rows


def write_csv(rows, out_path="emidec_case_summary.csv"):
    if not rows:
        print("No rows to write.")
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


def print_selection_suggestions(rows):
    """Prints candidate cases for each diversity criterion, to speed up manual picking."""
    pathological = [r for r in rows if r["group"] == "pathological"]
    normal = [r for r in rows if r["group"] == "normal"]

    with_mvo = [r for r in pathological if r["mvo_present"]]
    without_mvo = [r for r in pathological if not r["mvo_present"]]

    by_infarct_pct = sorted(pathological, key=lambda r: r["infarct_pct_of_myocardium"])

    print("\n=== Selection candidates ===")
    if normal:
        print(f"Normal case (no pathology): {normal[0]['case_id']}")
    if by_infarct_pct:
        print(f"Smallest infarct (pathological): {by_infarct_pct[0]['case_id']} "
              f"({by_infarct_pct[0]['infarct_pct_of_myocardium']}% of myocardium)")
        print(f"Largest infarct (pathological): {by_infarct_pct[-1]['case_id']} "
              f"({by_infarct_pct[-1]['infarct_pct_of_myocardium']}% of myocardium)")
    if with_mvo:
        print(f"MVO present example(s): {[r['case_id'] for r in with_mvo[:3]]}")
    if without_mvo:
        print(f"MVO absent (infarct, no MVO) example(s): {[r['case_id'] for r in without_mvo[:3]]}")

    mid = by_infarct_pct[len(by_infarct_pct)//2] if by_infarct_pct else None
    if mid:
        print(f"Mid-range infarct example: {mid['case_id']} "
              f"({mid['infarct_pct_of_myocardium']}% of myocardium)")

    print(
        "\nSuggested 8-case starting set: 1 normal + smallest infarct + largest infarct + "
        "mid-range infarct + 2 with MVO + 2 without MVO (adjust for overlap -- e.g. the "
        "largest infarct case likely already has MVO)."
    )


if __name__ == "__main__":
    rows = build_summary(root=".")
    write_csv(rows)
    print_selection_suggestions(rows)
