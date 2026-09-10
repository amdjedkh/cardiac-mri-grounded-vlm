"""
build_patient_jsons.py

Builds one real, verified patient JSON per selected case, using the
now-fixed measurements.py + patient_schema.py.

Run from inside your EMIDEC root folder:
    python build_patient_jsons.py

Output: ./patients/<case_id>.json, one per selected case.
"""

import os
from measurements import extract_measurements
from patient_schema import parse_emidec_clinical_txt, build_patient_record

# Final 8-case selection, per the scan results.
SELECTED_CASES = [
    "Case_N006",  # normal
    "Case_P055",  # smallest infarct
    "Case_P019",  # largest infarct, also MVO-present
    "Case_P060",  # mid-range infarct
    "Case_P001",  # MVO present
    "Case_P002",  # MVO present
    "Case_P004",  # MVO absent
    "Case_P007",  # MVO absent
]


def build_one(case_id: str, root: str = "."):
    group = "normal" if "_N" in case_id else "pathological"
    mask_path = os.path.join(root, case_id, "Contours", f"{case_id}.nii.gz")
    clinical_path = os.path.join(root, case_id.replace("_", " ") + ".txt")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    if not os.path.exists(clinical_path):
        raise FileNotFoundError(f"Clinical file not found: {clinical_path}")

    measurements = extract_measurements(mask_path=mask_path, patient_id=case_id)
    clinical = parse_emidec_clinical_txt(clinical_path)
    record = build_patient_record(case_id, measurements, clinical, emidec_group=group)
    return record


if __name__ == "__main__":
    os.makedirs("patients", exist_ok=True)

    print(f"Building {len(SELECTED_CASES)} patient records...\n")
    for case_id in SELECTED_CASES:
        try:
            record = build_one(case_id)
            out_path = os.path.join("patients", f"{case_id}.json")
            record.save(out_path)
            m = record.measurements
            print(
                f"{case_id}: infarct={m['infarct_present']} "
                f"({m['infarct_percentage_of_myocardium']}%), "
                f"mvo={m['mvo_present']}, sex={record.clinical.sex}, "
                f"age={record.clinical.age}  -> saved {out_path}"
            )
        except Exception as e:
            print(f"ERROR building {case_id}: {e}")

    print("\nDone. Check the 'patients/' folder -- one JSON per case, ready for the")
    print("prompt scaffolds in prompts.py once overlay images are also in place.")
