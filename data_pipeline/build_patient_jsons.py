"""
build_patient_jsons.py

Builds one real, verified patient JSON per case, using the now-fixed
measurements.py + patient_schema.py.

Run from inside your EMIDEC root folder:
    python build_patient_jsons.py              (all cases found)
    python build_patient_jsons.py Case_P001 Case_N006   (just specific ones)

Output: ./patients/<case_id>.json, one per case.
"""

import os
import sys
import glob
from measurements import extract_measurements
from patient_schema import parse_emidec_clinical_txt, build_patient_record


def find_all_case_ids(root: str = ".") -> list:
    """Discovers every real case by looking for <root>/Case_XXXX/Contours/
    folders -- this is how we know a case is genuinely present and complete,
    rather than assuming a fixed list. Works for both N (normal) and P
    (pathological) cases."""
    case_dirs = sorted(glob.glob(os.path.join(root, "Case_*")))
    case_ids = []
    for d in case_dirs:
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "Contours")):
            case_ids.append(os.path.basename(d))
    return case_ids


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

    case_ids = sys.argv[1:] if len(sys.argv) > 1 else find_all_case_ids()

    print(f"Building {len(case_ids)} patient records...\n")
    n_ok, n_failed = 0, 0
    for case_id in case_ids:
        out_path = os.path.join("patients", f"{case_id}.json")
        if os.path.exists(out_path):
            print(f"{case_id}: [SKIP] already done")
            n_ok += 1
            continue
        try:
            record = build_one(case_id)
            record.save(out_path)
            m = record.measurements
            print(
                f"{case_id}: infarct={m['infarct_present']} "
                f"({m['infarct_percentage_of_myocardium']}%), "
                f"mvo={m['mvo_present']}, sex={record.clinical.sex}, "
                f"age={record.clinical.age}  -> saved {out_path}"
            )
            n_ok += 1
        except Exception as e:
            print(f"ERROR building {case_id}: {e}")
            n_failed += 1

    print(f"\nDone. {n_ok} succeeded, {n_failed} failed, out of {len(case_ids)} total.")
    print("Check the 'patients/' folder -- one JSON per case, ready for the")
    print("prompt scaffolds in prompts.py once overlay images are also in place.")
