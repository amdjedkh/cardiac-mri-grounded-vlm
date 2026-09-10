"""
run_medgemma_region_grounded_poc.py

Same corrected region-grounding task as run_region_grounded_poc.py, but for
MedGemma via the existing Modal deployment, per Carlos's "optionally MedGemma"
note. Reuses modal_medgemma.py -- no redeploy needed if it's already up from
the earlier baseline runs.

Usage:
    python run_medgemma_region_grounded_poc.py
"""

import os
import json
import glob
import modal

from prompts_region_grounded import medgemma_region_grounded_report, check_region_grounding


def find_one_overlay_pair(case_id: str):
    overlay_dir = os.path.join("overlays", case_id)
    plains = sorted(glob.glob(os.path.join(overlay_dir, "*_plain.png")))
    overlays = sorted(glob.glob(os.path.join(overlay_dir, "*_overlay.png")))
    if not plains or not overlays:
        return None, None
    mid = len(plains) // 2
    return [plains[mid]], [overlays[mid]]


def read_bytes(paths: list) -> list:
    out = []
    for p in paths:
        with open(p, "rb") as f:
            out.append(f.read())
    return out


def run_all(case_ids: list = None):
    MedGemma = modal.Cls.from_name("medgemma-emidec-poc", "MedGemma")
    medgemma = MedGemma()

    if case_ids is None:
        patient_files = sorted(glob.glob(os.path.join("patients", "*.json")))
        case_ids = [os.path.splitext(os.path.basename(f))[0] for f in patient_files]

    out_dir = "outputs_region_grounded_medgemma"
    os.makedirs(out_dir, exist_ok=True)

    for case_id in case_ids:
        with open(os.path.join("patients", f"{case_id}.json")) as f:
            record = json.load(f)

        plain, overlay = find_one_overlay_pair(case_id)
        if plain is None:
            print(f"{case_id}: [SKIP] no overlay images found")
            continue

        out_path = os.path.join(out_dir, f"{case_id}.json")
        if os.path.exists(out_path):
            print(f"{case_id}: [SKIP] already done")
            continue

        print(f"\n=== {case_id} ===  (slice: {os.path.basename(plain[0])})")
        try:
            p = medgemma_region_grounded_report(record, plain, overlay)
            img_bytes = read_bytes(p["image_paths"])
            text = medgemma.generate.remote(p["system"], p["user_text"], img_bytes)
            check = check_region_grounding(text)

            result = {
                "patient_id": case_id,
                "model": "medgemma-4b-it",
                "slice_used": plain[0],
                "output": text,
                "grounding_check": check,
            }
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            print(f"  [OK] {check['n_region_tags_found']} region tags found, "
                  f"fully tagged: {check['likely_fully_tagged']}, "
                  f"invalid region names: {check['invalid_region_names']}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"\nDone. Check {out_dir}/<case_id>.json for results.")


if __name__ == "__main__":
    run_all()
