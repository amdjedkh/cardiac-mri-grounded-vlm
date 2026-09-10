"""
run_medgemma_fewshot2_test.py

Tests the two-contrasting-example prompt variant, to see if it stops the
templating behavior found with the single-example version (where MedGemma
copied the one example's wording for both real abnormal cases, including
getting P019's MVO status wrong).

Usage:
    python run_medgemma_fewshot2_test.py
"""

import os
import json
import glob
import modal

from prompts_region_grounded import check_region_grounding
from prompts_region_grounded_fewshot2 import medgemma_region_grounded_report_fewshot2


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


def run_test(case_ids: list = None):
    MedGemma = modal.Cls.from_name("medgemma-emidec-poc", "MedGemma")
    medgemma = MedGemma()

    if case_ids is None:
        case_ids = ["Case_N006", "Case_P019", "Case_P004"]

    out_dir = "outputs_medgemma_fewshot2_test"
    os.makedirs(out_dir, exist_ok=True)

    for case_id in case_ids:
        with open(os.path.join("patients", f"{case_id}.json")) as f:
            record = json.load(f)

        plain, overlay = find_one_overlay_pair(case_id)
        if plain is None:
            print(f"{case_id}: [SKIP] no overlay images found")
            continue

        print(f"\n=== {case_id} ===")
        try:
            p = medgemma_region_grounded_report_fewshot2(record, plain, overlay)
            img_bytes = read_bytes(p["image_paths"])
            text = medgemma.generate.remote(p["system"], p["user_text"], img_bytes)
            check = check_region_grounding(text)

            result = {"patient_id": case_id, "model": "medgemma-4b-it-fewshot2",
                      "output": text, "grounding_check": check}
            with open(os.path.join(out_dir, f"{case_id}.json"), "w") as f:
                json.dump(result, f, indent=2)

            print(f"  fully_tagged: {check['likely_fully_tagged']}")
            print(f"  output preview: {text[:200]}...")
        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"\nDone. Compare outputs_medgemma_fewshot2_test/ against "
          f"outputs_medgemma_fewshot_test/ (single-example version) -- "
          f"check whether P019 now correctly mentions MVO present, and "
          f"whether P019 and P004 no longer read almost identically.")


if __name__ == "__main__":
    run_test()
