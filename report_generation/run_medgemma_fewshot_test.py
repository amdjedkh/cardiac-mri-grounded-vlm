"""
run_medgemma_fewshot_test.py

Tests the few-shot prompt variant against a few real cases, to see if it
actually fixes MedGemma's grounding format compliance (Karen's hypothesis)
or if the problem persists (suggesting a real capability gap instead).

Usage:
    python run_medgemma_fewshot_test.py
"""

import os
import json
import glob
import modal

from prompts_region_grounded import check_region_grounding
from prompts_region_grounded_fewshot import medgemma_region_grounded_report_fewshot


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
        # test on a small subset first -- cheap way to check the hypothesis
        # before spending compute re-running all 8
        case_ids = ["Case_N006", "Case_P019", "Case_P004"]

    out_dir = "outputs_medgemma_fewshot_test"
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
            p = medgemma_region_grounded_report_fewshot(record, plain, overlay)
            img_bytes = read_bytes(p["image_paths"])
            text = medgemma.generate.remote(p["system"], p["user_text"], img_bytes)
            check = check_region_grounding(text)

            result = {"patient_id": case_id, "model": "medgemma-4b-it-fewshot",
                      "output": text, "grounding_check": check}
            with open(os.path.join(out_dir, f"{case_id}.json"), "w") as f:
                json.dump(result, f, indent=2)

            print(f"  fully_tagged: {check['likely_fully_tagged']}  "
                  f"tags_outside_findings: {bool(check['tags_outside_findings'])}  "
                  f"duplicated_impression: {check['duplicated_impression']}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"\nDone. Compare against outputs_region_grounded_medgemma/ (the original prompt)")
    print("to see whether the few-shot example actually improved compliance.")


if __name__ == "__main__":
    run_test()
