"""
run_region_grounded_poc.py

Generates the corrected proof of concept the supervisor asked for: MRI + segmentation
mask overlay -> Gemini -> report where every Findings sentence is explicitly
tagged to a real visual region, not a measurement field.

Run AFTER: patients/*.json exist (build_patient_jsons.py) and overlays/*
exist (overlay_renderer.py) for at least a few cases -- doesn't need all 8,
the supervisor asked for "several examples," not full coverage.

Usage:
    python run_region_grounded_poc.py
"""

import os
import json
import glob
import time
from google import genai

from prompts_region_grounded import gemini_region_grounded_report, check_region_grounding

from gemini_client import call_gemini, generation_settings, make_client


def find_one_overlay_pair(case_id: str):
    """Picks ONE representative slice (plain + overlay pair) per case -- the supervisor asked
    for example outputs, not a full per-slice sweep yet."""
    overlay_dir = os.path.join("overlays", case_id)
    plains = sorted(glob.glob(os.path.join(overlay_dir, "*_plain.png")))
    overlays = sorted(glob.glob(os.path.join(overlay_dir, "*_overlay.png")))
    if not plains or not overlays:
        return None, None
    mid = len(plains) // 2
    return [plains[mid]], [overlays[mid]]


def run_all(case_ids: list = None):
    client = make_client()

    if case_ids is None:
        patient_files = sorted(glob.glob(os.path.join("patients", "*.json")))
        case_ids = [os.path.splitext(os.path.basename(f))[0] for f in patient_files]

    out_dir = "outputs_region_grounded"
    os.makedirs(out_dir, exist_ok=True)

    for case_id in case_ids:
        with open(os.path.join("patients", f"{case_id}.json")) as f:
            record = json.load(f)

        plain, overlay = find_one_overlay_pair(case_id)
        if plain is None:
            print(f"{case_id}: [SKIP] no overlay images found, run overlay_renderer.py first")
            continue

        out_path = os.path.join(out_dir, f"{case_id}.json")
        if os.path.exists(out_path):
            print(f"{case_id}: [SKIP] already done")
            continue

        print(f"\n=== {case_id} ===  (slice: {os.path.basename(plain[0])})")
        try:
            p = gemini_region_grounded_report(record, plain, overlay)
            text, model = call_gemini(client, p["system"], p["user_text"], p["image_paths"])
            check = check_region_grounding(text)

            result = {
                "patient_id": case_id,
                "model": model,
                "generation_settings": generation_settings(),
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

        time.sleep(15)

    print(f"\nDone. Check {out_dir}/<case_id>.json for the example reports and grounding checks.")


if __name__ == "__main__":
    run_all()
