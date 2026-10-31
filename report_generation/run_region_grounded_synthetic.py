"""
run_region_grounded_synthetic.py

Generates region-grounded reports for SYNTHETIC cases, reusing the exact
prompt scaffold and Gemini-calling logic from run_region_grounded_poc.py.
Uses a stub {"patient_id": case_id} instead of patients/<id>.json: confirmed
safe since include_measurements=False means gemini_region_grounded_report
never reads `record`. Synthetic cases genuinely have no clinical metadata --
this script keeps that fact explicit rather than faking a clinical record.
"""
import os, sys, json, glob, time
from google import genai
from prompts_region_grounded import gemini_region_grounded_report, check_region_grounding

from gemini_client import call_gemini, generation_settings, make_client


def find_one_overlay_pair(case_id: str):
    overlay_dir = os.path.join("overlays", case_id)
    plains = sorted(glob.glob(os.path.join(overlay_dir, "*_plain.png")))
    overlays = sorted(glob.glob(os.path.join(overlay_dir, "*_overlay.png")))
    if not plains or not overlays:
        return None, None
    mid = len(plains) // 2
    return [plains[mid]], [overlays[mid]]


def run_all(case_ids=None):
    client = make_client()
    if case_ids is None:
        case_ids = sorted(os.path.basename(d) for d in glob.glob(os.path.join("overlays", "*")) if os.path.isdir(d))
    out_dir = "outputs_region_grounded"
    os.makedirs(out_dir, exist_ok=True)
    for case_id in case_ids:
        out_path = os.path.join(out_dir, f"{case_id}.json")
        if os.path.exists(out_path):
            print(f"{case_id}: [SKIP] already done")
            continue
        plain, overlay = find_one_overlay_pair(case_id)
        if plain is None:
            print(f"{case_id}: [SKIP] no overlays found, run overlay_renderer_synthetic.py first")
            continue
        print(f"\n=== {case_id} ===")
        try:
            p = gemini_region_grounded_report({"patient_id": case_id}, plain, overlay)
            text, model = call_gemini(client, p["system"], p["user_text"], p["image_paths"])
            check = check_region_grounding(text)
            result = {"patient_id": case_id, "case_type": "synthetic", "model": model, "generation_settings": generation_settings(), "slice_used": plain[0], "output": text, "grounding_check": check}
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  [OK] {check['n_region_tags_found']} tags, fully_tagged={check['likely_fully_tagged']}")
        except Exception as e:
            print(f"  [ERROR] {e}")
        time.sleep(15)
    print(f"\nDone. See {out_dir}/<case_id>.json")


if __name__ == "__main__":
    run_all(sys.argv[1:] or None)
