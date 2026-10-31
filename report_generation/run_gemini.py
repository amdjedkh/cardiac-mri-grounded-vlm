"""
run_gemini.py

Runs the two Gemini baselines (measurements-only, full-input) across all
patients in patients/*.json, using the overlays already rendered by
overlay_renderer.py. Saves outputs to outputs/<case_id>/gemini_*.json.

Setup:
    pip install google-genai
    setx GEMINI_API_KEY "your-key-here"      (PowerShell: then restart terminal)
    # or for the current session only:
    $env:GEMINI_API_KEY = "your-key-here"

Usage:
    python run_gemini.py
"""

import os
import json
import glob
import time
from google import genai

from prompts import gemini_measurements_only, gemini_full_input
from gemini_client import call_gemini, make_client

def load_patient(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def find_overlay_images(case_id: str):
    """Matches overlay_renderer.py's output layout."""
    overlay_dir = os.path.join("overlays", case_id)
    plain = sorted(glob.glob(os.path.join(overlay_dir, "*_plain.png")))
    overlay = sorted(glob.glob(os.path.join(overlay_dir, "*_overlay.png")))
    return plain, overlay


def run_all(only: str = "both"):
    client = make_client()

    patient_files = sorted(glob.glob(os.path.join("patients", "*.json")))
    if not patient_files:
        raise RuntimeError("No patient JSON files found in patients/ -- run build_patient_jsons.py first.")

    for pf in patient_files:
        record = load_patient(pf)
        case_id = record["patient_id"]
        out_dir = os.path.join("outputs", case_id)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== {case_id} ===")

        # Baseline 1: measurements only, no images (fast -- run this pass first)
        if only in ("both", "measurements"):
            out1_path = os.path.join(out_dir, "gemini_measurements_only.json")
            if os.path.exists(out1_path):
                print("  [SKIP] gemini_measurements_only already done")
            else:
                try:
                    p1 = gemini_measurements_only(record)
                    text1, model_used = call_gemini(client, p1["system"], p1["user_text"], [])
                    with open(out1_path, "w") as f:
                        json.dump({"model": model_used, "condition": "measurements_only", "output": text1}, f, indent=2)
                    print("  [OK] gemini_measurements_only")
                except Exception as e:
                    print(f"  [ERROR] gemini_measurements_only: {e}")

        # Baseline 2: images + overlays + measurements + metadata (slower -- run separately)
        if only in ("both", "full"):
            out2_path = os.path.join(out_dir, "gemini_full_input.json")
            if os.path.exists(out2_path):
                print("  [SKIP] gemini_full_input already done")
            else:
                try:
                    plain_imgs, overlay_imgs = find_overlay_images(case_id)
                    if not plain_imgs:
                        print(f"  [SKIP] gemini_full_input: no overlay images found for {case_id} "
                              f"-- run overlay_renderer.py {case_id} first")
                    else:
                        p2 = gemini_full_input(record, plain_imgs, overlay_imgs)
                        text2, model_used = call_gemini(client, p2["system"], p2["user_text"], p2["image_paths"])
                        with open(out2_path, "w") as f:
                            json.dump({"model": model_used, "condition": "full_input", "output": text2,
                                       "n_images_sent": len(p2["image_paths"])}, f, indent=2)
                        print(f"  [OK] gemini_full_input ({len(p2['image_paths'])} images)")
                except Exception as e:
                    print(f"  [ERROR] gemini_full_input: {e}")

        time.sleep(5)  # pace requests to stay under free-tier RPM limits

    print("\nDone. Check outputs/<case_id>/gemini_*.json for results.")


if __name__ == "__main__":
    import sys
    mode = "both"
    if "--measurements-only" in sys.argv:
        mode = "measurements"
    elif "--full-only" in sys.argv:
        mode = "full"
    run_all(only=mode)
