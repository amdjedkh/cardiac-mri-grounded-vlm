"""
run_region_grounded_poc.py

Generates the corrected proof of concept Carlos asked for: MRI + segmentation
mask overlay -> Gemini -> report where every Findings sentence is explicitly
tagged to a real visual region, not a measurement field.

Run AFTER: patients/*.json exist (build_patient_jsons.py) and overlays/*
exist (overlay_renderer.py) for at least a few cases -- doesn't need all 8,
Carlos asked for "several examples," not full coverage.

Usage:
    python run_region_grounded_poc.py
"""

import os
import json
import glob
import time
from google import genai
from google.genai import types

from prompts_region_grounded import gemini_region_grounded_report, check_region_grounding

MODEL_CANDIDATES = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3-flash", "gemini-2.5-flash"]
_working_model = {"name": None}


def resolve_working_model(client) -> str:
    if _working_model["name"]:
        return _working_model["name"]
    for candidate in MODEL_CANDIDATES:
        try:
            client.models.generate_content(
                model=candidate,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text="hi")])],
            )
            print(f"  (using model: {candidate})")
            _working_model["name"] = candidate
            return candidate
        except Exception as e:
            if "NOT_FOUND" in str(e) or "404" in str(e):
                continue
            _working_model["name"] = candidate
            return candidate
    raise RuntimeError("No candidate Gemini model available.")


def call_gemini(client, system, user_text, image_paths, max_retries=3):
    model = resolve_working_model(client)
    parts = [types.Part.from_text(text=user_text)]
    for p in image_paths:
        with open(p, "rb") as f:
            img_bytes = f.read()
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(system_instruction=system),
            )
            return response.text, model
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e) or "UNAVAILABLE" in str(e) or "503" in str(e):
                wait = 20 * (attempt + 1)
                print(f"    retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Failed after retries.")


def find_one_overlay_pair(case_id: str):
    """Picks ONE representative slice (plain + overlay pair) per case -- Carlos asked
    for example outputs, not a full per-slice sweep yet."""
    overlay_dir = os.path.join("overlays", case_id)
    plains = sorted(glob.glob(os.path.join(overlay_dir, "*_plain.png")))
    overlays = sorted(glob.glob(os.path.join(overlay_dir, "*_overlay.png")))
    if not plains or not overlays:
        return None, None
    mid = len(plains) // 2
    return [plains[mid]], [overlays[mid]]


def run_all(case_ids: list = None):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY not set. Run: $env:GEMINI_API_KEY = "your-key-here"')
    client = genai.Client(api_key=api_key)

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

        time.sleep(5)

    print(f"\nDone. Check {out_dir}/<case_id>.json for the example reports and grounding checks.")


if __name__ == "__main__":
    run_all()
