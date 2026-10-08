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
from google.genai import types
from prompts_region_grounded import gemini_region_grounded_report, check_region_grounding

MODEL_CANDIDATES = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3-flash", "gemini-2.5-flash"]
_working_model = {"name": None}


def resolve_working_model(client) -> str:
    if _working_model["name"]:
        return _working_model["name"]
    for candidate in MODEL_CANDIDATES:
        try:
            client.models.generate_content(model=candidate, contents=[types.Content(role="user", parts=[types.Part.from_text(text="hi")])])
            _working_model["name"] = candidate
            return candidate
        except Exception as e:
            if "NOT_FOUND" in str(e) or "404" in str(e):
                continue
            _working_model["name"] = candidate
            return candidate
    raise RuntimeError("No candidate Gemini model available.")


def call_gemini(client, system, user_text, image_paths, max_retries=6):
    parts = [types.Part.from_text(text=user_text)]
    for p in image_paths:
        with open(p, "rb") as f:
            parts.append(types.Part.from_bytes(data=f.read(), mime_type="image/png"))
    ordered = ([_working_model["name"]] if _working_model["name"] else []) + [c for c in MODEL_CANDIDATES if c != _working_model["name"]]
    tried = []
    for candidate in ordered:
        tried.append(candidate)
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(model=candidate, contents=[types.Content(role="user", parts=parts)], config=types.GenerateContentConfig(system_instruction=system))
                _working_model["name"] = candidate
                return response.text, candidate
            except Exception as e:
                if any(x in str(e) for x in ("RESOURCE_EXHAUSTED", "429", "UNAVAILABLE", "503")):
                    wait = min(30 * (attempt + 1), 120)
                    print(f"    [{candidate}] rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise
        print(f"    [{candidate}] exhausted retries, trying next model...")
    raise RuntimeError(f"Failed on all models: {tried}")


def find_one_overlay_pair(case_id: str):
    overlay_dir = os.path.join("overlays", case_id)
    plains = sorted(glob.glob(os.path.join(overlay_dir, "*_plain.png")))
    overlays = sorted(glob.glob(os.path.join(overlay_dir, "*_overlay.png")))
    if not plains or not overlays:
        return None, None
    mid = len(plains) // 2
    return [plains[mid]], [overlays[mid]]


def run_all(case_ids=None):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")
    client = genai.Client(api_key=api_key)
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
            result = {"patient_id": case_id, "case_type": "synthetic", "model": model, "slice_used": plain[0], "output": text, "grounding_check": check}
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  [OK] {check['n_region_tags_found']} tags, fully_tagged={check['likely_fully_tagged']}")
        except Exception as e:
            print(f"  [ERROR] {e}")
        time.sleep(15)
    print(f"\nDone. See {out_dir}/<case_id>.json")


if __name__ == "__main__":
    run_all(sys.argv[1:] or None)
