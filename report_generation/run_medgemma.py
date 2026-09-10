"""
run_medgemma.py

Runs the two MedGemma baselines (image-only zero-shot, full-input) across
all patients, by calling the deployed Modal function in modal_medgemma.py.

Setup:
    modal deploy modal_medgemma.py     (one-time, or after any code change)

Usage:
    python run_medgemma.py
"""

import os
import json
import glob
import modal

from prompts import medgemma_image_only, medgemma_full_input


def load_patient(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def find_overlay_images(case_id: str):
    overlay_dir = os.path.join("overlays", case_id)
    plain = sorted(glob.glob(os.path.join(overlay_dir, "*_plain.png")))
    overlay = sorted(glob.glob(os.path.join(overlay_dir, "*_overlay.png")))
    return plain, overlay


def read_bytes(paths: list) -> list:
    out = []
    for p in paths:
        with open(p, "rb") as f:
            out.append(f.read())
    return out


def run_all():
    MedGemma = modal.Cls.from_name("medgemma-emidec-poc", "MedGemma")
    medgemma = MedGemma()

    patient_files = sorted(glob.glob(os.path.join("patients", "*.json")))
    if not patient_files:
        raise RuntimeError("No patient JSON files found in patients/ -- run build_patient_jsons.py first.")

    for pf in patient_files:
        record = load_patient(pf)
        case_id = record["patient_id"]
        out_dir = os.path.join("outputs", case_id)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== {case_id} ===")
        plain_imgs, overlay_imgs = find_overlay_images(case_id)

        if not plain_imgs:
            print(f"  [SKIP] no overlay images found -- run overlay_renderer.py {case_id} first")
            continue

        # Baseline 3: MedGemma zero-shot, images only
        try:
            p3 = medgemma_image_only(record, plain_imgs)
            img_bytes = read_bytes(p3["image_paths"])
            text3 = medgemma.generate.remote(p3["system"], p3["user_text"], img_bytes)
            with open(os.path.join(out_dir, "medgemma_image_only.json"), "w") as f:
                json.dump({"model": "medgemma-4b-it", "condition": "image_only", "output": text3}, f, indent=2)
            print(f"  [OK] medgemma_image_only ({len(img_bytes)} images)")
        except Exception as e:
            print(f"  [ERROR] medgemma_image_only: {e}")

        # Baseline 4: MedGemma full-input
        try:
            p4 = medgemma_full_input(record, plain_imgs, overlay_imgs)
            img_bytes = read_bytes(p4["image_paths"])
            text4 = medgemma.generate.remote(p4["system"], p4["user_text"], img_bytes)
            with open(os.path.join(out_dir, "medgemma_full_input.json"), "w") as f:
                json.dump({"model": "medgemma-4b-it", "condition": "full_input", "output": text4,
                           "n_images_sent": len(img_bytes)}, f, indent=2)
            print(f"  [OK] medgemma_full_input ({len(img_bytes)} images)")
        except Exception as e:
            print(f"  [ERROR] medgemma_full_input: {e}")

    print("\nDone. Check outputs/<case_id>/medgemma_*.json for results.")


if __name__ == "__main__":
    run_all()
