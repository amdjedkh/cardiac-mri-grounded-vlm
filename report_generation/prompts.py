"""
prompts.py

Four baseline prompt scaffolds for the EMIDEC report-generation POC, per
Carlos's task list:

  1. gemini_measurements_only(record)         -- text-only, no images
  2. gemini_full_input(record)                -- images + overlay + measurements + metadata
  3. medgemma_image_only(record)              -- images only, zero-shot
  4. medgemma_full_input(record)              -- images + masks + measurements + metadata

Each returns a dict: {"system": ..., "user_text": ..., "image_paths": [...]}
so the same structure plugs into whichever provider SDK is used
(Gemini API, MedGemma local/HF inference, etc.) -- swap the calling code,
not the prompt content, when comparing providers.

All four share the same hallucination guard and negative-case instruction
from report_template.md, so any output difference reflects the input
condition, not a prompting inconsistency between baselines.
"""

from __future__ import annotations
import json


HALLUCINATION_GUARD = (
    "If a value is not present in the provided data, respond with 'not assessable' "
    "for that item. Do not estimate, infer, or invent any measurement, location, or "
    "diagnosis that is not explicitly supported by the provided masks, measurements, "
    "or clinical metadata."
)

NEGATIVE_CASE_INSTRUCTION = (
    "If infarct_present is false, you must explicitly state that no infarct / late "
    "gadolinium enhancement is identified -- do not simply omit it. If mvo_present is "
    "false, you must explicitly state that no microvascular obstruction is identified. "
    "A correct report about a normal or partially normal case states the negative "
    "findings directly."
)

GROUNDING_INSTRUCTION = (
    "For every sentence in the Findings section, also output a grounding tag in the "
    "form {\"sentence\": \"...\", \"source\": {\"field\": \"<json_field_name>\", "
    "\"slices\": [...]}} identifying exactly which measurement or metadata field "
    "supports that sentence. Every Findings sentence must have a tag; if a sentence "
    "cannot be grounded to a specific field, do not include it."
)

REPORT_STRUCTURE_INSTRUCTION = (
    "Produce a report with these sections in order: "
    "1) Technique/quality statement (one line). "
    "2) Findings (LV cavity, myocardium/infarct, MVO -- one short statement each). "
    "3) Quantitative summary (restate the key numbers). "
    "4) Clinical correlation (one line, only using provided clinical metadata). "
    "5) Impression (1-3 sentences, introduces no new findings). "
    "6) Limitations statement (single time point, mask-derived measurements, "
    "not yet expert-verified)."
)


def _measurements_and_metadata_text(record: dict) -> str:
    """Serializes the measurements + clinical metadata block shared across scaffolds."""
    m = record.get("measurements", {})
    c = record.get("clinical", {})
    payload = {
        "patient_id": record.get("patient_id"),
        "measurements": m,
        "clinical_metadata": c,
    }
    return json.dumps(payload, indent=2)


def gemini_measurements_only(record: dict) -> dict:
    """Baseline 1: text-only. Model never sees the image, only structured data."""
    system = (
        "You are assisting with cardiac MRI report drafting for a research pipeline. "
        f"{REPORT_STRUCTURE_INSTRUCTION} {NEGATIVE_CASE_INSTRUCTION} {GROUNDING_INSTRUCTION} "
        f"{HALLUCINATION_GUARD} You are NOT shown the image in this condition -- base the "
        "report only on the structured measurements and clinical metadata below."
    )
    user_text = (
        "Draft an LGE-CMR report using only the following structured data "
        "(no image is provided in this condition):\n\n"
        f"{_measurements_and_metadata_text(record)}"
    )
    return {"system": system, "user_text": user_text, "image_paths": []}


def gemini_full_input(record: dict, image_paths: list, overlay_paths: list) -> dict:
    """Baseline 2: images + segmentation overlay + measurements + metadata."""
    system = (
        "You are assisting with cardiac MRI report drafting for a research pipeline. "
        f"{REPORT_STRUCTURE_INSTRUCTION} {NEGATIVE_CASE_INSTRUCTION} {GROUNDING_INSTRUCTION} "
        f"{HALLUCINATION_GUARD} You are shown the LGE-CMR slice images and segmentation "
        "overlays (infarct and MVO regions highlighted) alongside structured measurements "
        "and clinical metadata. Use the images to support your description, but the "
        "quantitative values you report must match the structured data exactly -- do not "
        "re-estimate volumes visually."
    )
    user_text = (
        "Draft an LGE-CMR report using the attached slice images, segmentation overlays, "
        "and the following structured data:\n\n"
        f"{_measurements_and_metadata_text(record)}"
    )
    return {
        "system": system,
        "user_text": user_text,
        "image_paths": list(image_paths) + list(overlay_paths),
    }


def medgemma_image_only(record: dict, image_paths: list) -> dict:
    """Baseline 3: MedGemma zero-shot, images only, no measurements/metadata at all.
    This is the pure 'what can a medical VLM infer visually with no structured input' test."""
    system = (
        "You are a medical vision-language model analyzing cardiac LGE-MRI images. "
        f"{REPORT_STRUCTURE_INSTRUCTION} {NEGATIVE_CASE_INSTRUCTION} {HALLUCINATION_GUARD} "
        "No segmentation masks, measurements, or clinical metadata are provided in this "
        "condition -- assess only from the images. Note explicitly in the Limitations "
        "section that no quantitative measurements were available."
    )
    user_text = (
        f"Patient ID: {record.get('patient_id')}\n"
        "Analyze the attached LGE-CMR slice images and draft a report. "
        "No structured measurements or clinical metadata are provided."
    )
    return {"system": system, "user_text": user_text, "image_paths": list(image_paths)}


def medgemma_full_input(record: dict, image_paths: list, overlay_paths: list) -> dict:
    """Baseline 4: MedGemma with images, masks/overlays, measurements, and metadata."""
    system = (
        "You are a medical vision-language model analyzing cardiac LGE-MRI images. "
        f"{REPORT_STRUCTURE_INSTRUCTION} {NEGATIVE_CASE_INSTRUCTION} {GROUNDING_INSTRUCTION} "
        f"{HALLUCINATION_GUARD} You are given the LGE-CMR slice images, segmentation masks "
        "(infarct and MVO regions), quantitative measurements derived from those masks, "
        "and structured clinical metadata. Quantitative values you state must match the "
        "provided measurements exactly."
    )
    user_text = (
        "Draft an LGE-CMR report using the attached images, segmentation overlays, and "
        "the following structured data:\n\n"
        f"{_measurements_and_metadata_text(record)}"
    )
    return {
        "system": system,
        "user_text": user_text,
        "image_paths": list(image_paths) + list(overlay_paths),
    }


# ----------------------------------------------------------------------------
# Self-test: build all four prompts from the synthetic patient record
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    with open("/tmp/P_TEST.json") as f:
        record = json.load(f)

    fake_images = ["slice_03.png", "slice_04.png", "slice_05.png"]
    fake_overlays = ["slice_03_overlay.png", "slice_04_overlay.png", "slice_05_overlay.png"]

    p1 = gemini_measurements_only(record)
    p2 = gemini_full_input(record, fake_images, fake_overlays)
    p3 = medgemma_image_only(record, fake_images)
    p4 = medgemma_full_input(record, fake_images, fake_overlays)

    for name, p in [
        ("gemini_measurements_only", p1),
        ("gemini_full_input", p2),
        ("medgemma_image_only", p3),
        ("medgemma_full_input", p4),
    ]:
        assert "system" in p and "user_text" in p and "image_paths" in p
        print(f"--- {name} ---")
        print("system chars:", len(p["system"]), "| user_text chars:", len(p["user_text"]),
              "| n_images:", len(p["image_paths"]))

    print("\nAll four prompt scaffolds built successfully -- self-test passed.")
