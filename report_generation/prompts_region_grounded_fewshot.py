"""
prompts_region_grounded_fewshot.py

Tests Karen's hypothesis directly: is MedGemma's poor grounding-format
compliance a real capability gap, or a prompt-format problem? The original
prompt (prompts_region_grounded.py) gives MedGemma a dense, compound
instruction (structure rules + grounding rules + negative-case rules +
hallucination guard, all stacked together) and only abstract rules, no
worked example.

This variant keeps the same requirements but adds ONE concrete worked
example directly in the prompt, so the model can pattern-match a correct
output instead of parsing a rules list. Smaller instruction-tuned models
typically respond much better to "here's what a correct answer looks like"
than to a long list of constraints -- this is the first, cheapest thing to
try before concluding it's a real capability gap.

Usage: same call signature as medgemma_region_grounded_report(), swap the
import.
"""

from prompts_region_grounded import (
    DEFAULT_REGION_LEGEND, NEGATIVE_CASE_INSTRUCTION, HALLUCINATION_GUARD, _legend_text,
)

WORKED_EXAMPLE = """
Here is an example of a correctly formatted response for a DIFFERENT patient,
so you can see the exact format expected. Follow this format exactly, but
base your actual answer only on the NEW image you are given below, not on
this example's findings.

EXAMPLE INPUT: image + overlay showing a small red region and no yellow region.

EXAMPLE CORRECT OUTPUT:
1) Technique/quality statement: LGE-CMR short-axis imaging, adequate quality.
2) Findings: The LV cavity is normal in size. [region: lv_cavity] The myocardium
shows normal signal outside the marked region. [region: myocardium] A focal area
of hyperenhancement is present, consistent with infarct. [region: infarct] No
microvascular obstruction is identified. [region: mvo]
3) Impression: Focal myocardial infarct. No microvascular obstruction.

Notice in the example: the Technique line has NO [region: ...] tag. Every
Findings sentence has EXACTLY ONE tag, placed right after that sentence, not
attached as if it were the sentence's subject. The Impression is a SHORT NEW
summary -- it does not repeat the Findings sentences or their tags.
"""

REPORT_STRUCTURE_INSTRUCTION_SHORT = (
    "Produce a report with exactly these 3 sections, in order: "
    "1) Technique/quality statement -- no tag needed. "
    "2) Findings -- one short sentence per structure (LV cavity, myocardium, "
    "infarct, MVO), each ending with exactly one [region: ...] tag. "
    "3) Impression -- a short NEW summary sentence, no tags, do not repeat "
    "the Findings sentences."
)


def medgemma_region_grounded_report_fewshot(record: dict, image_paths: list, overlay_paths: list,
                                              region_legend: dict = None) -> dict:
    legend = region_legend or DEFAULT_REGION_LEGEND

    system = (
        "You are a medical vision-language model analyzing cardiac LGE-MRI images.\n\n"
        f"{REPORT_STRUCTURE_INSTRUCTION_SHORT}\n\n"
        f"{WORKED_EXAMPLE}\n\n"
        f"{NEGATIVE_CASE_INSTRUCTION} {HALLUCINATION_GUARD}"
    )
    user_text = (
        f"Patient ID: {record.get('patient_id')}\n\n"
        "Now analyze THIS image and its segmentation overlay, following the exact "
        "format shown in the example above.\n\n" + _legend_text(legend)
    )
    return {
        "system": system,
        "user_text": user_text,
        "image_paths": list(image_paths) + list(overlay_paths),
    }


if __name__ == "__main__":
    fake_record = {"patient_id": "Case_TEST"}
    p = medgemma_region_grounded_report_fewshot(fake_record, ["a.png"], ["a_overlay.png"])
    assert "EXAMPLE CORRECT OUTPUT" in p["system"]
    assert len(p["image_paths"]) == 2
    print("Few-shot MedGemma prompt built OK")
    print("system chars:", len(p["system"]))
