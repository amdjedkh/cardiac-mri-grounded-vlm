"""
prompts_region_grounded_fewshot2.py

Follow-up experiment after the single-example few-shot test. That test fixed
the FORMAT problem (0/8 -> 3/3 fully tagged) but revealed a new CONTENT
problem: MedGemma copied the single example's wording almost verbatim for
both real abnormal cases (P019, P004), including getting P019's MVO status
wrong (said "not present" when it's real and confirmed present).

This variant tests whether the problem was specifically "only one example to
copy" by giving TWO contrasting worked examples (small infarct/no MVO, and
large infarct/MVO present) plus an explicit instruction not to reuse the
examples' wording or conclusions -- the actual extent and MVO status must
come from the new image, not be templated from either example.

Usage: same call signature as the other few-shot variant.
"""

from prompts_region_grounded import (
    DEFAULT_REGION_LEGEND, NEGATIVE_CASE_INSTRUCTION, HALLUCINATION_GUARD, _legend_text,
)

TWO_WORKED_EXAMPLES = """
Here are TWO examples for TWO DIFFERENT patients, so you can see how the
description changes based on what is actually visible. Do NOT reuse the
wording, extent, or MVO status from either example -- your answer must be
based only on the NEW image given to you below, which may look like neither,
either, or something in between these two examples.

EXAMPLE A INPUT: image + overlay showing a SMALL red patch covering a small
part of the ring, and NO yellow region at all.
EXAMPLE A CORRECT OUTPUT:
1) Technique/quality statement: LGE-CMR short-axis imaging, adequate quality.
2) Findings: The LV cavity is normal in size. [region: lv_cavity] The myocardium
shows normal signal outside the marked region. [region: myocardium] A small,
localized area of hyperenhancement is present in one part of the wall,
consistent with a limited infarct. [region: infarct] No microvascular
obstruction is identified. [region: mvo]
3) Impression: Small, localized myocardial infarct. No microvascular obstruction.

EXAMPLE B INPUT: image + overlay showing a LARGE red band covering most of
the ring, with a distinct yellow patch inside part of the red area.
EXAMPLE B CORRECT OUTPUT:
1) Technique/quality statement: LGE-CMR short-axis imaging, adequate quality.
2) Findings: The LV cavity is normal in size. [region: lv_cavity] The myocardium
shows normal signal outside the marked region. [region: myocardium] A large,
near-circumferential area of hyperenhancement is present, consistent with an
extensive infarct. [region: infarct] A distinct area of hypoenhancement is
present within the infarcted region, consistent with microvascular
obstruction. [region: mvo]
3) Impression: Large, near-circumferential myocardial infarct with associated
microvascular obstruction.

Notice: the two examples describe DIFFERENT extents (small/localized vs
large/near-circumferential) and DIFFERENT MVO status (absent vs present),
because the two example images actually looked different. Your job is to
look at the actual size and shape of the red and yellow regions in the NEW
image you are given, and describe THAT accurately -- do not default to
either example's wording if the new image doesn't match it.
"""

REPORT_STRUCTURE_INSTRUCTION_SHORT = (
    "Produce a report with exactly these 3 sections, in order: "
    "1) Technique/quality statement -- no tag needed. "
    "2) Findings -- one short sentence per structure (LV cavity, myocardium, "
    "infarct, MVO), each ending with exactly one [region: ...] tag. "
    "3) Impression -- a short NEW summary sentence, no tags, do not repeat "
    "the Findings sentences."
)


def medgemma_region_grounded_report_fewshot2(record: dict, image_paths: list, overlay_paths: list,
                                               region_legend: dict = None) -> dict:
    legend = region_legend or DEFAULT_REGION_LEGEND

    system = (
        "You are a medical vision-language model analyzing cardiac LGE-MRI images.\n\n"
        f"{REPORT_STRUCTURE_INSTRUCTION_SHORT}\n\n"
        f"{TWO_WORKED_EXAMPLES}\n\n"
        f"{NEGATIVE_CASE_INSTRUCTION} {HALLUCINATION_GUARD}"
    )
    user_text = (
        f"Patient ID: {record.get('patient_id')}\n\n"
        "Now analyze THIS image and its segmentation overlay. Look carefully at the "
        "actual size, extent, and shape of the red (infarct) and yellow (MVO) regions "
        "-- do not copy either example above, describe only what is actually visible "
        "in this specific image.\n\n" + _legend_text(legend)
    )
    return {
        "system": system,
        "user_text": user_text,
        "image_paths": list(image_paths) + list(overlay_paths),
    }


if __name__ == "__main__":
    fake_record = {"patient_id": "Case_TEST"}
    p = medgemma_region_grounded_report_fewshot2(fake_record, ["a.png"], ["a_overlay.png"])
    assert "EXAMPLE A" in p["system"] and "EXAMPLE B" in p["system"]
    assert "do not reuse" in p["system"].lower() or "do not reuse" in p["system"]
    assert len(p["image_paths"]) == 2
    print("Two-example MedGemma prompt built OK")
    print("system chars:", len(p["system"]))
