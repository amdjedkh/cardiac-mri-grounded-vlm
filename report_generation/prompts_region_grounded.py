"""
prompts_region_grounded.py

Implements the CORRECTED proof of concept Carlos asked for after the region
grounding discussion.

Important distinction from prompts.py's GROUNDING_INSTRUCTION:
  - prompts.py tags each sentence to a MEASUREMENT FIELD, e.g.
    {"source": {"field": "infarct_volume_ml"}}. Carlos clarified this is
    NOT standard VLM grounding -- it's just bookkeeping over the numbers.
  - Standard grounding means every sentence is linked back to an actual
    VISUAL REGION in the image: a segmentation region id/color, or a
    bounding box. This module implements that version.

Input for this proof of concept, per Carlos: the MRI scan + the segmentation
mask (as a color overlay), nothing else required (measurements/metadata are
optional extras, not the grounding mechanism itself).

Usage:
    from prompts_region_grounded import gemini_region_grounded_report
    p = gemini_region_grounded_report(record, image_paths, overlay_paths, region_legend)
"""

from __future__ import annotations
import json


HALLUCINATION_GUARD = (
    "If you cannot identify a finding with confidence from the image and segmentation "
    "overlay provided, do not invent one. Only describe what is visually supported."
)

NEGATIVE_CASE_INSTRUCTION = (
    "If no infarct region (red) is visible in the overlay, you must explicitly state "
    "that no infarct is identified -- do not omit it. If no MVO region (yellow) is "
    "visible, you must explicitly state that no microvascular obstruction is identified."
)

# This is the corrected mechanism: every sentence must carry an explicit tag pointing
# to a real visual region (a segmentation region id/color/label), not a data field name.
REGION_GROUNDING_INSTRUCTION = (
    "This is a grounded report generation task. Grounding means every statement you make "
    "must be explicitly linked to a visual region in the segmentation overlay you were "
    "shown, not just to a number. "
    "You are given a segmentation overlay where colored regions correspond to specific "
    "anatomical or pathological structures (see the region legend below). "
    "For every sentence in the Findings section, immediately follow it with a grounding "
    "tag in this exact format: [region: <region_name>] where <region_name> is one of the "
    "legend labels below, referring to the actual colored region in the image that "
    "supports that sentence. "
    "If a sentence describes the overall study or a normal structure with no distinct "
    "colored region (e.g. general LV cavity shape), use [region: lv_cavity] or "
    "[region: myocardium] as appropriate rather than omitting the tag. "
    "Every single sentence in the Findings section must end with exactly one "
    "[region: ...] tag. Do not produce any Findings sentence without one."
)

REPORT_STRUCTURE_INSTRUCTION = (
    "Produce a report with these sections in order: "
    "1) Technique/quality statement (one line, no grounding tag needed here). "
    "2) Findings (one short statement per structure -- LV cavity, myocardium, infarct, "
    "MVO -- every sentence here MUST end with a [region: ...] tag). "
    "3) Impression (1-3 sentences, summarizing only what was stated in Findings, "
    "no new claims, no grounding tag required here)."
)

DEFAULT_REGION_LEGEND = {
    "lv_cavity": "blue region if present in the overlay, or the central blood-filled chamber",
    "myocardium": "unhighlighted heart-wall tissue surrounding the cavity (not red or yellow)",
    "infarct": "red-highlighted region in the overlay",
    "mvo": "yellow-highlighted region in the overlay",
}


def _legend_text(legend: dict) -> str:
    lines = [f'- "{name}": {desc}' for name, desc in legend.items()]
    return "Region legend (use these exact names in your [region: ...] tags):\n" + "\n".join(lines)


def gemini_region_grounded_report(
    record: dict,
    image_paths: list,
    overlay_paths: list,
    region_legend: dict = None,
    include_measurements: bool = False,
) -> dict:
    """
    The corrected proof of concept: MRI + segmentation mask overlay -> Gemini ->
    a report where every Findings sentence is tagged to a real visual region.

    include_measurements=False by default, matching Carlos's ask for a first pass
    using just the image + mask. Set True later to test whether adding numbers on
    top of the region tags changes anything (a natural follow-up ablation).
    """
    legend = region_legend or DEFAULT_REGION_LEGEND

    system = (
        "You are assisting with cardiac MRI grounded report drafting for a research "
        f"pipeline. {REPORT_STRUCTURE_INSTRUCTION} {REGION_GROUNDING_INSTRUCTION} "
        f"{NEGATIVE_CASE_INSTRUCTION} {HALLUCINATION_GUARD}"
    )

    parts = [
        "You are shown an LGE-CMR slice image and its segmentation overlay "
        "(same slice, mask colors applied on top).",
        _legend_text(legend),
    ]
    if include_measurements and "measurements" in record:
        parts.append(
            "For reference only (not required for grounding, grounding must still "
            "point to the image/overlay): " + json.dumps(record["measurements"], indent=2)
        )

    user_text = "\n\n".join(parts)

    return {
        "system": system,
        "user_text": user_text,
        "image_paths": list(image_paths) + list(overlay_paths),
    }


def medgemma_region_grounded_report(
    record: dict,
    image_paths: list,
    overlay_paths: list,
    region_legend: dict = None,
) -> dict:
    """Same corrected grounding task, for MedGemma (optional second model per Carlos)."""
    legend = region_legend or DEFAULT_REGION_LEGEND

    system = (
        "You are a medical vision-language model analyzing cardiac LGE-MRI images. "
        f"{REPORT_STRUCTURE_INSTRUCTION} {REGION_GROUNDING_INSTRUCTION} "
        f"{NEGATIVE_CASE_INSTRUCTION} {HALLUCINATION_GUARD}"
    )
    user_text = (
        f"Patient ID: {record.get('patient_id')}\n\n"
        "You are shown an LGE-CMR slice image and its segmentation overlay.\n\n"
        + _legend_text(legend)
    )
    return {
        "system": system,
        "user_text": user_text,
        "image_paths": list(image_paths) + list(overlay_paths),
    }


def check_region_grounding(report_text: str, legend: dict = None) -> dict:
    """
    Quick automated check for the comparison step: does every Findings sentence
    actually carry a [region: ...] tag, and are the region names valid (from the
    legend), not invented?
    """
    import re

    legend = legend or DEFAULT_REGION_LEGEND
    valid_names = set(legend.keys())

    # strip markdown bold markers before any analysis -- real model output uses
    # **Findings:** style headers that the original version of this check did not
    # account for, causing a stray "**" before "Impression" to be miscounted as
    # part of the last sentence (confirmed against a real Case_P019 output where
    # all 4 sentences were correctly tagged but this bug reported 5 vs 4).
    clean_text = report_text.replace("**", "").replace("*", "")

    # split into the three sections up front, so tag counting can be scoped
    # correctly -- an earlier version counted tags across the WHOLE text, which
    # let a MedGemma output that duplicated the Findings verbatim into the
    # Impression (with tags) falsely pass as "fully tagged" on padded count.
    before_findings, findings_section, impression_section = clean_text, "", ""
    if "Findings" in clean_text:
        before_findings, rest = clean_text.split("Findings", 1)
        if "Impression" in rest:
            findings_section, impression_section = rest.split("Impression", 1)
        else:
            findings_section = rest
    findings_section = findings_section.lstrip(":").strip()

    # tags counted ONLY within Findings -- this is what "fully tagged" should mean
    tags_found = re.findall(r"\[region:\s*([a-zA-Z_]+)\]", findings_section)
    invalid_tags = [t for t in tags_found if t not in valid_names]

    # tags appearing outside Findings (Technique or Impression) are a separate
    # problem worth flagging -- the instructions say no tag is needed there
    tags_outside_findings = (
        re.findall(r"\[region:\s*([a-zA-Z_]+)\]", before_findings) +
        re.findall(r"\[region:\s*([a-zA-Z_]+)\]", impression_section)
    )

    # crude duplicate-content check: does a substantial chunk of the Findings
    # text reappear verbatim in the Impression? (catches the MedGemma pattern
    # where Impression = Findings copy-pasted rather than a summary)
    findings_no_tags = re.sub(r"\[region:\s*[a-zA-Z_]+\]", "", findings_section).strip()
    impression_no_tags = re.sub(r"\[region:\s*[a-zA-Z_]+\]", "", impression_section).strip()
    duplicated_impression = False
    if len(findings_no_tags) > 20:
        # check if a long substring of findings text shows up in impression
        probe = findings_no_tags[:60].strip()
        if probe and probe in impression_no_tags:
            duplicated_impression = True

    raw_parts = re.split(r"[.!?]\s", findings_section)
    # A fragment only counts as a real sentence if, after removing any region
    # tags, there's still real alphabetic sentence content left. This is
    # deliberately format-agnostic: earlier versions tried to match specific
    # trailing artifacts (markdown "**", then separately numbered-list "3) "
    # prefixes) one at a time and kept missing new formatting styles real
    # models actually use. Checking "is there real sentence text here" once
    # a tag is stripped out avoids that whack-a-mole pattern.
    approx_sentences = 0
    for s in raw_parts:
        s_stripped = s.strip()
        if len(s_stripped) <= 5:
            continue
        remainder = re.sub(r"\[region:\s*[a-zA-Z_]+\]", "", s_stripped)
        remainder_alpha = re.sub(r"[^a-zA-Z]", "", remainder)
        if len(remainder_alpha) < 5:
            continue  # nothing left but tag(s) + formatting junk, not a real sentence
        approx_sentences += 1

    return {
        "n_region_tags_found": len(tags_found),
        "tags_found": tags_found,
        "invalid_region_names": invalid_tags,
        "approx_findings_sentences": approx_sentences,
        "tags_outside_findings": tags_outside_findings,
        "duplicated_impression": duplicated_impression,
        "likely_fully_tagged": (
            len(tags_found) >= approx_sentences
            and len(invalid_tags) == 0
            and len(tags_outside_findings) == 0
            and not duplicated_impression
        ),
    }


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    fake_record = {"patient_id": "Case_TEST", "measurements": {"infarct_volume_ml": 12.3}}
    fake_images = ["slice_04_plain.png"]
    fake_overlays = ["slice_04_overlay.png"]

    p1 = gemini_region_grounded_report(fake_record, fake_images, fake_overlays)
    assert "[region:" in p1["system"] or "region:" in p1["system"]
    assert len(p1["image_paths"]) == 2
    print("gemini_region_grounded_report built OK")
    print("system chars:", len(p1["system"]), "| user_text chars:", len(p1["user_text"]))

    p2 = medgemma_region_grounded_report(fake_record, fake_images, fake_overlays)
    assert len(p2["image_paths"]) == 2
    print("medgemma_region_grounded_report built OK")

    # Test the checker against a well-formed fake report
    good_report = (
        "Technique: LGE-CMR short axis.\n\n"
        "Findings: The LV cavity is normal in size. [region: lv_cavity] "
        "There is transmural infarct in the anteroseptal wall. [region: infarct] "
        "No microvascular obstruction is identified. [region: mvo]\n\n"
        "Impression: Anteroseptal infarct without MVO."
    )
    result = check_region_grounding(good_report)
    print(json.dumps(result, indent=2))
    assert result["n_region_tags_found"] == 3
    assert result["invalid_region_names"] == []
    assert result["likely_fully_tagged"] is True

    # Test against a badly-grounded report (missing tags, invented region name)
    bad_report = (
        "Findings: There is infarct in the heart. "
        "Some abnormal tissue is seen. [region: weird_made_up_area]"
    )
    result_bad = check_region_grounding(bad_report)
    print(json.dumps(result_bad, indent=2))
    assert "weird_made_up_area" in result_bad["invalid_region_names"]
    assert result_bad["likely_fully_tagged"] is False

    print("\nAll self-tests passed.")
