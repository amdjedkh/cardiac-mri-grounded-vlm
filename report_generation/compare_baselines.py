"""
compare_baselines.py

Loads all four baseline outputs per patient, cross-checks stated numbers
against the real measurement JSON, checks for mandatory negative-finding
statements, checks grounding tag presence, and prints a structured report
highlighting the clearest examples for each failure category -- this is
the actual deliverable Carlos asked for (illustrative failures), not a
full formal evaluation.

Usage:
    python compare_baselines.py
"""

import os
import json
import glob
import re

CONDITIONS = [
    ("gemini_measurements_only", "Gemini (measurements only)"),
    ("gemini_full_input", "Gemini (full input)"),
    ("medgemma_image_only", "MedGemma (image only, zero-shot)"),
    ("medgemma_full_input", "MedGemma (full input)"),
]


def load_patient(case_id: str) -> dict:
    with open(os.path.join("patients", f"{case_id}.json")) as f:
        return json.load(f)


def load_output(case_id: str, condition: str):
    path = os.path.join("outputs", case_id, f"{condition}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_numbers_mentioned(text: str) -> list:
    """Pulls out numeric values (with 1-2 decimal places) mentioned in the report text,
    to compare against ground truth. Rough heuristic, not exact parsing."""
    return [float(x) for x in re.findall(r"\b\d+\.\d{1,3}\b", text)]


def check_number_match(stated: float, truth: float, tol_pct: float = 5.0) -> bool:
    if truth == 0:
        return abs(stated) < 0.5  # near-zero tolerance for true-zero measurements
    return abs(stated - truth) / truth * 100 <= tol_pct


def check_negative_findings(text: str, infarct_present: bool, mvo_present: bool) -> list:
    """Returns a list of issues if mandatory negative statements are missing."""
    issues = []
    text_lower = text.lower()
    if not infarct_present:
        has_negative = any(p in text_lower for p in [
            "no myocardial infarct", "no infarct", "no late gadolinium enhancement identified",
            "not identified", "no evidence of myocardial infarction"
        ])
        if not has_negative:
            issues.append("MISSING negative-infarct statement (infarct_present=False but "
                           "no explicit 'no infarct' statement found)")
    if not mvo_present:
        has_negative = any(p in text_lower for p in [
            "no microvascular obstruction", "no mvo", "no evidence of microvascular"
        ])
        if not has_negative:
            issues.append("MISSING negative-MVO statement (mvo_present=False but no "
                           "explicit 'no MVO' statement found)")
    return issues


def check_grounding_tags(text: str) -> dict:
    """Counts grounding tags present in the output and checks basic structure."""
    tags = re.findall(r'\{"sentence":.*?\}\}', text)
    return {"n_tags": len(tags), "has_any": len(tags) > 0}


def check_measurement_accuracy(text: str, record: dict) -> dict:
    m = record["measurements"]
    truth_values = {
        "infarct_volume_ml": m["infarct_volume_ml"],
        "mvo_volume_ml": m["mvo_volume_ml"],
        "lv_cavity_volume_ml": m["lv_cavity_volume_ml"],
        "myocardium_total_volume_ml": m["myocardium_total_volume_ml"],
        "infarct_pct": m["infarct_percentage_of_myocardium"],
    }
    stated_numbers = extract_numbers_mentioned(text)
    matches = {}
    for name, truth in truth_values.items():
        matched = any(check_number_match(s, truth) for s in stated_numbers) if truth != 0 else True
        matches[name] = {"truth": truth, "matched_in_text": matched}
    return matches


def analyze_case(case_id: str):
    record = load_patient(case_id)
    m = record["measurements"]
    print(f"\n{'='*70}")
    print(f"{case_id}  (infarct={m['infarct_present']} [{m['infarct_percentage_of_myocardium']}%], "
          f"mvo={m['mvo_present']})")
    print(f"{'='*70}")

    for condition_key, condition_label in CONDITIONS:
        output = load_output(case_id, condition_key)
        if output is None:
            print(f"\n--- {condition_label}: NOT RUN ---")
            continue

        text = output.get("output", "")
        print(f"\n--- {condition_label} ---")

        neg_issues = check_negative_findings(text, m["infarct_present"], m["mvo_present"])
        for issue in neg_issues:
            print(f"  [FLAG] {issue}")

        grounding = check_grounding_tags(text)
        if not grounding["has_any"]:
            print(f"  [FLAG] No grounding tags found in output")
        else:
            print(f"  [OK] {grounding['n_tags']} grounding tag(s) present")

        measurement_flags = []
        if m["infarct_present"] or m["mvo_present"]:
            matches = check_measurement_accuracy(text, record)
            for name, info in matches.items():
                if info["truth"] != 0 and not info["matched_in_text"]:
                    measurement_flags.append(name)
                    print(f"  [FLAG] Stated value for {name} does not match truth "
                          f"({info['truth']}) within tolerance -- possible wrong measurement "
                          f"(or simply not mentioned in the text -- check manually)")

        if not neg_issues and grounding["has_any"] and not measurement_flags:
            print(f"  [CLEAN] No issues detected by automated checks")


def summarize_all():
    patient_files = sorted(glob.glob(os.path.join("patients", "*.json")))
    case_ids = [os.path.splitext(os.path.basename(f))[0] for f in patient_files]

    print("COMPARISON REPORT -- automated checks only, still needs a human read for")
    print("hallucinations/phrasing that automated checks can't catch (e.g. plausible-")
    print("sounding but unsupported clinical claims, wrong anatomical location, etc.)")

    for case_id in case_ids:
        analyze_case(case_id)

    print(f"\n{'='*70}")
    print("NEXT STEP: pick 3-5 of the [FLAG] lines above that are the clearest, most")
    print("illustrative failures -- these are your deliverable for the meeting, not")
    print("an exhaustive list. Prioritize: one hallucinated/wrong measurement, one")
    print("missing negative-finding case, one multi-image degradation example (compare")
    print("MedGemma on Case_P019 vs a low-image-count case), and one clean success.")


if __name__ == "__main__":
    summarize_all()
