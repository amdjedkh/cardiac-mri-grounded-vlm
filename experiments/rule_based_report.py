"""
rule_based_report.py

A report written from the expert mask with no AI: a fixed template filled with
claims derived from the mask of the slice shown to Gemini (same slice index as
the Gemini record's slice_used), for the 80 train + 20 test cases (no training,
so both are fine).

Claims, per slice: infarct present; the AHA segment of the infarct (the same
arc-midpoint segment the existing location checker uses) plus any other
segments holding >= 10% of the infarct pixels; how much of the wall thickness
the lesion occupies (median over occupied 10-degree bins of lesion pixels /
wall pixels: >= 0.75 transmural, 0.5-0.75 more than half, < 0.5 partial);
MVO present and in which segment.

CIRCULARITY (read this before quoting any number): the location claim is
produced with assign_segment(), the same function the existing location check
compares against, so the rule-based report matches that check BY CONSTRUCTION
(66 of 66). That score is a ceiling for the check, not evidence of accuracy.
Version 2 (build_report(version=2)) names every sector holding >= 15% of the
infarct pixels on the slice shown; v1 (default) is unchanged. v2 is derived
from the pixel shares, so on the independent check it is also close to a ceiling
(same geometry assumptions), not independent evidence of accuracy.
The independent check (location_check_independent.py, pixel-wise sector share,
written separately and not importing any of the checker's code) is the one to
read; both are printed.

Scored with the existing checks:
  format    check_region_grounding(): every Findings sentence tagged, valid names
  location  check_location_claims_batch: wall word(s) in the infarct sentence vs
            the AHA segment on the slice shown
  negative  compare_baselines.check_negative_findings: explicit "no infarct" /
            "no MVO" statements when absent (volume-level truth)
plus one ADDED check (not in the current code): presence stated in the infarct /
MVO sentences vs presence in the mask on the slice shown.

Gemini reports scored the same way: train cases from the 80-case pinned-model
run (run_train_v1), test cases from outputs_region_grounded. Reads EMIDEC_ROOT
from config.py; writes to EXPERIMENT_OUT_DIR/rule_based_reports/<case>.json and
rule_based_scores.csv.

    python experiments/rule_based_report.py
"""

import os
import re
import csv
import json
import sys
from collections import Counter, defaultdict

import numpy as np

import common
import location_check_independent as indep
from common import config, md, load_split
sys.path.insert(0, os.path.join(common.REPO_ROOT, "report_generation"))
from assign_aha17_segment import (assign_segment, determine_ring_level, nearest_sector,
                                  SECTOR_CENTERS_4, SECTOR_CENTERS_6, to_bullseye_angle)
from check_location_claims_batch import extract_claimed_walls, claim_matches_real_segment
from prompts_region_grounded import check_region_grounding
from compare_baselines import check_negative_findings

INF, MVO = 3, 4
SECTOR_MIN = 0.10          # v1: sectors listed in the 'also extends' sentence
SECTOR_MIN_V2 = 0.15       # v2 (declared 2026-11-03): the report names EVERY sector holding >= 15% of the infarct pixels


def slice_facts(case_id, slice_idx, sector_min=SECTOR_MIN):
    mask3 = md.load_case(config.EMIDEC_ROOT, case_id)[1]
    m2 = mask3[:, :, slice_idx]
    facts = {"slice": slice_idx, "n_slices": mask3.shape[2], "infarct": bool((m2 == INF).any()), "mvo": bool((m2 == MVO).any())}
    g = md.SliceGeom(m2)
    if not g.valid:
        return facts
    level = determine_ring_level(slice_idx, mask3.shape[2])
    centers = SECTOR_CENTERS_4 if level == "apical" else SECTOR_CENTERS_6
    bull = (np.degrees(g.theta) % 360 - 270) % 360
    sector = np.array([nearest_sector(b, centers) for b in bull])
    facts["level"] = level
    inf_px = g.label == INF
    if inf_px.any():
        counts = Counter(sector[inf_px])
        total = inf_px.sum()
        facts["infarct_sectors"] = [s for s, c in counts.most_common() if c / total >= sector_min]
        les = np.isin(g.label, (INF, MVO))
        bins = (np.degrees(g.theta) % 360 // 10).astype(int)
        fr = [les[bins == b].mean() for b in range(36) if les[bins == b].any()]
        facts["wall_fraction"] = float(np.median(fr))
    if facts["mvo"] and facts["infarct"]:
        mv, inf = m2 == MVO, m2 == INF
        from scipy import ndimage
        ring = ndimage.binary_dilation(mv) & ~mv & np.isin(m2, (2, INF, MVO))
        facts["mvo_within_infarct"] = bool(ring.any() and (ring & inf).sum() / ring.sum() >= 0.5)
    return facts


def wall_phrase(f):
    w = f["wall_fraction"]
    if w >= 0.75:
        return f"The infarct is transmural, occupying about {w:.0%} of the wall thickness"
    if w >= 0.5:
        return f"The infarct involves more than half of the wall thickness (about {w:.0%})"
    return f"The infarct is partial thickness, occupying about {w:.0%} of the wall thickness"


def build_report(case_id, slice_idx, version=1):
    f = slice_facts(case_id, slice_idx, SECTOR_MIN if version == 1 else SECTOR_MIN_V2)
    seg = assign_segment(case_id, slice_idx, config.EMIDEC_ROOT)
    lines = ["Technique/quality statement: The cardiac MRI study is of diagnostic quality.", "", "Findings:",
             "The left ventricular cavity is identified [region: lv_cavity]."]
    if f["infarct"]:
        lines.append("The myocardium shows abnormal enhancement on this slice [region: myocardium].")
        s = seg["infarct"]
        lines.append(f"Hyperenhancement consistent with myocardial infarction is present in the {s['aha_segment_name']} "
                     f"segment (AHA segment {s['aha_segment_number']}) [region: infarct].")
        extra = [x for x in f.get("infarct_sectors", []) if x not in s["aha_segment_name"]]
        if version == 2:
            names = f["infarct_sectors"]
            lines[-1] = ("Hyperenhancement consistent with myocardial infarction involves the "
                         + ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]
                         + (" sectors" if len(names) > 1 else " sector") + " [region: infarct].")
        elif extra:
            lines.append(f"On this slice the infarct also extends into the {' and '.join(extra)} sector(s) [region: infarct].")
        if "wall_fraction" in f:
            lines.append(wall_phrase(f) + " [region: infarct].")
    else:
        lines.append("The myocardium shows no abnormal enhancement on this slice [region: myocardium].")
        lines.append("No infarct is identified [region: infarct].")
    if f["mvo"]:
        m = seg["mvo"]
        where = "within the infarct, " if f.get("mvo_within_infarct") else ""
        lines.append(f"A region of microvascular obstruction is present {where}in the {m['aha_segment_name']} segment [region: mvo].")
    else:
        lines.append("No microvascular obstruction is identified [region: mvo].")
    lines += ["", "Impression:"]
    if f["infarct"]:
        lines.append("Myocardial infarction is present on this slice" + (" with microvascular obstruction." if f["mvo"] else " without microvascular obstruction."))
    else:
        lines.append("No myocardial infarction or microvascular obstruction is identified on this slice.")
    return "\n".join(lines), f


def presence_claims(text):
    """infarct / mvo presence as stated in the sentence carrying their [region] tag (None if no sentence)."""
    clean = text.replace("**", "").replace("*", "")
    out = {}
    for tag in ("infarct", "mvo"):
        verdicts = []
        for m in re.finditer(r"([^\n\[\]]*?)\[region:\s*%s\]" % tag, clean):
            s = re.split(r"(?<=[.!?])\s+", m.group(1).strip())[-1].lower()
            verdicts.append(not re.search(r"\bno\b|\bnot\b|\babsent\b|\bwithout\b|\bnone\b", s))
        out[tag] = any(verdicts) if verdicts else None
    return out


def score(text, case_id, slice_idx, truth_volume, facts):
    res = {"format_ok": bool(check_region_grounding(text)["likely_fully_tagged"])}
    walls = extract_claimed_walls(text)
    res["location_claim"] = bool(walls)
    res["location_match"] = None
    if walls and facts["infarct"]:
        seg = assign_segment(case_id, slice_idx, config.EMIDEC_ROOT)["infarct"]
        res["location_match"] = claim_matches_real_segment(walls, seg.get("aha_segment_name", ""))
    ind = indep.check(text, md.load_case(config.EMIDEC_ROOT, case_id)[1][:, :, slice_idx])
    res["indep_checkable"], res["indep_share"], res["indep_right"] = ind["checkable"], ind["share"], ind["right"]
    res["negative_issue"] = len(check_negative_findings(text, truth_volume["infarct_present"], truth_volume["mvo_present"])) > 0
    pc = presence_claims(text)
    res["infarct_presence_ok"] = None if pc["infarct"] is None else pc["infarct"] == facts["infarct"]
    res["mvo_presence_ok"] = None if pc["mvo"] is None else pc["mvo"] == facts["mvo"]
    return res


def gemini_record(case_id, split):
    base = config.EMIDEC_ROOT
    path = (f"{base}/run_train_v1/outputs_region_grounded/{case_id}.json" if split == "train"
            else f"{base}/outputs_region_grounded/{case_id}.json")
    return json.load(open(path))


def main():
    split = load_split()
    cases = [(c, "train") for c in sorted(split["train"])] + [(c, "test") for c in sorted(split["test"])]
    out_dir = os.path.join(config.EXPERIMENT_OUT_DIR, "rule_based_reports")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for cid, sp in cases:
        g = gemini_record(cid, sp)
        slice_idx = int(re.search(r"slice_(\d+)", g["slice_used"]).group(1))
        truth = json.load(open(f"{config.EMIDEC_ROOT}/patients/{cid}.json"))["measurements"]
        text, facts = build_report(cid, slice_idx)
        rec = {"patient_id": cid, "model": "rule_based_template", "slice_used": g["slice_used"], "output": text,
               "grounding_check": check_region_grounding(text), "claims": facts}
        json.dump(rec, open(os.path.join(out_dir, f"{cid}.json"), "w"), indent=2)
        for source, t in (("rule_based", text), ("gemini", g["output"])):
            r = {"case": cid, "split": sp, "source": source, "slice": slice_idx, "infarct_on_slice": facts["infarct"], "mvo_on_slice": facts["mvo"]}
            r.update(score(t, cid, slice_idx, truth, facts))
            rows.append(r)
    with open(os.path.join(config.EXPERIMENT_OUT_DIR, "rule_based_scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def frac(sub, key):
        v = [r[key] for r in sub if r[key] is not None]
        return f"{sum(v)}/{len(v)}" + (f" ({100 * sum(v) / len(v):.0f}%)" if v else "")

    for sp in ("all", "train", "test"):
        print(f"== {sp}")
        for source in ("rule_based", "gemini"):
            sub = [r for r in rows if r["source"] == source and (sp == "all" or r["split"] == sp)]
            print(f"  {source:10s} n={len(sub):3d}  format ok {frac(sub, 'format_ok')} | location claim made {sum(r['location_claim'] for r in sub)}, "
                  f"match {frac(sub, 'location_match')} | negative-statement issue {sum(r['negative_issue'] for r in sub)} | "
                  f"infarct presence ok {frac(sub, 'infarct_presence_ok')} | MVO presence ok {frac(sub, 'mvo_presence_ok')}")
            shares = [r["indep_share"] for r in sub if r["indep_share"] is not None]
            print(f"  {'':10s} independent location check (pixel-wise sector share, right if >= {indep.SHARE_RIGHT}): "
                  f"checkable {len(shares)}, right {frac(sub, 'indep_right')}, mean share {np.mean(shares) if shares else float('nan'):.2f}")


if __name__ == "__main__":
    main()
