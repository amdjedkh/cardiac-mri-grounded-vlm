"""
score_reports.py

Scores four report sources on the 100 EMIDEC cases with BOTH location checks,
each with a shuffle baseline, and reports all denominators:

  rule-based v1   template from the mask (experiments/rule_based_report.py, version 1)
  rule-based v2   same, but names every sector holding >= 15% of the infarct pixels
  gemini old100   EMIDEC/outputs_region_grounded (September run; 95 pinned-model
                  records, 5 from the fallback model; no settings stored)
  gemini new100   EMIDEC/run_all100_pinned (pinned config, settings stored)

Location checks:
  existing     check_location_claims_batch logic: wall word(s) in the infarct
               sentence vs the AHA segment of the arc midpoint (assign_segment)
  recall-only  location_check_independent: share of infarct pixels in the claimed
               sectors (lenient word map) >= 0.5
  two-sided    location_check_independent: every claimed sector >= 10% of the
               infarct pixels AND claimed sectors together >= 80% (strict word map)
Thresholds are fixed, none is tuned.

Shuffle baseline (both families): each report is paired with ANOTHER case's mask
slice (the slice that case's report used), 1000 seeded permutations without fixed
points; the chance level is the mean over permutations of right / checkable
(5th-95th percentile shown). Everything is also split train (80) / test (20).

Writes location_scores_v2.csv to EXPERIMENT_OUT_DIR.

    python experiments/score_reports.py
"""

import os
import re
import csv
import json

import numpy as np

import common
import location_check_independent as L
import rule_based_report as R
from common import config, md, load_split
from assign_aha17_segment import assign_segment
from check_location_claims_batch import extract_claimed_walls, claim_matches_real_segment

N_PERM, SEED = 1000, 42


def load_sources(cases, slices):
    base = config.EMIDEC_ROOT
    src = {"rule-based v1": {}, "rule-based v2": {}, "gemini old100": {}, "gemini new100": {}}
    out2 = os.path.join(config.EXPERIMENT_OUT_DIR, "rule_based_reports_v2")
    os.makedirs(out2, exist_ok=True)
    for c in cases:
        src["rule-based v1"][c] = R.build_report(c, slices[c], 1)[0]
        t2, f2 = R.build_report(c, slices[c], 2)
        src["rule-based v2"][c] = t2
        json.dump({"patient_id": c, "model": "rule_based_template_v2", "slice_used": f"slice_{slices[c]:02d}",
                   "output": t2, "claims": f2}, open(os.path.join(out2, f"{c}.json"), "w"), indent=2)
        src["gemini old100"][c] = json.load(open(f"{base}/outputs_region_grounded/{c}.json"))["output"]
        p = f"{base}/run_all100_pinned/outputs_region_grounded/{c}.json"
        if os.path.exists(p):
            src["gemini new100"][c] = json.load(open(p))["output"]
    return src


def derange(rng, n):
    p = rng.permutation(n)
    while (p == np.arange(n)).any():
        for i in np.nonzero(p == np.arange(n))[0]:
            j = (i + 1) % n
            p[i], p[j] = p[j], p[i]
    return p


def main():
    split = load_split()
    test = set(split["test"])
    cases = sorted(split["train"] + split["test"])
    slices = {c: int(re.search(r"slice_(\d+)", json.load(open(f"{config.EMIDEC_ROOT}/outputs_region_grounded/{c}.json"))["slice_used"]).group(1))
              for c in cases}
    masks = {c: md.load_case(config.EMIDEC_ROOT, c)[1][:, :, slices[c]] for c in cases}
    shares = {c: L.shares(masks[c]) for c in cases}
    seg = {}
    for c in cases:
        a = assign_segment(c, slices[c], config.EMIDEC_ROOT)["infarct"]
        seg[c] = a.get("aha_segment_name") if a.get("present") else None
    src = load_sources(cases, slices)
    rng = np.random.RandomState(SEED)
    perms = np.array([derange(rng, len(cases)) for _ in range(N_PERM)])
    idx = {c: i for i, c in enumerate(cases)}
    subsets = {"all 100": list(range(100)), "test 20": [idx[c] for c in cases if c in test],
               "train 80": [idx[c] for c in cases if c not in test]}
    rows = []
    for name, texts in src.items():
        if len(texts) < 100:
            print(f"{name}: only {len(texts)} records, skipped")
            continue
        walls = [extract_claimed_walls(texts[c]) for c in cases]
        len_sets = [L.claimed_from(texts[c], L.WORD_SECTORS) for c in cases]
        str_sets = [L.claimed_from(texts[c], L.STRICT_SECTORS) for c in cases]
        n = len(cases)
        mats = {k: (np.zeros((n, n), bool), np.zeros((n, n), bool)) for k in ("existing", "recall-only", "two-sided")}
        for i in range(n):
            for j, cj in enumerate(cases):
                if walls[i] and seg[cj]:
                    mats["existing"][1][i, j] = True
                    mats["existing"][0][i, j] = claim_matches_real_segment(walls[i], seg[cj])
                if shares[cj] is not None:
                    if len_sets[i]:
                        mats["recall-only"][1][i, j] = True
                        mats["recall-only"][0][i, j] = L.verdict_recall(len_sets[i], shares[cj])
                    if str_sets[i]:
                        mats["two-sided"][1][i, j] = True
                        mats["two-sided"][0][i, j] = L.verdict_two_sided(str_sets[i], shares[cj])
        for sub, ii in subsets.items():
            ii = np.array(ii)
            for check, (M, C) in mats.items():
                real_c, real_r = int(C[ii, ii].sum()), int(M[ii, ii].sum())
                rates, chk = [], []
                for p in perms:
                    c_ = C[ii, p[ii]].sum()
                    chk.append(c_)
                    if c_:
                        rates.append(M[ii, p[ii]].sum() / c_)
                rows.append({"source": name, "subset": sub, "check": check, "reports": len(ii),
                             "checkable": real_c, "right": real_r,
                             "right_rate": round(real_r / real_c, 3) if real_c else "",
                             "shuffle_mean_rate": round(float(np.mean(rates)), 3), "shuffle_p5": round(float(np.percentile(rates, 5)), 3),
                             "shuffle_p95": round(float(np.percentile(rates, 95)), 3),
                             "shuffle_mean_checkable": round(float(np.mean(chk)), 1)})
    # other existing checks, real pairing only
    for name, texts in src.items():
        if len(texts) < 100:
            continue
        for sub, ii in subsets.items():
            fmt = neg = pres_i = pres_m = 0
            n_i = n_m = 0
            for i in ii:
                c = cases[i]
                truth = json.load(open(f"{config.EMIDEC_ROOT}/patients/{c}.json"))["measurements"]
                f = R.slice_facts(c, slices[c])
                s = R.score(texts[c], c, slices[c], truth, f)
                fmt += s["format_ok"]; neg += s["negative_issue"]
                if s["infarct_presence_ok"] is not None:
                    n_i += 1; pres_i += s["infarct_presence_ok"]
                if s["mvo_presence_ok"] is not None:
                    n_m += 1; pres_m += s["mvo_presence_ok"]
            rows.append({"source": name, "subset": sub, "check": "other: format ok / negative issues / infarct presence ok / MVO presence ok",
                         "reports": len(ii), "checkable": "", "right": f"{fmt}/{len(ii)} | {neg} | {pres_i}/{n_i} | {pres_m}/{n_m}"})
    path = os.path.join(config.EXPERIMENT_OUT_DIR, "location_scores_v2.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "subset", "check", "reports", "checkable", "right", "right_rate",
                                          "shuffle_mean_rate", "shuffle_p5", "shuffle_p95", "shuffle_mean_checkable"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["check"].startswith("other"):
            print(f"{r['source']:14s} {r['subset']:9s} {r['check'][:5]}  {r['right']}")
        else:
            print(f"{r['source']:14s} {r['subset']:9s} {r['check']:11s} right {r['right']:>3}/{r['checkable']:<3} "
                  f"rate {r['right_rate']!s:>5} | shuffled chance {r['shuffle_mean_rate']:.3f} [{r['shuffle_p5']:.3f}-{r['shuffle_p95']:.3f}] "
                  f"(mean checkable {r['shuffle_mean_checkable']})")
    print("wrote", path)


if __name__ == "__main__":
    main()
