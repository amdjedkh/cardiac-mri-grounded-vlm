"""
check_split_leakage.py

Exits non-zero if any test-set patient (splits/split_v1.json) appears in a
training or synthetic list. Checks:

  1. the split's own train list
  2. synthetic case folders (Image/ and Mask/ file names, __dup suffix
     stripped back to the source patient) -- default: synthetic_cases and
     synthetic_samples next to the repo root
  3. the LeFusion input-pool filter used by synthetic_generation/
     modal_lefusion.py, run against a pool containing every real case, and a
     check that modal_lefusion.py actually calls it
  4. any extra lists passed with --list (a .json/.jsonl/.txt/.csv file; every
     token that looks like a case id is read). Use this for exported
     accepted-case manifests, training lists, and so on.

Usage:
    python splits/check_split_leakage.py
    python splits/check_split_leakage.py --list annotations_export.json --synthetic-dir /path/to/synthetic_cases
"""

import os
import re
import sys
import glob
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from split_utils import load_split, source_patient, filter_input_pool

ID_PATTERN = re.compile(r"Case[_ ]?[NP]\d+")


def ids_in_file(path: str) -> set:
    with open(path, errors="ignore") as f:
        text = f.read()
    return {source_patient(m) for m in ID_PATTERN.findall(text)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=os.path.join(HERE, "split_v1.json"))
    ap.add_argument("--synthetic-dir", action="append", default=None,
                    help="synthetic folder to scan (repeatable). Default: synthetic_cases and synthetic_samples")
    ap.add_argument("--list", action="append", default=[], help="extra list file to scan (repeatable)")
    args = ap.parse_args()

    split = load_split(args.split)
    test = set(split["test"])
    print(f"Split {split['version']}: {len(test)} test patients, seed {split['seed']}")

    failures = []  # (where, patient, detail)

    # 1. train list
    for p in split["train"]:
        if p in test:
            failures.append(("split train list", p, "also in test"))
    print(f"[1] train list: {len(split['train'])} patients")

    # 2. synthetic folders
    syn_dirs = args.synthetic_dir or [os.path.join(REPO_ROOT, "synthetic_cases"), os.path.join(REPO_ROOT, "synthetic_samples")]
    for d in syn_dirs:
        files = [f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True) if os.path.isfile(f)]
        hits = [(f, source_patient(f)) for f in files if source_patient(f) in test]
        print(f"[2] synthetic folder {os.path.relpath(d, REPO_ROOT)}: {len(files)} files, {len(hits)} from test patients")
        for f, p in hits:
            failures.append((f"synthetic: {os.path.relpath(f, REPO_ROOT)}", p, "synthetic file derived from a test patient"))

    # 3. LeFusion input pool
    pool = [f"{p}.nii.gz" for p in split["patients"]]
    survivors = filter_input_pool(pool, test)
    leaked = [f for f in survivors if source_patient(f) in test]
    print(f"[3] LeFusion pool filter: {len(pool)} in, {len(survivors)} out, {len(leaked)} test patients surviving")
    for f in leaked:
        failures.append(("LeFusion pool filter", source_patient(f), "survives the filter"))
    modal_src = open(os.path.join(REPO_ROOT, "synthetic_generation", "modal_lefusion.py")).read()
    if "filter_input_pool" not in modal_src or "split_v1.json" not in modal_src:
        failures.append(("modal_lefusion.py", "(config)", "does not apply the split filter to its input pool"))
        print("[3] modal_lefusion.py does NOT apply the split filter")
    else:
        print("[3] modal_lefusion.py applies the split filter")

    # 4. extra lists
    for path in args.list:
        found = ids_in_file(path)
        hits = sorted(found & test)
        print(f"[4] {path}: {len(found)} patients, {len(hits)} in test set")
        for p in hits:
            failures.append((path, p, "test patient in list"))

    print()
    if failures:
        by_patient = {}
        for where, p, detail in failures:
            by_patient.setdefault(p, []).append(where)
        print(f"FAIL: {len(by_patient)} test patient(s) leak, {len(failures)} offending entries")
        for p in sorted(by_patient):
            print(f"  {p}: {len(by_patient[p])} entries, e.g. {by_patient[p][0]}")
        sys.exit(1)
    print("PASS: no test patient appears in any checked list")


if __name__ == "__main__":
    main()
