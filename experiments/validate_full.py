"""
validate_full.py

Checks the 200-volume LeFusion insert run in FULL_INSERT_DIR (config.py):
  - 200 volumes, input and output file names identical per job
  - output images finite
  - output mask equals the centre-padded input mask
  - no file belongs to a test-split patient
Exits non-zero if any check fails.

    python experiments/validate_full.py
"""

import sys

import numpy as np

import common
from common import config, source_patient, load_split

EXPECTED = 200


def main():
    test = set(load_split()["test"])
    n, problems = 0, []
    for v in common.iter_volumes():
        n += 1
        name = v["file"]
        if source_patient(name) in test:
            problems.append((name, "test-split patient"))
        if not np.isfinite(v["out_img"]).all():
            problems.append((name, "non-finite output image"))
        if common.locate(v["lab_in"], v["out_mask"]) is None:
            problems.append((name, "output mask differs from padded input mask"))
    if n != EXPECTED:
        problems.append(("count", f"{n} volumes, expected {EXPECTED}"))
    print(f"{n} volumes checked in {config.FULL_INSERT_DIR}; {len(problems)} problems")
    for p in problems:
        print("  ", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
