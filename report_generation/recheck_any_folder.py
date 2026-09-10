"""
recheck_any_folder.py

Same idea as recheck_region_grounding.py, but works on any output folder --
use this for outputs_region_grounded_medgemma/ or any future run, without
needing a new script each time.

Usage:
    python recheck_any_folder.py outputs_region_grounded_medgemma
"""

import os
import sys
import json
import glob

from prompts_region_grounded import check_region_grounding

if len(sys.argv) < 2:
    print("Usage: python recheck_any_folder.py <folder_name>")
    print("Example: python recheck_any_folder.py outputs_region_grounded_medgemma")
    sys.exit(1)

out_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(out_dir, "*.json")))

if not files:
    print(f"No files found in {out_dir}/")
else:
    print(f"Rechecking {len(files)} files in {out_dir}/ with the current checker...\n")
    for path in files:
        with open(path) as f:
            data = json.load(f)

        old_check = data.get("grounding_check", {})
        new_check = check_region_grounding(data["output"])
        data["grounding_check"] = new_check

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        old_status = old_check.get("likely_fully_tagged", "?")
        new_status = new_check["likely_fully_tagged"]
        changed = " <- CHANGED" if old_status != new_status else ""
        extra_flags = []
        if new_check.get("tags_outside_findings"):
            extra_flags.append("tags outside Findings")
        if new_check.get("duplicated_impression"):
            extra_flags.append("duplicated Impression")
        flag_str = f" [{', '.join(extra_flags)}]" if extra_flags else ""

        print(f"{data.get('patient_id', path)}: was {old_status}, now {new_status}{changed}{flag_str}")

    print("\nDone.")
