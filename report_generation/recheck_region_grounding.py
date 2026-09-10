"""
recheck_region_grounding.py

Re-runs the (now fixed) grounding checker against the outputs you already
generated in outputs_region_grounded/, without calling the API again.
The earlier checker had a bug where markdown bold formatting (**Findings:**
style headers) caused a false "not fully tagged" result on real model output
-- this re-scores everything already saved using the corrected logic.

Usage:
    python recheck_region_grounding.py
"""

import os
import json
import glob

from prompts_region_grounded import check_region_grounding

out_dir = "outputs_region_grounded"
files = sorted(glob.glob(os.path.join(out_dir, "*.json")))

if not files:
    print(f"No files found in {out_dir}/ -- nothing to recheck.")
else:
    print(f"Rechecking {len(files)} saved outputs with the fixed checker...\n")
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
        print(f"{data['patient_id']}: was {old_status}, now {new_status}"
              f" ({new_check['n_region_tags_found']} tags, "
              f"invalid: {new_check['invalid_region_names']}){changed}")

    print("\nDone. Files updated in place with corrected grounding_check values.")
