"""
reapply_reference.py

Re-applies the reference-dependent accept rules of mask_diversity.evaluate()
(contiguity inside the real range; MVO touch not below the real minimum) to the
200 inserts, using the CURRENT mask_diversity_reference.json (TRAIN pathological
cases only) and, for comparison, the earlier reference (67 cases, which
included the test patients). The other accept rules (area ratio, lesion leaves
the wall, lesion lost) do not depend on the reference and are not repeated.

Writes experiments/insert_list_v2.txt: every one of the 200 files, tab separated
as  file  included|excluded  reason. Files are never deleted.

    python experiments/reapply_reference.py [--old PATH_TO_OLD_REFERENCE_JSON]
"""

import os
import glob
import argparse

import common
from common import config, md, load_nii

OLD_DEFAULT = os.path.expanduser("~/private_notes/mask_diversity_reference_67cases_old.json")


def rejection(m, ref):
    why = []
    c = m["contiguity"]
    if c is None:
        why.append("contiguity undefined")
    elif not ref["contiguity"]["min"] <= c <= ref["contiguity"]["max"]:
        why.append(f"contiguity {c:.3f} outside {ref['contiguity']['min']:.3f}-{ref['contiguity']['max']:.3f}")
    if m["mvo_px"] > 0:
        t = m["mvo_touch"]
        if t is None or t < ref["mvo_touch"]["min"]:
            why.append(f"MVO touch {t if t is None else round(t, 3)} < real min {ref['mvo_touch']['min']:.3f}")
    return why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=OLD_DEFAULT)
    a = ap.parse_args()
    import json
    new_ref, old_ref = md.load_reference(), json.load(open(a.old))
    print(f"new reference: {new_ref.get('source')}, n={new_ref['contiguity']['n']}, contiguity "
          f"{new_ref['contiguity']['min']:.4f}-{new_ref['contiguity']['max']:.4f}, MVO touch min {new_ref['mvo_touch']['min']:.4f}")
    print(f"old reference: n={old_ref['contiguity']['n']} (all Case_P*, incl. test), contiguity "
          f"{old_ref['contiguity']['min']:.4f}-{old_ref['contiguity']['max']:.4f}, MVO touch min {old_ref['mvo_touch']['min']:.4f}")
    rows, changed = [], []
    for job in common.JOBS:
        for p in sorted(glob.glob(f"{config.FULL_INSERT_DIR}/input_{job}/labels/*.nii.gz")):
            name = os.path.basename(p)
            m = md.lesion_metrics(load_nii(p).astype(int))
            old, new = rejection(m, old_ref), rejection(m, new_ref)
            rows.append((name, new))
            if bool(old) != bool(new):
                changed.append((name, old, new))
    print(f"rejected under the old reference: {sum(bool(rejection(md.lesion_metrics(load_nii(p).astype(int)), old_ref)) for p in glob.glob(config.FULL_INSERT_DIR + '/input_*/labels/*.nii.gz'))} of {len(rows)}")
    print(f"excluded under the new reference: {sum(bool(w) for _, w in rows)} of {len(rows)}; included {sum(not w for _, w in rows)}")
    for name, old, new in changed:
        print(f"  decision changes: {name}  was {'rejected' if old else 'accepted'}, now {'rejected: ' + '; '.join(new) if new else 'accepted'}")
    out = os.path.join(common.HERE, "insert_list_v2.txt")
    with open(out, "w") as f:
        for name, why in sorted(rows):
            f.write(f"{name}\t{'excluded' if why else 'included'}\t{'; '.join(why)}\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
