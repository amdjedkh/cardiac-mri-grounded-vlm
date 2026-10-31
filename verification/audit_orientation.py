"""
audit_orientation.py

Read-only audit of the orientation used by assign_aha17_segment.py.
Does NOT change any segment logic.

ORIENTATION USED (confirmed by author inspection of 5 cases, pending cardiologist confirmation: P019, P004, P055, P010,
P029; see orientation_check.png from make_orientation_check.py):
  - the array slice [axis0, axis1] is shown with axis0 as rows (row 0 = top)
    and axis1 as columns, no flip
  - anterior = image top
  - septal side (RV side) = image right; lateral = image left
  - slice index increases toward the BASE: slice 0 = apex, last = base
Data check printed below: in every case the cavity is larger in the
high-index half of the stack, consistent with high index = base.

The NIfTI headers are NOT a source of orientation: every file has the same
placeholder affine diag(-1, -1, 1) (axcodes LPS) with no translation and no
voxel size. What that placeholder would imply (anterior = image left, high
slice = superior) contradicts the orientation above, so axcodes are
printed only to show they are uninformative and are never used by the code.

For every case it prints the axis codes of the image and the mask, flags any
case that differs from the majority, and runs the slice-order data check.

Usage (EMIDEC root = folder holding Case_* directories):
    python verification/audit_orientation.py /path/to/emidec-dataset-1.0.1
"""

import os
import sys
import glob
from collections import Counter
import numpy as np
import nibabel as nib

OPPOSITE = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}
NAMES = {"L": "left", "R": "right", "A": "anterior", "P": "posterior", "S": "superior", "I": "inferior"}


def axcodes(path: str) -> tuple:
    return nib.aff2axcodes(nib.load(path).affine)


def implied_directions(codes: tuple) -> dict:
    """Axis codes name the direction each array axis INCREASES toward."""
    return {
        "image_top (axis0 decreasing)": NAMES[OPPOSITE[codes[0]]],
        "image_bottom (axis0 increasing)": NAMES[codes[0]],
        "image_right (axis1 increasing)": NAMES[codes[1]],
        "image_left (axis1 decreasing)": NAMES[OPPOSITE[codes[1]]],
        "slice index increasing (axis2)": NAMES[codes[2]],
    }


def main(root: str):
    rows = []
    cav_high_larger = []
    for case_dir in sorted(glob.glob(os.path.join(root, "Case_[NP]*"))):
        cid = os.path.basename(case_dir)
        ip = os.path.join(case_dir, "Images", f"{cid}.nii.gz")
        mp = os.path.join(case_dir, "Contours", f"{cid}.nii.gz")
        if not (os.path.exists(ip) and os.path.exists(mp)):
            continue
        ic, mc = axcodes(ip), axcodes(mp)
        rows.append((cid, ic, mc))
        flag = "" if ic == mc else "   <-- image/mask axcodes differ"
        print(f"{cid}: image {''.join(ic)}  mask {''.join(mc)}{flag}")
        m = np.asarray(nib.load(mp).dataobj).round().astype(int)
        area = [int((m[:, :, k] == 1).sum()) for k in range(m.shape[2])]
        half = len(area) // 2
        cav_high_larger.append(np.mean(area[half + len(area) % 2:]) > np.mean(area[:half]))

    print(f"\n{len(rows)} cases")
    counts = Counter(ic for _, ic, _ in rows)
    print("Image axcodes:", {''.join(k): v for k, v in counts.items()})
    print("Mask axcodes: ", {''.join(k): v for k, v in Counter(mc for _, _, mc in rows).items()})
    majority = counts.most_common(1)[0][0]
    odd = [cid for cid, ic, _ in rows if ic != majority]
    print("Cases not matching the majority image axcodes:", odd or "none")

    print(f"\nAffine-implied directions for axcodes {''.join(majority)} (placeholder header; NOT used, "
          "contradicts the inspected orientation):")
    for k, v in implied_directions(majority).items():
        print(f"  {k:36s} -> {v}")

    print(f"\nSlice-order data check: cavity area larger in the high-index half in "
          f"{sum(cav_high_larger)} of {len(cav_high_larger)} cases (high index = base).")

    print("\nOrientation used by assign_aha17_segment.py (confirmed by author inspection of 5 cases, pending cardiologist confirmation):")
    print("  anterior = image top | septal = image right | lateral = image left")
    print("  slice 0 = apex, last slice = base")
    print("  bullseye degrees, clockwise from image top:")
    print("  0 anterior | 60 anteroseptal | 120 inferoseptal | 180 inferior | 240 inferolateral | 300 anterolateral")
    print("  apical ring: 0 anterior | 90 septal | 180 inferior | 270 lateral")
    print("  The code never reads the affine or axcodes; this is fixed geometry.")
    print("  Limitation: base/mid/apical is a coarse thirds split of the slice stack on 5-9 slices (4-10 across EMIDEC), not a precise measurement.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
