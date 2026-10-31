"""
assign_aha17_segment.py

Applies the AHA17 pipeline for real, on a real case, instead of just
describing it. Reuses the exact angle math from verify_slice_claims.py
(already tested against synthetic geometry with known answers).

What this actually resolves: on Case_P004, the report claimed the infarct
was in the "anteroseptal wall." This script computes which AHA17 angular
sector the infarct's centroid actually falls into, so that claim can be
checked against something real instead of being left as "unverified."

ORIENTATION (confirmed by author inspection of 5 cases, pending cardiologist confirmation: P019, P004, P055, P010, P029;
the NIfTI headers carry no usable orientation, so this rests on the images):
  - array slice [axis0, axis1] is read with axis0 as rows (row 0 = image top)
    and axis1 as columns, no flip
  - anterior = image top
  - septal side (RV side) = image right, lateral = image left
  - slice index increases toward the BASE: slice 0 = apex, last slice = base
  Limitation: base/mid/apical is a coarse thirds split of the slice stack on 5-9 slices (4-10 across EMIDEC), not a precise measurement.

Usage:
    python assign_aha17_segment.py Case_P004 3
"""

import sys
from verify_slice_claims import analyze_slice, DEFAULT_LABEL_MAP
import json
import os

# Standard AHA17 angular sectors (6-way for basal/mid rings), sector centers
# in "bullseye degrees" where 0 = anterior (top), clockwise.
SECTOR_CENTERS_6 = {
    "anterior": 0, "anteroseptal": 60, "inferoseptal": 120,
    "inferior": 180, "inferolateral": 240, "anterolateral": 300,
}
# 4-way for the apical ring
SECTOR_CENTERS_4 = {
    "apical anterior": 0, "apical septal": 90, "apical inferior": 180, "apical lateral": 270,
}

SEGMENT_NUMBERS = {
    ("basal", "anterior"): 1, ("basal", "anteroseptal"): 2, ("basal", "inferoseptal"): 3,
    ("basal", "inferior"): 4, ("basal", "inferolateral"): 5, ("basal", "anterolateral"): 6,
    ("mid", "anterior"): 7, ("mid", "anteroseptal"): 8, ("mid", "inferoseptal"): 9,
    ("mid", "inferior"): 10, ("mid", "inferolateral"): 11, ("mid", "anterolateral"): 12,
    ("apical", "apical anterior"): 13, ("apical", "apical septal"): 14,
    ("apical", "apical inferior"): 15, ("apical", "apical lateral"): 16,
}


def to_bullseye_angle(raw_angle_deg: float) -> float:
    """Converts the raw angle (0=image-right, increasing clockwise since y
    grows downward -- the convention already used and tested in
    verify_slice_claims.py) into 'bullseye degrees' where 0=top(anterior),
    clockwise. Image top = anterior and image right = septal (author-inspected,
    pending cardiologist confirmation; see module docstring)."""
    return (raw_angle_deg - 270) % 360


def nearest_sector(bullseye_angle: float, centers: dict) -> str:
    best_name, best_dist = None, 999
    for name, center in centers.items():
        dist = min(abs(bullseye_angle - center), 360 - abs(bullseye_angle - center))
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name


def determine_ring_level(slice_idx: int, total_slices: int) -> str:
    """Basal / mid / apical based on where this slice sits in the stack.
    Slice index increases toward the base: slice 0 = apex, last slice = base
    (author inspection of 5 cases, pending cardiologist confirmation; cavity
    area also grows with slice index in all 100 EMIDEC cases). This is a
    coarse thirds split on 5-9 slices (4-10 across EMIDEC), not a precise
    measurement."""
    frac = 1 - slice_idx / max(total_slices - 1, 1)  # 0 at the base, 1 at the apex
    if frac < 0.33:
        return "basal"
    elif frac < 0.67:
        return "mid"
    else:
        return "apical"


def assign_segment(case_id: str, slice_idx: int, root: str = ".") -> dict:
    analysis = analyze_slice(case_id, slice_idx, root)
    infarct = analysis["infarct"]
    mvo = analysis["mvo"]

    with open(os.path.join(root, "patients", f"{case_id}.json")) as f:
        patient = json.load(f)
    total_slices = patient["measurements"]["total_slices"]

    level = determine_ring_level(slice_idx, total_slices)
    result = {"case_id": case_id, "slice_idx": slice_idx, "total_slices": total_slices,
              "ring_level": level, "infarct": None, "mvo": None}

    for region_name, region_data in [("infarct", infarct), ("mvo", mvo)]:
        if not region_data["present"]:
            result[region_name] = {"present": False}
            continue

        # centroid angle across the whole present arc, using the arc midpoint
        arc_start, arc_end = region_data["arc_start_deg"], region_data["arc_end_deg"]
        span = (arc_end - arc_start) % 360 or 360
        centroid_angle = (arc_start + span / 2) % 360
        bullseye = to_bullseye_angle(centroid_angle)

        if level == "apical":
            sector = nearest_sector(bullseye, SECTOR_CENTERS_4)
            seg_num = SEGMENT_NUMBERS.get((level, sector))
        else:
            sector = nearest_sector(bullseye, SECTOR_CENTERS_6)
            seg_num = SEGMENT_NUMBERS.get((level, sector))

        result[region_name] = {
            "present": True,
            "centroid_raw_angle_deg": round(centroid_angle, 1),
            "centroid_bullseye_angle_deg": round(bullseye, 1),
            "assigned_sector": sector,
            "aha_segment_number": seg_num,
            "aha_segment_name": f"{level} {sector}" if level != "apical" else sector,
        }

    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python assign_aha17_segment.py <Case_ID> <slice_index>")
        sys.exit(1)

    case_id, slice_idx = sys.argv[1], int(sys.argv[2])
    result = assign_segment(case_id, slice_idx)

    print(f"\n=== AHA17 segment assignment: {result['case_id']}, slice {result['slice_idx']} "
          f"of {result['total_slices']} ===")
    print(f"Ring level: {result['ring_level']}  "
          f"(slice {result['slice_idx']}/{result['total_slices']-1} along the stack)")
    print()
    for region in ["infarct", "mvo"]:
        r = result[region]
        if not r["present"]:
            print(f"{region}: not present on this slice")
            continue
        print(f"{region}: segment {r['aha_segment_number']} -- \"{r['aha_segment_name']}\"")
        print(f"  (centroid at {r['centroid_bullseye_angle_deg']}\u00b0 bullseye angle, "
              f"nearest named sector)")

    print("\nOrientation used: image top = anterior, image right = septal, slice 0 = apex (last slice = base).")
    print("(Author inspection of 5 cases, pending cardiologist confirmation. Base/mid/apical is a coarse thirds split.)")
