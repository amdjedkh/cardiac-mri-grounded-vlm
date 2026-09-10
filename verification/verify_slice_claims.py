"""
verify_slice_claims.py

Turns fuzzy report language ("near-circumferential", "distinct area", "central")
into precise, independently-computed numbers from the REAL segmentation mask,
so a claim can be checked against geometry rather than eyeballed.

For one slice, computes:
  - infarct angular coverage around the LV cavity (in degrees, and % of 360)
  - MVO angular coverage
  - MVO area as % of infarct area, ON THIS SLICE (2D), independent of and
    cross-checkable against the whole-volume % already in measurements.py
  - MVO's position along the infarct arc (0 = one edge, 0.5 = dead center,
    1 = other edge) -- this is what actually tests a word like "central"

Also renders a simple polar diagram: infarct arc in red, MVO arc in yellow,
degree-labeled, which is a strong two-second visual for a meeting.

Usage:
    python verify_slice_claims.py Case_P019 4
    (case id, slice index -- slice index must match the slice used in the
    region-grounded report you're checking, e.g. slice_04 -> 4)
"""

import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_LABEL_MAP = {"background": 0, "lv_cavity": 1, "myocardium_normal": 2, "infarct": 3, "mvo": 4}


def load_slice_mask(case_id: str, slice_idx: int, root: str = ".") -> np.ndarray:
    import os
    mask_path = os.path.join(root, case_id, "Contours", f"{case_id}.nii.gz")
    img = nib.load(mask_path)
    data = np.asarray(img.get_fdata()).round().astype(int)
    return data[:, :, slice_idx]


def angular_coverage(mask_2d: np.ndarray, label: int, center: tuple) -> dict:
    """Returns the angular span (degrees) covered by a label's pixels around
    a center point, handling the case where the region is one contiguous arc
    (finds the coverage as 360 minus the largest empty gap)."""
    ys, xs = np.where(mask_2d == label)
    if len(ys) == 0:
        return {"present": False, "coverage_deg": 0.0, "angles_deg": []}

    cy, cx = center
    angles = np.degrees(np.arctan2(ys - cy, xs - cx)) % 360
    angles_sorted = np.sort(np.unique(np.round(angles).astype(int)) % 360)

    if len(angles_sorted) == 1:
        return {"present": True, "coverage_deg": 1.0, "angles_deg": angles_sorted.tolist(),
                "arc_start": float(angles_sorted[0]), "arc_end": float(angles_sorted[0])}

    # find the largest gap between consecutive angles (wrapping around 360->0)
    gaps = np.diff(angles_sorted)
    wrap_gap = 360 - (angles_sorted[-1] - angles_sorted[0])
    all_gaps = np.append(gaps, wrap_gap)
    max_gap_idx = np.argmax(all_gaps)

    if max_gap_idx == len(gaps):  # the wrap-around gap is the largest
        arc_start, arc_end = angles_sorted[0], angles_sorted[-1]
    else:
        arc_start = angles_sorted[max_gap_idx + 1]
        arc_end = angles_sorted[max_gap_idx]

    coverage = 360 - all_gaps[max_gap_idx]
    return {
        "present": True,
        "coverage_deg": float(coverage),
        "coverage_pct_of_circle": float(coverage / 360 * 100),
        "arc_start_deg": float(arc_start),
        "arc_end_deg": float(arc_end),
        "angles_deg": angles_sorted.tolist(),
    }


def contiguity_score(mask_2d: np.ndarray, label: int, center: tuple) -> dict:
    """Measures whether a label's angular arc is a smooth, solid band or a
    scattered/fragmented pattern. Within the arc span found by
    angular_coverage, checks what fraction of individual whole-degree
    positions actually contain at least one pixel of that label.

    1.0 = every degree within the arc span has label pixels (solid arc,
    like the real EMIDEC cases we've checked). Lower values mean gaps
    inside the span -- a scattered/speckled pattern rather than a
    contiguous region.
    """
    cov = angular_coverage(mask_2d, label, center)
    if not cov["present"] or cov["coverage_deg"] <= 1:
        return {"present": cov["present"], "contiguity": None, "coverage_deg": cov.get("coverage_deg", 0.0)}

    angles_present = set(cov["angles_deg"])
    arc_start, arc_end = cov["arc_start_deg"], cov["arc_end_deg"]

    # walk the arc span degree by degree (handling wraparound past 360),
    # counting how many of those degree positions actually have the label
    span = int(round(cov["coverage_deg"]))
    covered_count = 0
    for i in range(span + 1):
        deg = int(round(arc_start + i)) % 360
        if deg in angles_present:
            covered_count += 1

    contiguity = covered_count / (span + 1)
    return {
        "present": True,
        "contiguity": round(contiguity, 3),
        "coverage_deg": cov["coverage_deg"],
        "degrees_with_label": covered_count,
        "degrees_in_span": span + 1,
    }


def position_along_arc(point_angle: float, arc_start: float, arc_end: float) -> float:
    """Returns 0-1: where point_angle falls between arc_start and arc_end,
    going the short way around. 0.5 = exactly centered in the arc."""
    span = (arc_end - arc_start) % 360
    if span == 0:
        span = 360
    offset = (point_angle - arc_start) % 360
    return round(offset / span, 3)


def analyze_slice_array(mask_2d: np.ndarray, case_id: str = "test", slice_idx: int = 0, label_map: dict = None):
    lm = label_map or DEFAULT_LABEL_MAP

    cavity_ys, cavity_xs = np.where(mask_2d == lm["lv_cavity"])
    if len(cavity_ys) == 0:
        raise ValueError(f"No LV cavity found on {case_id} slice {slice_idx} -- wrong slice?")
    center = (cavity_ys.mean(), cavity_xs.mean())

    infarct_cov = angular_coverage(mask_2d, lm["infarct"], center)
    mvo_cov = angular_coverage(mask_2d, lm["mvo"], center)

    infarct_area = int(np.sum(mask_2d == lm["infarct"]))
    mvo_area = int(np.sum(mask_2d == lm["mvo"]))
    mvo_pct_of_infarct_this_slice = round(mvo_area / infarct_area * 100, 1) if infarct_area > 0 else 0.0

    mvo_position = None
    if infarct_cov["present"] and mvo_cov["present"]:
        mvo_ys, mvo_xs = np.where(mask_2d == lm["mvo"])
        mvo_centroid_angle = np.degrees(np.arctan2(
            mvo_ys.mean() - center[0], mvo_xs.mean() - center[1]
        )) % 360
        mvo_position = position_along_arc(
            mvo_centroid_angle, infarct_cov["arc_start_deg"], infarct_cov["arc_end_deg"]
        )

    return {
        "case_id": case_id,
        "slice_idx": slice_idx,
        "cavity_center_rc": (round(center[0], 1), round(center[1], 1)),
        "infarct": infarct_cov,
        "mvo": mvo_cov,
        "mvo_area_pct_of_infarct_this_slice": mvo_pct_of_infarct_this_slice,
        "mvo_position_along_infarct_arc": mvo_position,
        "mask_2d": mask_2d,
        "center": center,
        "label_map": lm,
    }


def analyze_slice(case_id: str, slice_idx: int, root: str = ".", label_map: dict = None):
    mask_2d = load_slice_mask(case_id, slice_idx, root)
    return analyze_slice_array(mask_2d, case_id, slice_idx, label_map)


def fact_check_claims(analysis: dict) -> list:
    """Turns the computed numbers into plain statements checking common report
    phrasing against what's actually measurable."""
    checks = []
    infarct = analysis["infarct"]
    mvo = analysis["mvo"]

    if infarct["present"]:
        pct = infarct["coverage_pct_of_circle"]
        near_circumferential = pct >= 60
        checks.append(
            f"infarct angular coverage: {pct:.1f}% of the ring "
            f"({'supports' if near_circumferential else 'does NOT clearly support'} "
            f"a \"near-circumferential\" description)"
        )
    else:
        checks.append("infarct not present on this slice per the mask")

    if mvo["present"]:
        pct_area = analysis["mvo_area_pct_of_infarct_this_slice"]
        checks.append(f"MVO area is {pct_area:.1f}% of infarct area on this slice")

        pos = analysis["mvo_position_along_infarct_arc"]
        if pos is not None:
            if 0.35 <= pos <= 0.65:
                pos_desc = "roughly central within the infarct arc"
            elif pos < 0.35:
                pos_desc = "toward one edge of the infarct arc, not central"
            else:
                pos_desc = "toward the other edge of the infarct arc, not central"
            checks.append(
                f"MVO position along infarct arc: {pos:.2f} (0=edge, 0.5=center, 1=edge) "
                f"-- {pos_desc}"
            )
    else:
        checks.append("MVO not present on this slice per the mask")

    return checks


def render_polar_diagram(analysis: dict, out_path: str):
    infarct = analysis["infarct"]
    mvo = analysis["mvo"]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    ax.plot(np.linspace(0, 2 * np.pi, 360), [1] * 360, color="#CCCCCC", linewidth=8, alpha=0.5)

    if infarct["present"]:
        start_rad = np.radians(infarct["arc_start_deg"])
        span_rad = np.radians(infarct["coverage_deg"])
        theta = np.linspace(start_rad, start_rad + span_rad, 200)
        ax.plot(theta, [1] * len(theta), color="#E24A4A", linewidth=10,
                label=f"infarct ({infarct['coverage_pct_of_circle']:.0f}% of ring)")

    if mvo["present"]:
        start_rad = np.radians(mvo["arc_start_deg"])
        span_rad = np.radians(mvo["coverage_deg"])
        theta = np.linspace(start_rad, start_rad + span_rad, 100)
        ax.plot(theta, [1] * len(theta), color="#E2C34A", linewidth=10,
                label=f"MVO ({analysis['mvo_area_pct_of_infarct_this_slice']:.0f}% of infarct area)")

    ax.set_yticklabels([])
    ax.set_title(f"{analysis['case_id']} slice {analysis['slice_idx']}\n"
                 f"quantified region coverage around the LV ring",
                 fontsize=10, pad=20)
    if infarct["present"] or mvo["present"]:
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8, frameon=False)
    else:
        ax.text(0, 0, "no infarct or MVO\npresent on this slice", ha="center", va="center",
                 fontsize=9, color="#555555", transform=ax.transData)

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_slice_claims.py <Case_ID> <slice_index>")
        print("Example: python verify_slice_claims.py Case_P019 4")
        sys.exit(1)

    case_id = sys.argv[1]
    slice_idx = int(sys.argv[2])

    analysis = analyze_slice(case_id, slice_idx)

    print(f"\n=== Quantitative check: {case_id}, slice {slice_idx} ===\n")
    for line in fact_check_claims(analysis):
        print(f"- {line}")

    print("\n=== Contiguity (smooth arc vs scattered) ===")
    for region_name, label_key in [("infarct", "infarct"), ("mvo", "mvo")]:
        region_data = analysis[label_key]
        if not region_data["present"]:
            print(f"- {region_name}: not present on this slice")
            continue
        cscore = contiguity_score(analysis["mask_2d"], analysis["label_map"][label_key], analysis["center"])
        if cscore["contiguity"] is None:
            print(f"- {region_name}: present but too small a span to score contiguity")
        else:
            print(f"- {region_name}: contiguity = {cscore['contiguity']:.3f} "
                  f"({cscore['degrees_with_label']}/{cscore['degrees_in_span']} degrees within the "
                  f"{cscore['coverage_deg']:.1f}\u00b0 span actually contain the label) "
                  f"-- {'smooth/solid' if cscore['contiguity'] >= 0.8 else 'fragmented/scattered' if cscore['contiguity'] < 0.5 else 'somewhat patchy'}")

    out_path = f"{case_id}_slice{slice_idx}_polar_check.png"
    render_polar_diagram(analysis, out_path)
    print(f"\nSaved polar diagram to {out_path}")
