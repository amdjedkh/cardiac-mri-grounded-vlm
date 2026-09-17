"""
slice_consistency_metrics.py

Shared logic for measuring whether a 3D volume's slices are anatomically
coherent (a real heart tapering smoothly from base to apex) or independent/
erratic (as if each slice were generated without knowledge of its
neighbors).

For each slice: cavity centroid position and area, myocardium area.
Across the volume: how much do these change slice-to-slice, and how
CONSISTENT is that change. A real heart's slice-to-slice change should be
small and fairly steady (smooth tapering). Erratic generation would show
large, inconsistent jumps.

Not a script on its own -- imported by scan_real_consistency.py and the
matching Modal function for synthetic cases.
"""

import statistics


def compute_slice_profile(mask_3d, label_cavity=1, label_myo=2):
    """For each slice, cavity centroid (y, x), cavity area, myocardium area.
    Slices with no cavity present get centroid=None (e.g. far apical/basal
    slices where the LV cavity hasn't started or has tapered to nothing)."""
    profile = []
    for s in range(mask_3d.shape[2]):
        mask_2d = mask_3d[:, :, s]
        cavity_ys, cavity_xs = (mask_2d == label_cavity).nonzero()
        myo_area = int((mask_2d == label_myo).sum())
        if len(cavity_ys) == 0:
            profile.append({"slice": s, "cavity_area": 0, "myo_area": myo_area, "centroid": None})
            continue
        centroid = (float(cavity_ys.mean()), float(cavity_xs.mean()))
        profile.append({"slice": s, "cavity_area": int(len(cavity_ys)), "myo_area": myo_area, "centroid": centroid})
    return profile


def compute_consistency_metrics(profile: list) -> dict:
    """Slice-to-slice consistency metrics from a profile built above.
    Returns None if there are fewer than 2 slices with a cavity to compare."""
    valid = [p for p in profile if p["centroid"] is not None]
    if len(valid) < 2:
        return None

    centroid_shifts = []
    area_changes_pct = []
    for i in range(len(valid) - 1):
        c1, c2 = valid[i]["centroid"], valid[i + 1]["centroid"]
        shift = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
        centroid_shifts.append(shift)

        a1, a2 = valid[i]["cavity_area"], valid[i + 1]["cavity_area"]
        rel_change = abs(a2 - a1) / max(a1, a2, 1)
        area_changes_pct.append(rel_change * 100)

    def safe_stdev(vals):
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    return {
        "n_valid_slices": len(valid),
        "mean_centroid_shift_px": round(statistics.mean(centroid_shifts), 2),
        "std_centroid_shift_px": round(safe_stdev(centroid_shifts), 2),
        "mean_area_change_pct": round(statistics.mean(area_changes_pct), 1),
        "std_area_change_pct": round(safe_stdev(area_changes_pct), 1),
        "max_area_change_pct": round(max(area_changes_pct), 1),
    }
