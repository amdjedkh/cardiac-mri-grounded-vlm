"""
test_assign_aha17_segment.py

Unit test for assign_aha17_segment.assign_segment using a synthetic LV with
lesions placed at KNOWN angles.

The convention it encodes (confirmed by author inspection of 5 cases, pending cardiologist confirmation: P019, P004, P055, P010, P029):
    image top (row 0 side) = anterior, clockwise on screen,
    septal sectors toward image right, lateral toward image left,
    slice 0 = APEX, last slice = BASE (high slice index = base).
This checks the arithmetic of that convention; the convention itself was
confirmed by author inspection of 5 cases, pending cardiologist confirmation, not by this test.
Limitation: base/mid/apical is a coarse thirds split of the slice stack on 5-9 slices (4-10 across EMIDEC), not a precise measurement.

Run:
    python -m unittest verification/test_assign_aha17_segment.py   (from repo root)
    python verification/test_assign_aha17_segment.py
"""

import os
import sys
import json
import tempfile
import unittest
import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from assign_aha17_segment import assign_segment

N_SLICES = 9
SIZE = 64
CENTER = (32.0, 32.0)  # (row, col)


def make_ring_slice(lesion_bullseye_deg=None, lesion_label=3, half_width_deg=15):
    """Cavity (label 1) r<6, myocardium (label 2) 6<=r<12. If given, pixels of
    the myocardium within +-half_width_deg of the bullseye angle (0 = image top,
    clockwise) get lesion_label."""
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    dy, dx = rows - CENTER[0], cols - CENTER[1]
    r = np.hypot(dy, dx)
    sl = np.zeros((SIZE, SIZE), dtype=int)
    sl[r < 6] = 1
    ring = (r >= 6) & (r < 12)
    sl[ring] = 2
    if lesion_bullseye_deg is not None:
        # raw angle used by the script: arctan2(row offset, col offset), 0 = image right,
        # growing clockwise on screen. bullseye 0 (top) = raw 270.
        raw = np.degrees(np.arctan2(dy, dx)) % 360
        target_raw = (lesion_bullseye_deg + 270) % 360
        diff = np.minimum(abs(raw - target_raw), 360 - abs(raw - target_raw))
        sl[ring & (diff <= half_width_deg)] = lesion_label
    return sl


def build_case(root, case_id, lesions_by_slice):
    """lesions_by_slice: {slice_idx: [(bullseye_deg, label), ...]}"""
    vol = np.zeros((SIZE, SIZE, N_SLICES), dtype=np.int16)
    for k in range(N_SLICES):
        sl = make_ring_slice()
        for deg, label in lesions_by_slice.get(k, []):
            lesion = make_ring_slice(deg, label)
            sl = np.where(lesion == label, label, sl)
        vol[:, :, k] = sl
    os.makedirs(os.path.join(root, case_id, "Contours"))
    nib.save(nib.Nifti1Image(vol, np.eye(4)), os.path.join(root, case_id, "Contours", f"{case_id}.nii.gz"))
    os.makedirs(os.path.join(root, "patients"), exist_ok=True)
    with open(os.path.join(root, "patients", f"{case_id}.json"), "w") as f:
        json.dump({"measurements": {"total_slices": N_SLICES}}, f)


# (slice, lesion bullseye degrees) -> expected (ring level, sector, AHA segment number)
EXPECTED = [
    (8, 0,   ("basal", "anterior", 1)),
    (8, 60,  ("basal", "anteroseptal", 2)),
    (8, 120, ("basal", "inferoseptal", 3)),
    (8, 180, ("basal", "inferior", 4)),
    (8, 240, ("basal", "inferolateral", 5)),
    (8, 300, ("basal", "anterolateral", 6)),
    (4, 60,  ("mid", "anteroseptal", 8)),
    (4, 240, ("mid", "inferolateral", 11)),
    (0, 90,  ("apical", "apical septal", 14)),
    (0, 270, ("apical", "apical lateral", 16)),
]


class TestAssignSegment(unittest.TestCase):
    def test_known_lesions(self):
        with tempfile.TemporaryDirectory() as root:
            for i, (slice_idx, deg, _) in enumerate(EXPECTED):
                build_case(root, f"Case_T{i:02d}", {slice_idx: [(deg, 3)]})
            for i, (slice_idx, deg, (level, sector, seg)) in enumerate(EXPECTED):
                r = assign_segment(f"Case_T{i:02d}", slice_idx, root)["infarct"]
                with self.subTest(slice=slice_idx, lesion_deg=deg):
                    self.assertTrue(r["present"])
                    self.assertEqual(assign_segment(f"Case_T{i:02d}", slice_idx, root)["ring_level"], level)
                    self.assertEqual(r["assigned_sector"], sector)
                    self.assertEqual(r["aha_segment_number"], seg)
                    # centroid angle should land close to where the lesion was drawn
                    self.assertLess(min(abs(r["centroid_bullseye_angle_deg"] - deg),
                                        360 - abs(r["centroid_bullseye_angle_deg"] - deg)), 6)

    def test_mvo_assigned_separately(self):
        with tempfile.TemporaryDirectory() as root:
            build_case(root, "Case_T99", {8: [(0, 3), (180, 4)]})
            r = assign_segment("Case_T99", 8, root)
            self.assertEqual(r["infarct"]["assigned_sector"], "anterior")
            self.assertEqual(r["mvo"]["assigned_sector"], "inferior")

    def test_base_apex_order(self):
        """High slice index = base, slice 0 = apex."""
        with tempfile.TemporaryDirectory() as root:
            build_case(root, "Case_T97", {0: [(60, 3)], N_SLICES - 1: [(60, 3)]})
            first = assign_segment("Case_T97", 0, root)
            last = assign_segment("Case_T97", N_SLICES - 1, root)
            self.assertEqual(first["ring_level"], "apical")
            self.assertEqual(last["ring_level"], "basal")
            self.assertEqual(last["infarct"]["aha_segment_number"], 2)   # basal anteroseptal
            self.assertEqual(first["infarct"]["aha_segment_number"], 14) # apical septal (60 deg -> nearest of 0/90)
            # ring level steps monotonically from apex (0) to base (last)
            order = {"apical": 0, "mid": 1, "basal": 2}
            levels = [order[assign_segment("Case_T97", k, root)["ring_level"]] for k in range(N_SLICES)]
            self.assertEqual(levels, sorted(levels))

    def test_slice_without_lesion(self):
        with tempfile.TemporaryDirectory() as root:
            build_case(root, "Case_T98", {})
            r = assign_segment("Case_T98", 8, root)
            self.assertFalse(r["infarct"]["present"])
            self.assertFalse(r["mvo"]["present"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
