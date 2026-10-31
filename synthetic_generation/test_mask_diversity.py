"""
test_mask_diversity.py

Checks mask_diversity.py on a synthetic LV with a lesion at a known place.
Run: python synthetic_generation/test_mask_diversity.py
"""
import os
import sys
import unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mask_diversity as md

SIZE, N = 80, 7
C = (40.0, 40.0)
REF = {"contiguity": {"min": 0.0, "max": 1.0}, "mvo_touch": {"min": 0.9}}


def ring(lesion_deg=None, half=25, mvo=False, z=range(2, 5)):
    """Cavity r<10, wall 10<=r<20. Lesion: wall pixels within +-half degrees of
    bullseye angle lesion_deg (raw array angle, 0 = +col), outer part of the wall
    only up to depth 0.6; MVO = inner third of the lesion."""
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    r = np.hypot(rows - C[0], cols - C[1])
    ang = np.degrees(np.arctan2(rows - C[0], cols - C[1])) % 360
    vol = np.zeros((SIZE, SIZE, N), dtype=int)
    for k in range(N):
        s = np.zeros((SIZE, SIZE), dtype=int)
        s[r < 10] = md.CAV
        s[(r >= 10) & (r < 20)] = md.MYO
        if lesion_deg is not None and k in z:
            d = np.minimum(abs(ang - lesion_deg), 360 - abs(ang - lesion_deg))
            les = (r >= 10) & (r < 20) & (d <= half)
            s[les] = md.INF
            if mvo:
                s[les & (r < 14)] = md.MVO
        vol[:, :, k] = s
    return vol


def mean_angle(mask, label):
    th = []
    for k in range(mask.shape[2]):
        ys, xs = np.nonzero(mask[:, :, k] == label)
        th.append(np.arctan2(ys - C[0], xs - C[1]))
    th = np.concatenate(th)
    return np.degrees(np.arctan2(np.sin(th).mean(), np.cos(th).mean())) % 360


class TestMaskDiversity(unittest.TestCase):
    def setUp(self):
        self.m = ring(lesion_deg=90, mvo=True)

    def test_identity_is_exact(self):
        new, leave, lost = md.transform_volume(self.m, md.Params())
        self.assertTrue(np.array_equal(new, self.m))
        self.assertEqual((leave, lost), (0.0, 0.0))

    def test_heart_unchanged_for_random_params(self):
        rng = np.random.RandomState(0)
        for _ in range(15):
            new, _, _ = md.transform_volume(self.m, md.sample_params(rng))
            self.assertTrue(np.array_equal(new == md.CAV, self.m == md.CAV))
            self.assertTrue(np.array_equal(np.isin(new, (2, 3, 4)), np.isin(self.m, (2, 3, 4))))

    def test_rotation_moves_lesion(self):
        new, _, _ = md.transform_volume(self.m, md.Params(rotation_deg=50))
        self.assertAlmostEqual(mean_angle(new, md.INF), (90 + 50) % 360, delta=4)
        self.assertAlmostEqual(int((new == md.INF).sum()) / int((self.m == md.INF).sum()), 1.0, delta=0.08)

    def test_angular_scaling_changes_extent(self):
        new, _, _ = md.transform_volume(self.m, md.Params(angular_scale=1.3))
        ratio = int((new == md.INF).sum()) / int((self.m == md.INF).sum())
        self.assertGreater(ratio, 1.15)

    def test_slice_shift_moves_slices(self):
        new, _, lost = md.transform_volume(self.m, md.Params(slice_shift=1))
        has = [bool((new[:, :, k] == md.INF).any()) for k in range(N)]
        self.assertEqual(has, [False, False, False, True, True, True, False])
        self.assertEqual(lost, 0.0)

    def test_shift_off_the_heart_is_rejected(self):
        m = ring(lesion_deg=90, z=range(5, 7))
        new, leave, lost = md.transform_volume(m, md.Params(slice_shift=1))
        why = md.evaluate(md.lesion_metrics(m), md.lesion_metrics(new), leave, lost, REF)
        self.assertTrue(any("pushed off" in w for w in why), why)

    def test_depth_blowup_is_rejected(self):
        p = md.Params(depth_scale=2.5)
        new, leave, lost = md.transform_volume(self.m, p)
        why = md.evaluate(md.lesion_metrics(self.m), md.lesion_metrics(new), leave, lost, REF)
        self.assertTrue(any("leaves myocardium" in w for w in why), why)

    def test_area_limit(self):
        p = md.Params(angular_scale=0.4)
        new, leave, lost = md.transform_volume(self.m, p)
        why = md.evaluate(md.lesion_metrics(self.m), md.lesion_metrics(new), leave, lost, REF)
        self.assertTrue(any("area" in w for w in why), why)

    def test_mvo_leaving_infarct_is_rejected(self):
        bad = self.m.copy()
        bad[:, :, 3][(np.arange(SIZE)[:, None] > 0) & (bad[:, :, 3] == md.MYO) & (np.arange(SIZE)[None, :] < 40) & (np.arange(SIZE)[:, None] < 40)] = md.MVO
        why = md.evaluate(md.lesion_metrics(self.m), md.lesion_metrics(bad), 0.0, 0.0, REF)
        self.assertTrue(any("MVO not enclosed" in w for w in why), why)

    def test_insert_into_normal_heart(self):
        lib = md.LibraryLesion("lib", self.m)
        normal = ring(lesion_deg=None)
        new, leave, lost = md.insert_lesion(normal, lib, md.Params())
        self.assertTrue(np.array_equal(new, self.m))          # same geometry, so lesion lands exactly
        smaller = ring(lesion_deg=None)[:, :, :5]              # fewer slices: lesion must still map
        new2, _, lost2 = md.insert_lesion(smaller, lib, md.Params())
        self.assertGreater(int((new2 == md.INF).sum()), 0)

    def test_seeded_and_deterministic(self):
        a, _ = md.augment(self.m, 3, np.random.RandomState(1), REF)
        b, _ = md.augment(self.m, 3, np.random.RandomState(1), REF)
        self.assertEqual([x["params"] for x in a], [x["params"] for x in b])


if __name__ == "__main__":
    unittest.main(verbosity=2)
