"""
measurements.py

Extracts quantitative measurements from EMIDEC LGE-MRI segmentation masks.

EMIDEC contour convention (per the EMIDEC challenge label definition):
    0 = background
    1 = LV cavity
    2 = normal myocardium
    3 = myocardial infarction (scar)
    4 = no-reflow / microvascular obstruction (MVO)

If your local copy of the masks uses a different label mapping, pass a custom
`label_map` into `extract_measurements` — do not assume the above without
checking the actual EMIDEC Case_N.nii + Case_N_labels metadata / challenge docs
against the files you downloaded.

Usage:
    from measurements import extract_measurements
    result = extract_measurements(mask_path="Case_P1_labels.nii", patient_id="P1")
"""

from __future__ import annotations
import numpy as np
import nibabel as nib
from dataclasses import dataclass, field, asdict
from typing import Optional


DEFAULT_LABEL_MAP = {
    "background": 0,
    "lv_cavity": 1,
    "myocardium_normal": 2,
    "infarct": 3,
    "mvo": 4,
}
# CONFIRMED via web search against three independent sources, including the official
# EMIDEC-Challenge GitHub evaluation-metrics repo, which states explicitly:
# {"background":0, "cavity":1, ...} with myocardium=2, infarction=3, no-reflow=4
# (github.com/EMIDEC-Challenge/Evaluation-metrics). Corroborated by Lalande et al.-derived
# papers stating "label 0 for background, 1 for LV cavity, 2 for healthy myocardium,
# 3 for myocardial infarction, 4 for no-reflow."
#
# NOTE: an earlier version of this file had this backwards (1=myocardium, 2=cavity),
# reasoned from the Readme.txt's prose ordering ("background, myocardium, cavity, ...").
# That prose order does NOT match the actual label integer order -- always prefer a
# verified source (official repo / paper) over inferring from descriptive text.
# Sanity-checked against real Case_P001 data: with THIS mapping, myocardium (normal)
# volume = 96.9 mL -> total myocardium (normal+infarct+MVO) = 200.36 mL -> approx mass
# ~210g, elevated but plausible for a severe case (this patient's troponin=130,
# FEVG=35%, infarct=33% of myocardium -- consistent with an extensive MI).


@dataclass
class MeasurementResult:
    patient_id: str
    voxel_spacing_mm: tuple
    voxel_volume_ml: float

    lv_cavity_volume_ml: float
    myocardium_total_volume_ml: float          # normal + infarct + MVO (whole wall)
    infarct_volume_ml: float
    mvo_volume_ml: float

    infarct_percentage_of_myocardium: float     # infarct / total myocardium * 100
    mvo_percentage_of_myocardium: float
    mvo_percentage_of_infarct: Optional[float]   # MVO / infarct * 100, if infarct > 0

    infarct_present: bool
    mvo_present: bool

    total_slices: int
    infarct_affected_slices: list = field(default_factory=list)   # 0-indexed slice numbers
    mvo_affected_slices: list = field(default_factory=list)

    label_map_used: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["voxel_spacing_mm"] = list(d["voxel_spacing_mm"])
        return d


def _voxel_volume_ml_from_header(img: "nib.Nifti1Image") -> tuple:
    """
    Returns (spacing_xyz_mm, voxel_volume_ml) using the NIfTI header's pixdim field.

    Preferred over reading the affine matrix: EMIDEC's Contours .nii.gz files were
    found to have a degenerate/identity affine (qform/sform not properly set), which
    silently produces a wrong (1,1,1)mm spacing if you trust the affine. The header's
    pixdim field carries the real acquisition voxel size regardless of affine validity.
    Cross-check against the case's own clinical .txt "Gap between slices" field --
    they should match for the through-plane (3rd) dimension.
    """
    zooms = img.header.get_zooms()
    spacing = tuple(float(z) for z in zooms[:3])
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    voxel_vol_ml = voxel_vol_mm3 / 1000.0
    return spacing, voxel_vol_ml


def _voxel_volume_ml(affine_or_spacing) -> tuple:
    """Returns (spacing_xyz_mm, voxel_volume_ml) from an explicit spacing tuple or affine.
    Kept for the mask_array + explicit voxel_spacing_mm code path; for file-based loads,
    extract_measurements uses _voxel_volume_ml_from_header instead (see note there)."""
    if isinstance(affine_or_spacing, (tuple, list, np.ndarray)) and np.ndim(affine_or_spacing) == 1:
        spacing = tuple(float(s) for s in affine_or_spacing)
    else:
        # affine matrix: spacing = norm of each column vector (first 3x3 block)
        affine = affine_or_spacing
        spacing = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    voxel_vol_ml = voxel_vol_mm3 / 1000.0  # 1 mL = 1000 mm^3
    return spacing, voxel_vol_ml


def _affected_slices(mask: np.ndarray, label: int, slice_axis: int = 2) -> list:
    """Returns list of slice indices (along slice_axis) containing at least one voxel of `label`."""
    present = np.any(mask == label, axis=tuple(a for a in range(mask.ndim) if a != slice_axis))
    return [int(i) for i, p in enumerate(present) if p]


def extract_measurements(
    mask_path: Optional[str] = None,
    mask_array: Optional[np.ndarray] = None,
    voxel_spacing_mm: Optional[tuple] = None,
    patient_id: str = "unknown",
    label_map: dict = None,
    slice_axis: int = 2,
) -> MeasurementResult:
    """
    Compute infarct volume, infarct %, MVO volume, and affected slices from an
    EMIDEC-style segmentation mask.

    Provide EITHER:
      - mask_path: path to a NIfTI (.nii/.nii.gz) segmentation file (spacing read from affine), OR
      - mask_array + voxel_spacing_mm: a raw numpy array and explicit (dx, dy, dz) in mm.

    label_map: override DEFAULT_LABEL_MAP if your files use different integer codes.
    """
    lm = label_map or DEFAULT_LABEL_MAP

    if mask_path is not None:
        img = nib.load(mask_path)
        mask = np.asarray(img.get_fdata()).round().astype(int)
        spacing, voxel_vol_ml = _voxel_volume_ml_from_header(img)
    elif mask_array is not None and voxel_spacing_mm is not None:
        mask = np.asarray(mask_array).round().astype(int)
        spacing, voxel_vol_ml = _voxel_volume_ml(voxel_spacing_mm)
    else:
        raise ValueError("Provide either mask_path, or (mask_array and voxel_spacing_mm).")

    def vol_of(label_name):
        label = lm[label_name]
        return float(np.sum(mask == label)) * voxel_vol_ml

    lv_cavity_ml = vol_of("lv_cavity")
    myo_normal_ml = vol_of("myocardium_normal")
    infarct_ml = vol_of("infarct")
    mvo_ml = vol_of("mvo")

    myo_total_ml = myo_normal_ml + infarct_ml + mvo_ml  # whole myocardial wall

    infarct_pct = (infarct_ml / myo_total_ml * 100.0) if myo_total_ml > 0 else 0.0
    mvo_pct_myo = (mvo_ml / myo_total_ml * 100.0) if myo_total_ml > 0 else 0.0
    mvo_pct_infarct = (mvo_ml / infarct_ml * 100.0) if infarct_ml > 0 else None

    infarct_slices = _affected_slices(mask, lm["infarct"], slice_axis=slice_axis)
    mvo_slices = _affected_slices(mask, lm["mvo"], slice_axis=slice_axis)

    return MeasurementResult(
        patient_id=patient_id,
        voxel_spacing_mm=spacing,
        voxel_volume_ml=voxel_vol_ml,
        lv_cavity_volume_ml=round(lv_cavity_ml, 3),
        myocardium_total_volume_ml=round(myo_total_ml, 3),
        infarct_volume_ml=round(infarct_ml, 3),
        mvo_volume_ml=round(mvo_ml, 3),
        infarct_percentage_of_myocardium=round(infarct_pct, 2),
        mvo_percentage_of_myocardium=round(mvo_pct_myo, 2),
        mvo_percentage_of_infarct=round(mvo_pct_infarct, 2) if mvo_pct_infarct is not None else None,
        infarct_present=infarct_ml > 0,
        mvo_present=mvo_ml > 0,
        total_slices=mask.shape[slice_axis],
        infarct_affected_slices=infarct_slices,
        mvo_affected_slices=mvo_slices,
        label_map_used=lm,
    )


# ----------------------------------------------------------------------------
# Self-test on a synthetic mask (no real EMIDEC data required to verify logic)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Build a synthetic 3D mask: 10x10 slice, 8 slices.
    # Slices 3-5 get a myocardium ring with infarct wedge; slice 4 also gets MVO.
    shape = (10, 10, 8)
    mask = np.zeros(shape, dtype=int)

    yy, xx = np.mgrid[0:10, 0:10]
    center = (5, 5)
    dist = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    cavity_ring = (dist <= 2)
    myo_ring = (dist > 2) & (dist <= 4)

    for s in range(2, 6):
        mask[cavity_ring, s] = DEFAULT_LABEL_MAP["lv_cavity"]
        mask[myo_ring, s] = DEFAULT_LABEL_MAP["myocardium_normal"]

    # infarct wedge (a slice of the myocardium ring) on slices 3-5
    infarct_wedge = myo_ring & (xx >= 5) & (yy >= 5)
    for s in range(3, 6):
        mask[infarct_wedge, s] = DEFAULT_LABEL_MAP["infarct"]

    # MVO core within the infarct, only on slice 4
    mvo_core = infarct_wedge & (dist <= 3.2)
    mask[mvo_core, 4] = DEFAULT_LABEL_MAP["mvo"]

    result = extract_measurements(
        mask_array=mask,
        voxel_spacing_mm=(1.5, 1.5, 8.0),  # typical LGE-MRI: fine in-plane, thick slices
        patient_id="SYNTHETIC_TEST_01",
    )

    import json
    print(json.dumps(result.to_dict(), indent=2))

    # sanity assertions
    assert result.infarct_present is True
    assert result.mvo_present is True
    assert result.infarct_affected_slices == [3, 4, 5]
    assert result.mvo_affected_slices == [4]
    assert result.mvo_percentage_of_infarct is not None
    print("\nSelf-test passed.")
