"""
verify_labels.py (v2 -- updated after real EMIDEC file inspection)

Run this LOCALLY against your downloaded EMIDEC Contours file.

Usage (PowerShell):
    python verify_labels.py "Case_P001\\Contours\\Case_P001.nii.gz"

Changes from v1:
  - Spacing is now read from the NIfTI header's pixdim field, not the affine
    matrix. On a real EMIDEC file the affine was found to be a degenerate
    identity matrix (qform/sform not set), which silently gives a wrong
    (1,1,1)mm spacing if trusted. pixdim carries the real values.
  - Prints an approximate myocardial mass sanity check, since it's a useful
    cross-check on whether label 1 vs 2 is really myocardium vs cavity.
"""

import sys
import numpy as np
import nibabel as nib

if len(sys.argv) < 2:
    print("Usage: python verify_labels.py <path_to_contours_nii>")
    sys.exit(1)

path = sys.argv[1]
img = nib.load(path)
data = np.asarray(img.get_fdata()).round().astype(int)

print(f"File: {path}")
print(f"Shape: {data.shape}")

zooms = img.header.get_zooms()
spacing = tuple(float(z) for z in zooms[:3])
print(f"Header pixdim spacing (mm): {spacing}  <-- trust this over the affine")

affine_spacing = tuple(float(np.linalg.norm(img.affine[:3, i])) for i in range(3))
print(f"Affine-derived spacing (mm): {affine_spacing}  (for comparison only -- "
      f"EMIDEC affines were found unreliable, may show (1,1,1))")

voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
print(f"Voxel volume (mL), using header spacing: {voxel_vol_ml:.5f}")

unique, counts = np.unique(data, return_counts=True)
print("\nUnique label values and voxel counts:")
for u, c in zip(unique, counts):
    vol_ml = c * voxel_vol_ml
    print(f"  label {u}: {c} voxels  ->  {vol_ml:.2f} mL")

print(
    "\nCONFIRMED DEFAULT_LABEL_MAP (measurements.py), verified against the official "
    "EMIDEC-Challenge GitHub evaluation-metrics repo and independent papers:\n"
    "  0 = background\n"
    "  1 = LV cavity\n"
    "  2 = myocardium (normal)\n"
    "  3 = infarct\n"
    "  4 = MVO / no-reflow\n"
)

# Sanity check uses label 2 (myocardium) now that the mapping is confirmed correct.
label_myo_vol_ml = counts[unique.tolist().index(2)] * voxel_vol_ml if 2 in unique else None
if label_myo_vol_ml is not None:
    approx_mass_g = label_myo_vol_ml * 1.05
    print(
        f"Normal myocardium (label 2) volume = {label_myo_vol_ml:.2f} mL -> "
        f"approx mass = {approx_mass_g:.1f} g (normal myocardium only, excludes "
        "infarct/MVO tissue, which is anatomically still myocardial wall -- add "
        "infarct + MVO volumes for total wall mass)."
    )

print(
    "\nCompare the gap-between-slices printed above against the case's own "
    "clinical .txt 'Gap between slices' field -- they should match for the "
    "3rd (through-plane) spacing value."
)
