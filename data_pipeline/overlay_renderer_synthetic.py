"""
overlay_renderer_synthetic.py

Renders plain + overlay PNGs for synthetic LeFusion cases (Image/Mask layout),
mirroring overlay_renderer.py's real-case output format so both feed the same
prompts_region_grounded.py legend and physician_tool.py Module 2 pairing
convention. Confirmed differences from the real-case renderer:
  - Reads SYNTHETIC_CASES_DIR/Image/<id>.nii.gz and .../Mask/<id>.nii.gz.
  - Applies the exact axis-fix from physician_tool.py's load_case_arrays
    (mask ships [D,H,W], image [H,W,D] -- confirmed: Case_P001 mask
    (10,72,72) -> perm [1,2,0] -> (72,72,10)).
  - Normalizes per-slice by percentile regardless of absolute intensity
    range (synthetic images are rescaled to ~[-1,1], real EMIDEC is in the
    thousands).
"""
import os, sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYNTHETIC_CASES_DIR = os.environ.get("SYNTHETIC_CASES_DIR", "synthetic_cases")
OUT_DIR = "overlays"
LABEL_INFARCT = 3
LABEL_MVO = 4


def load_synthetic_case(case_id: str):
    img_path = os.path.join(SYNTHETIC_CASES_DIR, "Image", f"{case_id}.nii.gz")
    mask_path = os.path.join(SYNTHETIC_CASES_DIR, "Mask", f"{case_id}.nii.gz")
    img_data = np.asarray(nib.load(img_path).get_fdata())
    mask_data = np.asarray(nib.load(mask_path).get_fdata()).round().astype(int)
    if img_data.shape != mask_data.shape and sorted(img_data.shape) == sorted(mask_data.shape):
        remaining = list(range(mask_data.ndim))
        perm = []
        for target_size in img_data.shape:
            for ax in remaining:
                if mask_data.shape[ax] == target_size:
                    perm.append(ax); remaining.remove(ax); break
        mask_data = np.transpose(mask_data, perm)
    return img_data, mask_data


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice_2d, [1, 99])
    if hi <= lo:
        return np.zeros_like(slice_2d, dtype=float)
    return np.clip((slice_2d.astype(float) - lo) / (hi - lo), 0, 1)


def slices_to_render(mask_data: np.ndarray) -> list:
    n_slices = mask_data.shape[2]
    pathology = sorted(set(
        s for s in range(n_slices)
        if np.any(mask_data[:, :, s] == LABEL_INFARCT) or np.any(mask_data[:, :, s] == LABEL_MVO)
    ))
    mid = n_slices // 2
    if mid not in pathology:
        pathology.append(mid)
    return sorted(set(pathology))


def render_case(case_id: str):
    img_data, mask_data = load_synthetic_case(case_id)
    out_dir = os.path.join(OUT_DIR, case_id)
    os.makedirs(out_dir, exist_ok=True)
    slices = slices_to_render(mask_data)
    for s in slices:
        plain = normalize_slice(img_data[:, :, s])
        mask_2d = mask_data[:, :, s]
        overlay_rgb = np.stack([plain, plain, plain], axis=-1)
        overlay_rgb[mask_2d == LABEL_INFARCT] = [1.0, 0.0, 0.0]
        overlay_rgb[mask_2d == LABEL_MVO] = [1.0, 1.0, 0.0]
        plt.imsave(os.path.join(out_dir, f"slice_{s:02d}_plain.png"), plain, cmap="gray", vmin=0, vmax=1)
        plt.imsave(os.path.join(out_dir, f"slice_{s:02d}_overlay.png"), overlay_rgb)
    print(f"{case_id}: rendered {len(slices)} slice(s) -> {out_dir}/")


def main():
    case_ids = sys.argv[1:]
    if not case_ids:
        img_dir = os.path.join(SYNTHETIC_CASES_DIR, "Image")
        case_ids = sorted(f[:-len(".nii.gz")] for f in os.listdir(img_dir) if f.endswith(".nii.gz"))
    for case_id in case_ids:
        try:
            render_case(case_id)
        except Exception as e:
            print(f"{case_id}: [ERROR] {e}")


if __name__ == "__main__":
    main()
