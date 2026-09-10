"""
overlay_renderer.py

Renders LGE-MRI slices as PNG images, both plain (grayscale) and with the
segmentation mask overlaid (infarct / MVO highlighted), for use as the
image_paths / overlay_paths inputs to the prompts.py scaffolds.

Usage (after downloading, run from your EMIDEC root folder):
    python overlay_renderer.py Case_P001

This renders every slice that contains infarct or MVO (the clinically
relevant ones), plus one normal mid-stack slice for context, to:
    overlays/Case_P001/slice_XX_plain.png
    overlays/Case_P001/slice_XX_overlay.png
"""

import os
import sys
import numpy as np
import nibabel as nib
from PIL import Image

from measurements import DEFAULT_LABEL_MAP

# Colors for overlay (RGBA), semi-transparent so underlying anatomy stays visible
OVERLAY_COLORS = {
    "infarct": (255, 60, 60, 130),   # red
    "mvo": (255, 220, 40, 160),      # yellow, drawn on top of infarct where they overlap
}


def load_and_normalize_image(image_path: str) -> np.ndarray:
    """Loads the LGE image volume and normalizes intensities to 0-255 per-volume
    (not per-slice, so brightness is comparable across slices of the same case)."""
    img = nib.load(image_path)
    data = np.asarray(img.get_fdata()).astype(np.float32)
    lo, hi = np.percentile(data, [1, 99])  # robust to outlier bright pixels
    data = np.clip((data - lo) / max(hi - lo, 1e-6), 0, 1)
    return (data * 255).astype(np.uint8)


def load_mask(mask_path: str) -> np.ndarray:
    img = nib.load(mask_path)
    return np.asarray(img.get_fdata()).round().astype(int)


def render_slice(image_slice: np.ndarray, mask_slice: np.ndarray = None,
                  label_map: dict = None) -> Image.Image:
    """Renders one 2D slice as an RGB PIL image, with optional mask overlay."""
    lm = label_map or DEFAULT_LABEL_MAP
    base = Image.fromarray(image_slice).convert("RGBA")

    if mask_slice is not None:
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        overlay_arr = np.zeros((*mask_slice.shape, 4), dtype=np.uint8)

        infarct_px = mask_slice == lm["infarct"]
        overlay_arr[infarct_px] = OVERLAY_COLORS["infarct"]

        mvo_px = mask_slice == lm["mvo"]
        overlay_arr[mvo_px] = OVERLAY_COLORS["mvo"]  # drawn after, takes priority on overlap

        overlay = Image.fromarray(overlay_arr)
        base = Image.alpha_composite(base, overlay)

    return base.convert("RGB")


def render_patient(case_id: str, root: str = ".", label_map: dict = None,
                    include_context_slice: bool = True):
    lm = label_map or DEFAULT_LABEL_MAP
    image_path = os.path.join(root, case_id, "Images", f"{case_id}.nii.gz")
    mask_path = os.path.join(root, case_id, "Contours", f"{case_id}.nii.gz")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    images = load_and_normalize_image(image_path)
    mask = load_mask(mask_path)

    if images.shape != mask.shape:
        raise ValueError(
            f"Image shape {images.shape} != mask shape {mask.shape} for {case_id} -- "
            "these must be spatially aligned before rendering. Do not proceed without "
            "checking this."
        )

    n_slices = images.shape[2]
    infarct_or_mvo_slices = sorted(set(
        [s for s in range(n_slices) if np.any(mask[:, :, s] == lm["infarct"])] +
        [s for s in range(n_slices) if np.any(mask[:, :, s] == lm["mvo"])]
    ))

    slices_to_render = list(infarct_or_mvo_slices)
    if include_context_slice:
        mid = n_slices // 2
        if mid not in slices_to_render:
            slices_to_render.append(mid)
    slices_to_render = sorted(set(slices_to_render))

    out_dir = os.path.join("overlays", case_id)
    os.makedirs(out_dir, exist_ok=True)

    plain_paths, overlay_paths = [], []
    for s in slices_to_render:
        plain_img = render_slice(images[:, :, s])
        overlay_img = render_slice(images[:, :, s], mask[:, :, s], lm)

        plain_path = os.path.join(out_dir, f"slice_{s:02d}_plain.png")
        overlay_path = os.path.join(out_dir, f"slice_{s:02d}_overlay.png")
        plain_img.save(plain_path)
        overlay_img.save(overlay_path)

        plain_paths.append(plain_path)
        overlay_paths.append(overlay_path)

    return plain_paths, overlay_paths, slices_to_render


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python overlay_renderer.py <Case_ID> [Case_ID2 ...]")
        print("Example: python overlay_renderer.py Case_P001 Case_P019")
        sys.exit(1)

    for case_id in sys.argv[1:]:
        try:
            plain, overlay, slices = render_patient(case_id)
            print(f"{case_id}: rendered slices {slices}")
            print(f"  plain: {plain}")
            print(f"  overlay: {overlay}")
        except Exception as e:
            print(f"ERROR rendering {case_id}: {e}")
