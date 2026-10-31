"""
colors.py

The ONE place overlay colors are defined. Used by:
  - data_pipeline/overlay_renderer.py and overlay_renderer_synthetic.py
    (PNGs that go to the report model and to the physician tool's report view)
  - physician_tool/physician_tool.py and its index.html (mask overlay,
    callout markers)
  - report_generation/prompts_region_grounded.py (the region legend the
    report model is told, built from COLOR_NAMES so prompt and image cannot
    drift apart)

Change a color here and every consumer follows. Do not redefine colors
anywhere else.
"""

import numpy as np

# mask label id -> region name (EMIDEC label mapping)
REGION_LABEL_IDS = {"lv_cavity": 1, "myocardium": 2, "infarct": 3, "mvo": 4}
LABEL_TO_REGION = {v: k for k, v in REGION_LABEL_IDS.items()}

# Drawing order: later regions are painted on top of earlier ones.
REGION_ORDER = ["lv_cavity", "myocardium", "infarct", "mvo"]

COLORS_RGB = {
    "lv_cavity":  (74, 144, 217),   # blue
    "myocardium": (60, 190, 110),   # green
    "infarct":    (240, 60, 60),    # red
    "mvo":        (255, 215, 40),   # yellow
}

# Opacity of each region's fill (0-1). Infarct and MVO are stronger so they
# stay the focus; cavity and myocardium are lighter context.
ALPHAS = {
    "lv_cavity":  0.35,
    "myocardium": 0.30,
    "infarct":    0.55,
    "mvo":        0.70,
}

# Plain-language color names, used in the report prompts' region legend.
COLOR_NAMES = {
    "lv_cavity": "blue",
    "myocardium": "green",
    "infarct": "red",
    "mvo": "yellow",
}


def hex_color(region: str) -> str:
    r, g, b = COLORS_RGB[region]
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_colors() -> dict:
    return {region: hex_color(region) for region in REGION_ORDER}


def apply_overlay(gray_2d: np.ndarray, mask_2d: np.ndarray, regions=None) -> np.ndarray:
    """Alpha-blends the mask onto a grayscale slice.

    gray_2d: 2D array scaled 0-1 (float) or 0-255 (uint8).
    mask_2d: 2D integer label array, same shape.
    Returns an (H, W, 3) uint8 RGB array. Regions are painted in REGION_ORDER,
    so MVO sits on top of infarct, infarct on top of myocardium, and so on.
    """
    gray = np.asarray(gray_2d, dtype=float)
    if gray.max() > 1.0:
        gray = gray / 255.0
    out = np.stack([gray, gray, gray], axis=-1) * 255.0
    for region in (regions or REGION_ORDER):
        sel = np.asarray(mask_2d) == REGION_LABEL_IDS[region]
        if not sel.any():
            continue
        color = np.array(COLORS_RGB[region], dtype=float)
        a = ALPHAS[region]
        out[sel] = (1 - a) * out[sel] + a * color
    return np.clip(out, 0, 255).astype(np.uint8)


def rgba_overlay(mask_2d: np.ndarray) -> np.ndarray:
    """(H, W, 4) float RGBA layer for matplotlib imshow, transparent where
    there is no labeled region. Same colors and opacities as apply_overlay."""
    mask_2d = np.asarray(mask_2d)
    layer = np.zeros((*mask_2d.shape, 4), dtype=float)
    for region in REGION_ORDER:
        sel = mask_2d == REGION_LABEL_IDS[region]
        r, g, b = [c / 255.0 for c in COLORS_RGB[region]]
        layer[sel] = (r, g, b, ALPHAS[region])
    return layer
