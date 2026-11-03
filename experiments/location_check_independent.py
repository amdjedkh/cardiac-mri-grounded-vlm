"""
location_check_independent.py

A second wall-location check, written separately from the existing one: it does
NOT import assign_aha17_segment, verify_slice_claims or
check_location_claims_batch, and does not use the arc-midpoint segment.

Method (pixel-wise sector share), on the slice shown to the model:
  - cavity centroid = mean of label-1 pixels
  - each infarct pixel (label 3) gets a clockwise angle from image-top (anterior),
    and a 60-degree sector: anterior [330,30), anteroseptal [30,90),
    inferoseptal [90,150), inferior [150,210), inferolateral [210,270),
    anterolateral [270,330)  (image top = anterior, image right = septal: the
    same assumed orientation as the rest of the project, author-inspected,
    not cardiologist-confirmed)
  - the report's claim = wall words in the sentence carrying [region: infarct]:
      anterior   -> anterior, anteroseptal, anterolateral sectors
      inferior   -> inferior, inferoseptal, inferolateral
      septal     -> anteroseptal, inferoseptal
      lateral    -> anterolateral, inferolateral
      compound words (anteroseptal, ...) -> exactly that sector
    several words -> union of their sectors
  - share = fraction of the slice's infarct pixels inside the claimed sectors
  - verdict: right if share >= SHARE_RIGHT (0.5, declared here before scoring)
Ring level (basal / mid / apical) is not checked. Reports with no wall word, or a
slice without infarct, are "not checkable".
"""

import re

import numpy as np

SHARE_RIGHT = 0.5
SECTORS = ["anterior", "anteroseptal", "inferoseptal", "inferior", "inferolateral", "anterolateral"]
WORD_SECTORS = {
    "anteroseptal": {"anteroseptal"}, "anterior-septal": {"anteroseptal"}, "antero-septal": {"anteroseptal"},
    "inferoseptal": {"inferoseptal"}, "inferior-septal": {"inferoseptal"}, "infero-septal": {"inferoseptal"},
    "inferolateral": {"inferolateral"}, "inferior-lateral": {"inferolateral"}, "infero-lateral": {"inferolateral"},
    "anterolateral": {"anterolateral"}, "anterior-lateral": {"anterolateral"}, "antero-lateral": {"anterolateral"},
    "anterior": {"anterior", "anteroseptal", "anterolateral"},
    "inferior": {"inferior", "inferoseptal", "inferolateral"},
    "septal": {"anteroseptal", "inferoseptal"},
    "lateral": {"anterolateral", "inferolateral"},
}


def sector_of(mask2d):
    """(sector name per infarct pixel) or None when there is no cavity or no infarct."""
    cy, cx = np.nonzero(mask2d == 1)
    ys, xs = np.nonzero(mask2d == 3)
    if len(cy) == 0 or len(ys) == 0:
        return None
    ang = np.degrees(np.arctan2(xs - cx.mean(), -(ys - cy.mean()))) % 360   # 0 = up, clockwise
    idx = (((ang + 30) % 360) // 60).astype(int)
    return np.array(SECTORS)[idx]


def claim_sentence(text):
    clean = text.replace("**", "").replace("*", "")
    pos = clean.find("[region: infarct]")
    if pos < 0:
        return ""
    return re.split(r"(?<=[.!?])\s+", clean[:pos].strip())[-1]


def claimed_sectors(text):
    s = claim_sentence(text).lower()
    found = set()
    for word in sorted(WORD_SECTORS, key=len, reverse=True):
        if word in s:
            found |= WORD_SECTORS[word]
            s = s.replace(word, " ")
    return found


def check(text, mask2d):
    """{'checkable': bool, 'share': float|None, 'right': bool|None}"""
    claimed = claimed_sectors(text)
    secs = sector_of(mask2d)
    if not claimed or secs is None:
        return {"checkable": False, "share": None, "right": None}
    share = float(np.isin(secs, list(claimed)).mean())
    return {"checkable": True, "share": share, "right": share >= SHARE_RIGHT}
