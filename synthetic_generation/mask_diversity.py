"""
mask_diversity.py

Makes new lesion layouts on the SAME heart, and can insert real lesions into
normal hearts. Works on EMIDEC-style label maps shaped [H, W, D]:
    1 LV cavity, 2 myocardium, 3 infarct, 4 MVO.
(Synthetic LeFusion masks are stored [D, H, W]; transpose them first.)

How it works
  Every wall pixel (labels 2, 3, 4) on a slice gets two coordinates:
    angle  - direction from that slice's cavity centroid (radians, same
             convention as verification/verify_slice_claims.py)
    depth  - 0 at the cavity edge, 1 at the outer wall edge, from distance
             transforms: depth = d_cavity / (d_cavity + d_outside)
  The lesion is moved in (angle, depth) space with
    rotation          angle' = angle + rotation
    angular scaling   about the lesion's mean angle
    depth scaling     about the lesion's mean depth
    slice shift       lesion slice k lands on slice k + shift
  and is mapped back onto the original wall pixels by nearest neighbour in
  (angle, depth) space. Cavity and wall pixels never move, so the heart is
  unchanged; only the split of the wall into myocardium / infarct / MVO is.

A result is REJECTED if
  - more than MASK_LEAVE_TOLERANCE of the lesion would leave the wall
    (depth outside 0..1, angular extent wrapping onto itself) or if lesion
    slices are shifted off the heart,
  - MVO is no longer enclosed by infarct (share of the MVO outline that
    touches infarct falls below the real-data minimum),
  - infarct area, MVO area or total lesion area changes by outside 0.5-1.5x,
  - infarct contiguity falls outside the range seen in real EMIDEC cases.
The real ranges are computed from the real cases (`reference` command) and
cached in mask_diversity_reference.json.

Commands (EMIDEC root = folder holding Case_* directories):
    python synthetic_generation/mask_diversity.py reference --emidec ROOT
    python synthetic_generation/mask_diversity.py augment --emidec ROOT \\
        --case Case_P019 --n 4 --out mask_diversity_out
    python synthetic_generation/mask_diversity.py insert --emidec ROOT \\
        --library Case_P019 Case_P004 --targets Case_N006 --n 2 --out mask_diversity_out

Test-set patients (splits/split_v1.json) are refused as sources and targets
unless --allow-test is given. All settings and the seed come from config.py.
"""

import os
import sys
import json
import glob
import argparse
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "verification"))
sys.path.insert(0, os.path.join(REPO_ROOT, "splits"))
import config
import colors
from verify_slice_claims import contiguity_score

CAV, MYO, INF, MVO = 1, 2, 3, 4
REFERENCE_PATH = os.path.join(HERE, "mask_diversity_reference.json")


# ----------------------------------------------------------------- geometry
@dataclass
class Params:
    rotation_deg: float = 0.0
    angular_scale: float = 1.0
    depth_scale: float = 1.0
    slice_shift: int = 0


class SliceGeom:
    """Wall pixels of one slice with their (angle, depth) coordinates."""

    def __init__(self, mask2d):
        cav = mask2d == CAV
        wall = np.isin(mask2d, (MYO, INF, MVO))
        self.valid = bool(cav.any() and wall.any())
        if not self.valid:
            return
        cy, cx = [c.mean() for c in np.nonzero(cav)]
        self.center = (cy, cx)
        d_cav = ndimage.distance_transform_edt(~cav)
        d_out = ndimage.distance_transform_edt(cav | wall)
        self.ys, self.xs = np.nonzero(wall)
        a, b = d_cav[self.ys, self.xs], d_out[self.ys, self.xs]
        self.depth = a / (a + b)
        self.theta = np.arctan2(self.ys - cy, self.xs - cx)
        self.label = mask2d[self.ys, self.xs]
        self.R = float(np.hypot(self.ys - cy, self.xs - cx).mean())  # px per radian
        self.T = float((a + b).mean())                                # px per depth unit
        self._tree = None

    def feats(self, theta, depth):
        return np.stack([self.R * np.cos(theta), self.R * np.sin(theta), self.T * depth], axis=1)

    @property
    def tree(self):
        if self._tree is None:
            self._tree = cKDTree(self.feats(self.theta, self.depth))
        return self._tree


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def lesion_center(geoms):
    """Mean angle (circular) and mean depth over all lesion pixels."""
    th, dp = [], []
    for g in geoms:
        if g is not None and g.valid:
            sel = np.isin(g.label, (INF, MVO))
            th.append(g.theta[sel]); dp.append(g.depth[sel])
    if not th or sum(len(t) for t in th) == 0:
        return None
    th, dp = np.concatenate(th), np.concatenate(dp)
    return float(np.arctan2(np.sin(th).mean(), np.cos(th).mean())), float(dp.mean())


def resample_labels(src, tgt, p, theta_c, depth_c):
    """Labels (2/3/4) for every wall pixel of tgt, taking the lesion from src
    moved by p. Inverse mapping: each target pixel asks 'which source pixel
    moved here?'."""
    delta = wrap(tgt.theta - theta_c - np.deg2rad(p.rotation_deg))
    src_dth = delta / p.angular_scale
    src_theta = theta_c + src_dth
    src_depth = depth_c + (tgt.depth - depth_c) / p.depth_scale
    dist, idx = src.tree.query(src.feats(src_theta, src_depth))
    lab = src.label[idx]
    hit = (np.abs(src_dth) <= np.pi) & (dist <= config.MASK_NEIGHBOUR_TOL_PX) & np.isin(lab, (INF, MVO))
    return np.where(hit, lab, MYO)


def leave_fraction(geoms, p, theta_c, depth_c):
    """Forward check: share of source lesion pixels that would land outside
    the wall (depth outside 0..1) or whose angular extent wraps onto itself."""
    total = out = 0
    for g in geoms:
        if g is None or not g.valid:
            continue
        sel = np.isin(g.label, (INF, MVO))
        if not sel.any():
            continue
        d2 = depth_c + p.depth_scale * (g.depth[sel] - depth_c)
        wraps = p.angular_scale * np.abs(wrap(g.theta[sel] - theta_c)) > np.pi
        out += int(((d2 < 0) | (d2 > 1) | wraps).sum())
        total += int(sel.sum())
    return out / total if total else 0.0


# ------------------------------------------------------------------ metrics
def lesion_metrics(mask):
    inf, mvo = int((mask == INF).sum()), int((mask == MVO).sum())
    wall = int(np.isin(mask, (MYO, INF, MVO)).sum())
    conts, touch_b, touch_n = [], 0, 0
    for k in range(mask.shape[2]):
        m2 = mask[:, :, k]
        cav = m2 == CAV
        if not cav.any():
            continue
        center = tuple(c.mean() for c in np.nonzero(cav))
        if (m2 == INF).any():
            c = contiguity_score(m2, INF, center)["contiguity"]
            if c is not None:
                conts.append(c)
        if (m2 == MVO).any():
            ring = ndimage.binary_dilation(m2 == MVO) & ~(m2 == MVO) & np.isin(m2, (MYO, INF, MVO))
            touch_b += int((ring & (m2 == INF)).sum()); touch_n += int(ring.sum())
    return {
        "infarct_px": inf, "mvo_px": mvo, "lesion_px": inf + mvo, "wall_px": wall,
        "infarct_frac": inf / wall if wall else 0.0,
        "mvo_frac": mvo / wall if wall else 0.0,
        "lesion_frac": (inf + mvo) / wall if wall else 0.0,
        "contiguity": float(np.mean(conts)) if conts else None,
        "mvo_touch": touch_b / touch_n if touch_n else None,
    }


def compute_reference(emidec_root, save=True):
    cont, touch = {}, {}
    for d in sorted(glob.glob(os.path.join(emidec_root, "Case_P*"))):
        cid = os.path.basename(d)
        mask = np.asarray(nib.load(os.path.join(d, "Contours", f"{cid}.nii.gz")).dataobj).round().astype(int)
        m = lesion_metrics(mask)
        if m["contiguity"] is not None:
            cont[cid] = m["contiguity"]
        if m["mvo_touch"] is not None:
            touch[cid] = m["mvo_touch"]
    ref = {
        "contiguity": {"min": min(cont.values()), "max": max(cont.values()), "n": len(cont), "per_case": cont},
        "mvo_touch": {"min": min(touch.values()), "n": len(touch), "per_case": touch},
    }
    if save:
        with open(REFERENCE_PATH, "w") as f:
            json.dump(ref, f, indent=1)
    return ref


def load_reference(emidec_root=None):
    if os.path.exists(REFERENCE_PATH):
        return json.load(open(REFERENCE_PATH))
    if emidec_root is None:
        sys.exit("No mask_diversity_reference.json yet: run `reference --emidec ROOT` first.")
    return compute_reference(emidec_root)


def evaluate(src, new, leave, lost, ref, use_fraction=False):
    """Returns the list of rejection reasons (empty = accepted)."""
    why = []
    if leave > config.MASK_LEAVE_TOLERANCE:
        why.append(f"lesion leaves myocardium ({leave:.0%} of pixels)")
    if lost > 0:
        why.append(f"lesion pushed off the heart ({lost:.0%} of lesion pixels on lost slices)")
    lo, hi = config.MASK_AREA_RATIO_RANGE
    suffix = "frac" if use_fraction else "px"
    for name in ("infarct", "mvo", "lesion"):
        s, n = src[f"{name}_{suffix}"], new[f"{name}_{suffix}"]
        if s > 0:
            r = n / s
            if not lo <= r <= hi:
                why.append(f"{name} area x{r:.2f} outside {lo}-{hi}")
    c = new["contiguity"]
    if c is None:
        why.append("contiguity undefined")
    elif not ref["contiguity"]["min"] <= c <= ref["contiguity"]["max"]:
        why.append(f"contiguity {c:.3f} outside real range "
                   f"{ref['contiguity']['min']:.3f}-{ref['contiguity']['max']:.3f}")
    if new["mvo_px"] > 0:
        t = new["mvo_touch"]
        if t is None or t < ref["mvo_touch"]["min"]:
            why.append(f"MVO not enclosed by infarct (touch {t if t is None else round(t, 2)} "
                       f"< real min {ref['mvo_touch']['min']:.2f})")
    return why


# ------------------------------------------------------------------ augment
def transform_volume(mask, p):
    """Moves the lesion by p on the same heart. Returns (new_mask, leave, lost)
    or None if the volume has no lesion."""
    n = mask.shape[2]
    geoms = [SliceGeom(mask[:, :, k]) for k in range(n)]
    center = lesion_center(geoms)
    if center is None:
        return None
    tc, dc = center
    leave = leave_fraction(geoms, p, tc, dc)
    out = mask.copy()
    for k, g in enumerate(geoms):
        if g.valid:
            out[g.ys, g.xs, k] = np.where(np.isin(g.label, (INF, MVO)), MYO, g.label)
    lost_px = total_px = 0
    for sk, g in enumerate(geoms):
        if not g.valid:
            continue
        n_les = int(np.isin(g.label, (INF, MVO)).sum())
        total_px += n_les
        tk = sk + p.slice_shift
        if n_les and not (0 <= tk < n and geoms[tk].valid):
            lost_px += n_les
    for tk, tg in enumerate(geoms):
        sk = tk - p.slice_shift
        if tg.valid and 0 <= sk < n and geoms[sk].valid:
            out[tg.ys, tg.xs, tk] = resample_labels(geoms[sk], tg, p, tc, dc)
    return out, leave, (lost_px / total_px if total_px else 0.0)


def sample_params(rng):
    return Params(
        rotation_deg=float(rng.uniform(-config.MASK_ROTATION_MAX_DEG, config.MASK_ROTATION_MAX_DEG)),
        angular_scale=float(np.exp(rng.uniform(*np.log(config.MASK_ANGULAR_SCALE_RANGE)))),
        depth_scale=float(rng.uniform(*config.MASK_DEPTH_SCALE_RANGE)),
        slice_shift=int(rng.randint(-config.MASK_SLICE_SHIFT_MAX, config.MASK_SLICE_SHIFT_MAX + 1)),
    )


def augment(mask, n, rng, ref, max_attempts=None):
    """Returns (accepted, rejected). Each item: dict(params, mask, metrics, why)."""
    src = lesion_metrics(mask)
    accepted, rejected = [], []
    for _ in range(max_attempts or n * 25):
        if len(accepted) >= n:
            break
        p = sample_params(rng)
        res = transform_volume(mask, p)
        if res is None:
            break
        new, leave, lost = res
        m = lesion_metrics(new)
        why = evaluate(src, m, leave, lost, ref)
        (rejected if why else accepted).append({"params": asdict(p), "mask": new, "metrics": m, "why": why})
    return accepted, rejected


# ------------------------------------------------------------------- insert
class LibraryLesion:
    def __init__(self, case_id, mask):
        self.case_id = case_id
        self.n = mask.shape[2]
        self.geoms = [SliceGeom(mask[:, :, k]) for k in range(self.n)]
        self.zfrac = np.array([k / max(self.n - 1, 1) for k in range(self.n)])
        self.center = lesion_center(self.geoms)
        self.metrics = lesion_metrics(mask)
        self.lesion_px = [int(np.isin(g.label, (INF, MVO)).sum()) if g.valid else 0 for g in self.geoms]


def insert_lesion(normal_mask, lib, p):
    """Puts the library lesion into a normal heart. Slices are matched by
    relative position along the stack (0 = first slice, 1 = last)."""
    nt = normal_mask.shape[2]
    tgeoms = [SliceGeom(normal_mask[:, :, k]) for k in range(nt)]
    tc, dc = lib.center
    out = normal_mask.copy()
    valid_src = [j for j in range(lib.n) if lib.geoms[j].valid]
    tol = max(0.5 / max(lib.n - 1, 1), 0.5 / max(nt - 1, 1)) + 1e-9
    used = set()
    for tk, tg in enumerate(tgeoms):
        if not tg.valid:
            continue
        zf = (tk - p.slice_shift) / max(nt - 1, 1)
        j = min(valid_src, key=lambda s: abs(lib.zfrac[s] - zf))
        if abs(lib.zfrac[j] - zf) > tol:
            continue
        used.add(j)
        out[tg.ys, tg.xs, tk] = resample_labels(lib.geoms[j], tg, p, tc, dc)
    lost = sum(px for j, px in enumerate(lib.lesion_px) if j not in used)
    lost_frac = lost / max(sum(lib.lesion_px), 1)
    return out, leave_fraction(lib.geoms, p, tc, dc), lost_frac


def insert_many(normal_mask, lib, n, rng, ref, max_attempts=None):
    accepted, rejected = [], []
    for i in range(max_attempts or n * 25):
        if len(accepted) >= n:
            break
        p = Params() if (i == 0 and n == 1) else sample_params(rng)
        new, leave, lost = insert_lesion(normal_mask, lib, p)
        m = lesion_metrics(new)
        why = evaluate(lib.metrics, m, leave, lost, ref, use_fraction=True)
        (rejected if why else accepted).append({"params": asdict(p), "mask": new, "metrics": m, "why": why})
    return accepted, rejected


# ---------------------------------------------------------------------- I/O
def load_case(emidec_root, case_id):
    base = os.path.join(emidec_root, case_id)
    img = nib.load(os.path.join(base, "Images", f"{case_id}.nii.gz"))
    msk = nib.load(os.path.join(base, "Contours", f"{case_id}.nii.gz"))
    return np.asarray(img.dataobj).astype(float), np.asarray(msk.dataobj).round().astype(int), msk


def refuse_test_patients(case_ids, allow_test):
    from split_utils import test_ids
    bad = sorted(set(case_ids) & test_ids())
    if bad and not allow_test:
        sys.exit(f"Refusing test-set patients {bad} (splits/split_v1.json). Use --allow-test to override.")


def save_review_png(path, panels, title, max_rows=4):
    """panels: list of (column title, image, mask); the LAST panel is the result.
    One row per slice of the result that has a lesion (at most max_rows, spread
    evenly). Other panels show the slice at the same relative position along
    their own stack, so hearts with different slice counts still line up.
    Each panel is cropped around its own heart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res_mask = panels[-1][2]
    nt = res_mask.shape[2]
    rows = [k for k in range(nt) if np.isin(res_mask[:, :, k], (INF, MVO)).any()]
    if len(rows) > max_rows:
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, max_rows).round().astype(int)]
    crops = []
    for _, _, mask in panels:
        ys, xs = np.nonzero(np.isin(mask, (CAV, MYO, INF, MVO)).any(axis=2))
        m = 8
        crops.append((max(ys.min() - m, 0), ys.max() + m, max(xs.min() - m, 0), xs.max() + m))
    fig, axes = plt.subplots(len(rows), len(panels), figsize=(3.2 * len(panels), 3.2 * len(rows)), squeeze=False)
    for r, k in enumerate(rows):
        for c, (name, image, mask) in enumerate(panels):
            ki = int(round(k / max(nt - 1, 1) * (mask.shape[2] - 1)))
            sl = image[:, :, ki]
            lo, hi = np.percentile(sl, [1, 99])
            gray = np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)
            y0, y1, x0, x1 = crops[c]
            ax = axes[r][c]
            ax.imshow(colors.apply_overlay(gray, mask[:, :, ki])[y0:y1, x0:x1])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"slice {ki}", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fmt_params(p):
    return (f"rot {p['rotation_deg']:+.0f} deg, ang x{p['angular_scale']:.2f}, "
            f"depth x{p['depth_scale']:.2f}, shift {p['slice_shift']:+d}")


def summarize(rejected):
    c = Counter(w.split(" (")[0].split(" x")[0].split(" 0.")[0] for r in rejected for w in r["why"])
    return dict(c)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["reference", "augment", "insert"])
    ap.add_argument("--emidec", required=True)
    ap.add_argument("--case", help="augment: source case")
    ap.add_argument("--library", nargs="+", help="insert: cases whose lesions form the library")
    ap.add_argument("--targets", nargs="+", help="insert: normal cases to insert into")
    ap.add_argument("--n", type=int, default=3, help="accepted results wanted per case")
    ap.add_argument("--out", default="mask_diversity_out")
    ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args()
    rng = np.random.RandomState(config.MASK_DIVERSITY_SEED)

    if a.command == "reference":
        ref = compute_reference(a.emidec)
        print(f"contiguity over {ref['contiguity']['n']} real cases: "
              f"{ref['contiguity']['min']:.3f} - {ref['contiguity']['max']:.3f}")
        print(f"MVO-touch-infarct over {ref['mvo_touch']['n']} real cases: min {ref['mvo_touch']['min']:.3f}")
        print("saved", REFERENCE_PATH)
        return

    ref = load_reference(a.emidec)
    log = []
    if a.command == "augment":
        refuse_test_patients([a.case], a.allow_test)
        image, mask, nii = load_case(a.emidec, a.case)
        src = lesion_metrics(mask)
        acc, rej = augment(mask, a.n, rng, ref)
        print(f"{a.case}: {len(acc)} accepted, {len(rej)} rejected {summarize(rej)}")
        jobs = [(a.case, image, mask, nii, "", acc, rej, src)]
    else:
        refuse_test_patients(a.library + a.targets, a.allow_test)
        jobs = []
        for lib_id in a.library:
            _, lmask, _ = load_case(a.emidec, lib_id)
            lib = LibraryLesion(lib_id, lmask)
            for tid in a.targets:
                image, nmask, nii = load_case(a.emidec, tid)
                acc, rej = insert_many(nmask, lib, a.n, rng, ref)
                print(f"{lib_id} -> {tid}: {len(acc)} accepted, {len(rej)} rejected {summarize(rej)}")
                jobs.append((tid, image, nmask, nii, lib_id, acc, rej, lib.metrics))

    for cid, image, mask, nii, lib_id, acc, rej, src in jobs:
        tag = f"{lib_id}_into_{cid}" if lib_id else cid
        base = os.path.join(a.out, tag)
        os.makedirs(base, exist_ok=True)
        for i, r in enumerate(acc):
            nib.save(nib.Nifti1Image(r["mask"].astype(np.int16), nii.affine, nii.header), os.path.join(base, f"mask_{i:02d}.nii.gz"))
            panels = [("before: " + cid, image, mask), (f"after (result {i})", image, r["mask"])]
            if lib_id:
                limg, lmask, _ = load_case(a.emidec, lib_id)
                panels = [("lesion source: " + lib_id, limg, lmask)] + panels
            save_review_png(os.path.join(base, f"review_{i:02d}.png"), panels,
                            f"{tag}  {fmt_params(r['params'])}  | contiguity {r['metrics']['contiguity']:.3f}")
            print(f"   [{i}] {fmt_params(r['params'])} | infarct {r['metrics']['infarct_px']} px "
                  f"(src {src['infarct_px']}), mvo {r['metrics']['mvo_px']} (src {src['mvo_px']}), "
                  f"contiguity {r['metrics']['contiguity']:.3f}")
        log.append({"case": tag, "source_metrics": src,
                    "accepted": [{k: v for k, v in r.items() if k != "mask"} for r in acc],
                    "rejected": [{k: v for k, v in r.items() if k != "mask"} for r in rej]})
    with open(os.path.join(a.out, f"log_{a.command}.json"), "w") as f:
        json.dump({"seed": config.MASK_DIVERSITY_SEED, "runs": log}, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
