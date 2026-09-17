"""
plot_consistency_comparison.py

Makes an actual visual for the slice-consistency check, not just numbers:
plots cavity size across every slice for several real cases and several
synthetic cases, on the same chart. If synthetic tapers as smoothly as
real, the lines should look similarly shaped -- gradual curves, not
jagged zig-zags.

Usage (run from your EMIDEC folder):
    python plot_consistency_comparison.py --real Case_P019 Case_P004 --synthetic Case_P001.nii.gz Case_P002.nii.gz
"""

import argparse
import os
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import modal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verification"))
from slice_consistency_metrics import compute_slice_profile

DEFAULT_REAL_CASES = ["Case_P019", "Case_P004", "Case_P055", "Case_P060"]
DEFAULT_SYNTHETIC_CASES = ["Case_P001.nii.gz", "Case_P002.nii.gz", "Case_P003.nii.gz", "Case_P004.nii.gz"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", nargs="+", default=DEFAULT_REAL_CASES,
                         help="Real case IDs, e.g. Case_P019 Case_P004")
    parser.add_argument("--synthetic", nargs="+", default=DEFAULT_SYNTHETIC_CASES,
                         help="Synthetic case filenames, e.g. Case_P001.nii.gz Case_P002.nii.gz")
    return parser.parse_args()


def load_real_profile(case_id: str, root: str = "."):
    path = os.path.join(root, case_id, "Contours", f"{case_id}.nii.gz")
    mask = np.asarray(nib.load(path).get_fdata()).round().astype(int)
    profile = compute_slice_profile(mask)
    return [p["cavity_area"] for p in profile]


def main():
    args = parse_args()
    fn = modal.Function.from_name("lefusion-emidec-poc", "check_synthetic_consistency")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax_real = axes[0]
    for case_id in args.real:
        try:
            areas = load_real_profile(case_id)
            ax_real.plot(range(len(areas)), areas, marker="o", label=case_id)
        except Exception as e:
            print(f"Skipping real {case_id}: {e}")
    ax_real.set_title("REAL cases\ncavity size across slices", fontweight="bold")
    ax_real.set_xlabel("slice number")
    ax_real.set_ylabel("cavity area (pixels)")
    ax_real.legend(fontsize=8)
    ax_real.grid(alpha=0.3)

    ax_synth = axes[1]
    for case_id in args.synthetic:
        result = fn.remote(case_id=case_id)
        if "error" in result:
            print(f"Skipping synthetic {case_id}: {result['error']}")
            continue
        areas = result["cavity_areas_by_slice"]
        ax_synth.plot(range(len(areas)), areas, marker="o", label=case_id)
    ax_synth.set_title("SYNTHETIC cases (LeFusion)\ncavity size across slices", fontweight="bold")
    ax_synth.set_xlabel("slice number")
    ax_synth.set_ylabel("cavity area (pixels)")
    ax_synth.legend(fontsize=8)
    ax_synth.grid(alpha=0.3)

    fig.suptitle("Does the heart taper smoothly slice-to-slice, real vs. synthetic?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = "consistency_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    print("Smooth, gradually-curving lines = consistent tapering (good).")
    print("Jagged, zig-zagging lines = inconsistent/erratic slices (bad).")


if __name__ == "__main__":
    main()
