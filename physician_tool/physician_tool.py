"""
physician_tool.py

A simple local web tool for physician validation of cardiac MRI cases.
Module 1 (this file): synthetic/real MRI + mask validation.
Module 2 (report validation) will be added separately.

Run:
    python physician_tool.py

Then open http://localhost:5050 in a browser.

Case folder layout expected:

REAL cases (matches the existing EMIDEC folder structure):
    <REAL_CASES_DIR>/Case_XXXX/Images/Case_XXXX.nii.gz
    <REAL_CASES_DIR>/Case_XXXX/Contours/Case_XXXX.nii.gz

SYNTHETIC cases (matches the LeFusion output folder structure, downloaded
locally first with e.g.:
    modal volume get cardiac-data LeFusion_output/Image ./synthetic_cases/Image
    modal volume get cardiac-data LeFusion_output/Mask ./synthetic_cases/Mask
):
    <SYNTHETIC_CASES_DIR>/Image/Case_XXXX.nii.gz
    <SYNTHETIC_CASES_DIR>/Mask/Case_XXXX.nii.gz

Decisions are saved to annotations/image_validation.jsonl, one JSON line per
decision, so later analysis is straightforward (read line by line, no
special parser needed).
"""

import os
import io
import json
import glob
from datetime import datetime, timezone

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, jsonify, request, send_file, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_CASES_DIR = os.environ.get("REAL_CASES_DIR", os.path.join(BASE_DIR, "real_cases"))
SYNTHETIC_CASES_DIR = os.environ.get("SYNTHETIC_CASES_DIR", os.path.join(BASE_DIR, "synthetic_cases"))
ANNOTATIONS_PATH = os.path.join(BASE_DIR, "annotations", "image_validation.jsonl")
os.makedirs(os.path.dirname(ANNOTATIONS_PATH), exist_ok=True)

LABEL_COLORS = {
    1: (74, 144, 217),   # lv_cavity - blue
    2: (226, 195, 74),   # myocardium - yellow
    3: (226, 74, 74),    # infarct - red
    4: (126, 217, 87),   # mvo - green
}


def find_all_cases():
    """Returns a list of {id, type} for every case found in both folders."""
    cases = []

    if os.path.isdir(REAL_CASES_DIR):
        for case_dir in sorted(glob.glob(os.path.join(REAL_CASES_DIR, "*"))):
            case_id = os.path.basename(case_dir)
            img_path = os.path.join(case_dir, "Images", f"{case_id}.nii.gz")
            mask_path = os.path.join(case_dir, "Contours", f"{case_id}.nii.gz")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                cases.append({"id": case_id, "type": "real"})

    if os.path.isdir(SYNTHETIC_CASES_DIR):
        img_dir = os.path.join(SYNTHETIC_CASES_DIR, "Image")
        mask_dir = os.path.join(SYNTHETIC_CASES_DIR, "Mask")
        if os.path.isdir(img_dir):
            for f in sorted(os.listdir(img_dir)):
                case_id = f.replace(".nii.gz", "")
                if os.path.exists(os.path.join(mask_dir, f)):
                    cases.append({"id": case_id, "type": "synthetic"})

    return cases


def get_case_paths(case_id: str, case_type: str):
    if case_type == "real":
        img_path = os.path.join(REAL_CASES_DIR, case_id, "Images", f"{case_id}.nii.gz")
        mask_path = os.path.join(REAL_CASES_DIR, case_id, "Contours", f"{case_id}.nii.gz")
    else:
        img_path = os.path.join(SYNTHETIC_CASES_DIR, "Image", f"{case_id}.nii.gz")
        mask_path = os.path.join(SYNTHETIC_CASES_DIR, "Mask", f"{case_id}.nii.gz")
    return img_path, mask_path


def load_case_arrays(case_id: str, case_type: str):
    img_path, mask_path = get_case_paths(case_id, case_type)
    img_data = np.asarray(nib.load(img_path).get_fdata())
    mask_data = np.asarray(nib.load(mask_path).get_fdata()).round().astype(int)

    # same axis-order fix confirmed necessary for LeFusion synthetic masks
    if img_data.shape != mask_data.shape and sorted(img_data.shape) == sorted(mask_data.shape):
        remaining = list(range(mask_data.ndim))
        perm = []
        for target_size in img_data.shape:
            for ax in remaining:
                if mask_data.shape[ax] == target_size:
                    perm.append(ax)
                    remaining.remove(ax)
                    break
        mask_data = np.transpose(mask_data, perm)

    return img_data, mask_data


def render_slice_png(img_data, mask_data, slice_idx: int, show_mask: bool) -> bytes:
    img_slice = img_data[:, :, slice_idx]
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#111111")
    ax.imshow(img_slice.T, cmap="gray", origin="lower")

    if show_mask:
        mask_slice = mask_data[:, :, slice_idx]
        overlay = np.zeros((*mask_slice.T.shape, 4))
        for label, color in LABEL_COLORS.items():
            r, g, b = [c / 255 for c in color]
            overlay[(mask_slice.T == label)] = (r, g, b, 0.5)
        ax.imshow(overlay, origin="lower")

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, facecolor="#111111")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def get_latest_decisions():
    """Reads the append-only log and returns only the most recent decision
    per case, so 'status' always reflects the latest choice, not a history
    of every click. This is what makes revisiting and changing a decision
    work correctly."""
    latest = {}
    if os.path.exists(ANNOTATIONS_PATH):
        with open(ANNOTATIONS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec["case_type"], rec["case_id"])
                # later lines are always more recent since we only append
                latest[key] = rec
    return latest


@app.route("/api/summary")
def api_summary():
    all_cases = find_all_cases()
    latest = get_latest_decisions()
    counts = {"pending": 0, "accepted": 0, "rejected": 0}
    case_list = []
    for c in all_cases:
        rec = latest.get((c["type"], c["id"]))
        status = "pending" if rec is None else ("accepted" if rec["decision"] == "accept" else "rejected")
        counts[status] += 1
        case_list.append({"id": c["id"], "type": c["type"], "status": status})
    return jsonify({"counts": counts, "total": len(all_cases), "cases": case_list})


@app.route("/api/case/<case_type>/<case_id>/latest")
def api_case_latest(case_type, case_id):
    latest = get_latest_decisions()
    rec = latest.get((case_type, case_id))
    return jsonify(rec) if rec else jsonify(None)


@app.route("/api/export")
def api_export():
    """Returns a manifest of every currently-accepted case with its real
    file paths, ready to be handed directly to a training pipeline --
    this is the actual output of the whole review process."""
    latest = get_latest_decisions()
    manifest = []
    for (case_type, case_id), rec in latest.items():
        if rec["decision"] != "accept":
            continue
        img_path, mask_path = get_case_paths(case_id, case_type)
        manifest.append({
            "case_id": case_id,
            "case_type": case_type,
            "image_path": img_path,
            "mask_path": mask_path,
            "reviewed_at": rec["timestamp"],
            "comment": rec.get("comment", ""),
        })
    return jsonify({"n_accepted": len(manifest), "cases": manifest})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cases")
def api_cases():
    return jsonify(find_all_cases())


@app.route("/api/case/<case_type>/<case_id>/info")
def api_case_info(case_type, case_id):
    try:
        img_data, mask_data = load_case_arrays(case_id, case_type)
    except Exception as e:
        return jsonify({"error": str(e)}), 404
    return jsonify({
        "case_id": case_id,
        "case_type": case_type,
        "n_slices": int(img_data.shape[2]),
        "shape": list(img_data.shape),
    })


@app.route("/api/case/<case_type>/<case_id>/slice/<int:slice_idx>.png")
def api_case_slice(case_type, case_id, slice_idx):
    show_mask = request.args.get("mask", "1") == "1"
    try:
        img_data, mask_data = load_case_arrays(case_id, case_type)
        if slice_idx < 0 or slice_idx >= img_data.shape[2]:
            return jsonify({"error": "slice index out of range"}), 400
        png_bytes = render_slice_png(img_data, mask_data, slice_idx, show_mask)
    except Exception as e:
        return jsonify({"error": str(e)}), 404
    return send_file(io.BytesIO(png_bytes), mimetype="image/png")


@app.route("/api/validate", methods=["POST"])
def api_validate():
    data = request.get_json()
    required = ["case_id", "case_type", "decision"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"missing field: {field}"}), 400
    if data["decision"] not in ("accept", "reject"):
        return jsonify({"error": "decision must be 'accept' or 'reject'"}), 400

    record = {
        "case_id": data["case_id"],
        "case_type": data["case_type"],
        "decision": data["decision"],
        "comment": data.get("comment", ""),
        "bbox": data.get("bbox"),
        "slice_idx": data.get("slice_idx"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(ANNOTATIONS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    return jsonify({"status": "saved", "record": record})


@app.route("/api/validations")
def api_validations():
    """Returns all saved decisions so far, useful for checking progress."""
    records = []
    if os.path.exists(ANNOTATIONS_PATH):
        with open(ANNOTATIONS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return jsonify(records)


if __name__ == "__main__":
    print(f"Real cases folder:      {REAL_CASES_DIR}")
    print(f"Synthetic cases folder: {SYNTHETIC_CASES_DIR}")
    print(f"Annotations saved to:   {ANNOTATIONS_PATH}")
    cases = find_all_cases()
    print(f"Found {len(cases)} cases total.")
    app.run(host="0.0.0.0", port=5050, debug=True)
