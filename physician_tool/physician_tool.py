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
import re
from datetime import datetime, timezone

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, jsonify, request, send_file, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Defaults follow the CURRENT WORKING DIRECTORY, not the script's own
# location -- matching every other script in this project (run from your
# EMIDEC folder, relative paths). Confirmed necessary after a real run
# showed "total: 0" with no error: the script silently defaulted to
# looking for data next to itself in the repo, not where it was run from.
WORK_DIR = os.getcwd()
REAL_CASES_DIR = os.environ.get("REAL_CASES_DIR", WORK_DIR)
SYNTHETIC_CASES_DIR = os.environ.get("SYNTHETIC_CASES_DIR", os.path.join(WORK_DIR, "synthetic_cases"))
ANNOTATIONS_PATH = os.path.join(WORK_DIR, "annotations", "image_validation.jsonl")
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


REPORTS_DIR = os.environ.get("REPORTS_DIR", os.path.join(WORK_DIR, "outputs_region_grounded"))
REPORT_OVERLAYS_DIR = os.environ.get("REPORT_OVERLAYS_DIR", os.path.join(WORK_DIR, "overlays"))
REPORT_ANNOTATIONS_PATH = os.path.join(WORK_DIR, "annotations", "report_validation.jsonl")


def find_all_reports():
    reports = []
    if os.path.isdir(REPORTS_DIR):
        for f in sorted(os.listdir(REPORTS_DIR)):
            if f.endswith(".json"):
                reports.append({"id": f.replace(".json", "")})
    return reports


def load_report(case_id: str) -> dict:
    path = os.path.join(REPORTS_DIR, f"{case_id}.json")
    with open(path) as f:
        return json.load(f)


def split_into_sentences(text: str) -> list:
    """Splits report text into individually flaggable sentences. Deliberately
    simple (split on sentence-ending punctuation followed by whitespace)
    rather than free-form text selection -- much easier to save, restore on
    revisit, and reason about reliably than arbitrary text ranges."""
    text = text.replace("**", "").replace("*", "")
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


REGION_TAG_RE = re.compile(r"\[region:\s*([a-zA-Z_]+)\]")

# label ids from LABEL_COLORS above, keyed by the same region names the
# report generation prompts use in [region: ...] tags (see
# report_generation/prompts_region_grounded.py DEFAULT_REGION_LEGEND).
# "myocardium" here intentionally maps to label 2, matching LABEL_COLORS.
REGION_LABEL_IDS = {"lv_cavity": 1, "myocardium": 2, "infarct": 3, "mvo": 4}


def split_into_findings(text: str) -> list:
    """Pairs each Findings sentence with the region name from its trailing
    [region: ...] tag (tag stripped from the displayed text), for the
    callout visualization only -- the flowing-paragraph view keeps using
    split_into_sentences unchanged, so report generation and the existing
    view are untouched.

    Deliberately NOT split_into_sentences + a per-chunk tag search: splitting
    on sentence-ending punctuation first breaks each tag away from the
    sentence it belongs to and onto the front of the next one (confirmed by
    hand -- "...size. [region: lv_cavity] There is..." splits into "...size."
    and "[region: lv_cavity] There is..."). Walking tag matches directly and
    taking the text since the previous tag avoids that mispairing."""
    text = text.replace("**", "").replace("*", "").strip()
    findings = []
    pos = 0
    for match in REGION_TAG_RE.finditer(text):
        chunk = text[pos:match.start()].strip()
        if chunk:
            findings.append({"text": chunk, "region": match.group(1)})
        pos = match.end()
    return findings


def compute_region_centroid(mask_2d: np.ndarray, label: int):
    """Same centroid pattern used in assign_aha17_segment.py / analyze_slice
    (verify_slice_claims.py): the mean row/col of a label's pixels. Returns
    normalized (x, y) in 0-1 image-fraction coordinates, or None if the label
    isn't present on this slice. mask_2d is indexed [row, col] == [y, x],
    same as the array overlay_renderer.py renders directly to PNG with no
    transpose, so col-mean -> x fraction, row-mean -> y fraction lines up
    with the rendered image without any additional flip."""
    ys, xs = np.where(mask_2d == label)
    if len(ys) == 0:
        return None
    h, w = mask_2d.shape
    return {"x": float(xs.mean()) / w, "y": float(ys.mean()) / h}


def compute_report_callouts(case_id: str, data: dict) -> list:
    """Builds one callout per Findings sentence: its text and where its
    region's real centroid sits on the slice image, using the case's real
    mask (not the model's claim) -- same source of truth
    assign_aha17_segment.py uses for its own placement math."""
    findings = split_into_findings(data.get("output", ""))
    if not findings:
        return []

    slice_used = data.get("slice_used", "")
    match = re.search(r"slice_(\d+)", slice_used or "")
    if not match:
        return []
    slice_idx = int(match.group(1))

    try:
        img_data, mask_data = load_case_arrays(case_id, "real")
    except Exception:
        return []
    if slice_idx < 0 or slice_idx >= mask_data.shape[2]:
        return []
    # NOT transposed: the report's slice image is the overlay_renderer.py PNG,
    # which does Image.fromarray(mask_slice) directly with no transpose/flip
    # (unlike render_slice_png's ax.imshow(...T, origin="lower") used in
    # Module 1) -- so mask_2d must stay in that same [row, col] orientation
    # for centroid fractions to land on the right spot in the displayed PNG.
    mask_2d = mask_data[:, :, slice_idx]

    callouts = []
    for i, finding in enumerate(findings):
        label = REGION_LABEL_IDS.get(finding["region"])
        centroid = compute_region_centroid(mask_2d, label) if label is not None else None
        callouts.append({
            "finding_index": i,
            "text": finding["text"],
            "region": finding["region"],
            "x": centroid["x"] if centroid else None,
            "y": centroid["y"] if centroid else None,
        })
    return callouts


def find_report_slice_image(case_id: str, slice_used: str) -> str:
    """slice_used looks like 'overlays/Case_P019/slice_05_plain.png' or
    similar -- resolve it against REPORT_OVERLAYS_DIR rather than trusting
    the stored path directly, since it may have been generated on a
    different machine with a different folder layout."""
    match = re.search(r"slice_(\d+)", slice_used or "")
    if not match:
        return None
    slice_num = match.group(1)
    candidate = os.path.join(REPORT_OVERLAYS_DIR, case_id, f"slice_{slice_num}_overlay.png")
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(REPORT_OVERLAYS_DIR, case_id, f"slice_{slice_num}_plain.png")
    if os.path.exists(candidate):
        return candidate
    return None


@app.route("/api/reports")
def api_reports():
    return jsonify(find_all_reports())


@app.route("/api/report/<case_id>")
def api_report(case_id):
    try:
        data = load_report(case_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 404
    sentences = split_into_sentences(data.get("output", ""))
    return jsonify({
        "case_id": case_id,
        "model": data.get("model"),
        "slice_used": data.get("slice_used"),
        "sentences": sentences,
        "has_image": find_report_slice_image(case_id, data.get("slice_used", "")) is not None,
    })


@app.route("/api/report/<case_id>/image.png")
def api_report_image(case_id):
    try:
        data = load_report(case_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 404
    img_path = find_report_slice_image(case_id, data.get("slice_used", ""))
    if not img_path:
        return jsonify({"error": "no image found for this report"}), 404
    return send_file(img_path, mimetype="image/png")


@app.route("/api/report/<case_id>/callouts")
def api_report_callouts(case_id):
    """Same report data as /api/report/<case_id>, just re-packaged as
    per-finding callouts with real mask-centroid positions -- an alternate
    view, not a different source of truth. Only available for real cases
    (reports are only ever generated from real EMIDEC cases, see
    build_patient_jsons.py), so a synthetic case_id here simply yields no
    positioned callouts rather than erroring."""
    try:
        data = load_report(case_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 404
    callouts = compute_report_callouts(case_id, data)
    return jsonify({"case_id": case_id, "callouts": callouts})


@app.route("/api/validate_report", methods=["POST"])
def api_validate_report():
    data = request.get_json()
    required = ["case_id", "decision"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"missing field: {field}"}), 400
    if data["decision"] not in ("accept", "reject"):
        return jsonify({"error": "decision must be 'accept' or 'reject'"}), 400

    record = {
        "case_id": data["case_id"],
        "decision": data["decision"],
        "comment": data.get("comment", ""),
        "flagged_sentences": data.get("flagged_sentences", []),
        # physician edits made in the callout visualization (Module 2's
        # alternate view) -- {finding_index, text} pairs, keyed against
        # split_into_findings, separate from flagged_sentences' indices
        # (split_into_sentences) since the two splits aren't the same list.
        "callout_edits": data.get("callout_edits", []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(REPORT_ANNOTATIONS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return jsonify({"status": "saved", "record": record})


def get_latest_report_decisions():
    latest = {}
    if os.path.exists(REPORT_ANNOTATIONS_PATH):
        with open(REPORT_ANNOTATIONS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                latest[rec["case_id"]] = rec
    return latest


@app.route("/api/report/<case_id>/latest")
def api_report_latest(case_id):
    latest = get_latest_report_decisions()
    rec = latest.get(case_id)
    return jsonify(rec) if rec else jsonify(None)


@app.route("/api/report_summary")
def api_report_summary():
    all_reports = find_all_reports()
    latest = get_latest_report_decisions()
    counts = {"pending": 0, "accepted": 0, "rejected": 0}
    report_list = []
    for r in all_reports:
        rec = latest.get(r["id"])
        status = "pending" if rec is None else ("accepted" if rec["decision"] == "accept" else "rejected")
        counts[status] += 1
        report_list.append({"id": r["id"], "status": status})
    return jsonify({"counts": counts, "total": len(all_reports), "reports": report_list})


@app.route("/api/export_reports")
def api_export_reports():
    """Manifest of every accepted report, plus every flagged sentence across
    ALL reviewed reports (accepted or not) -- the flagged sentences are the
    real error-analysis material, useful even from a rejected report."""
    latest = get_latest_report_decisions()
    accepted = []
    all_flags = []
    all_callout_edits = []
    for case_id, rec in latest.items():
        if rec["decision"] == "accept":
            accepted.append({"case_id": case_id, "reviewed_at": rec["timestamp"], "comment": rec.get("comment", "")})
        for flag in rec.get("flagged_sentences", []):
            all_flags.append({"case_id": case_id, **flag})
        for edit in rec.get("callout_edits", []):
            all_callout_edits.append({"case_id": case_id, **edit})
    return jsonify({"n_accepted": len(accepted), "accepted": accepted,
                     "n_flagged_sentences": len(all_flags), "flagged_sentences": all_flags,
                     "n_callout_edits": len(all_callout_edits), "callout_edits": all_callout_edits})



if __name__ == "__main__":
    print(f"Working directory:      {WORK_DIR}")
    print(f"Real cases folder:      {REAL_CASES_DIR}")
    print(f"Synthetic cases folder: {SYNTHETIC_CASES_DIR}")
    print(f"Reports folder:         {REPORTS_DIR}")
    print(f"Report overlays folder: {REPORT_OVERLAYS_DIR}")
    print(f"Annotations saved to:   {ANNOTATIONS_PATH}")
    cases = find_all_cases()
    reports = find_all_reports()
    print(f"Found {len(cases)} image/mask cases and {len(reports)} reports.")
    app.run(host="0.0.0.0", port=5050, debug=True)
