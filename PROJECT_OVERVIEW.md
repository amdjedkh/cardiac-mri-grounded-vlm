# Project Overview

This repo builds reports for cardiac MRI scans that can be checked against the scan, and tests whether synthetic (computer-generated) scans help. It describes the repo as of 2026-10-08. Where the older docs and the code disagree, the code wins, and the differences are listed at the end.

## 1. The idea

A vision-language model (VLM) reads a cardiac MRI and writes a report. Each statement points to a real region in the image (for example the infarct), so a person can check it. Cardiologists then correct the reports.

Three hypotheses:

- **H1:** reports built with mask measurements and region links are more correct, with fewer unsupported claims, than reports from the image alone.
- **H2:** cardiologist-corrected reports are better training data than uncorrected model reports.
- **H3:** synthetic scans with masks, filtered and mixed with real data, improve reports on unseen real patients.

Plan: train on a small set of expert-checked real cases (gold) plus a larger set of unchecked synthetic cases (silver), then test on real patients the model has not seen.

No public dataset pairs cardiac MRI masks with real grounded reports, so the reports are generated first and corrected later. That is why the repo has verification and review tools.

## 2. Data

| Data | What it is | Where |
|---|---|---|
| Real cases | EMIDEC dataset, 100 cases (N = normal, P = pathological). Each has an MRI, a mask and a clinical `.txt` file. | Local, not in git |
| Synthetic cases | Made by LeFusion from real pathological cases. Image + mask per case. 103 downloaded. | Modal volume `cardiac-data`, local `synthetic_cases/` |
| Patient JSONs | Measurements and clinical info per case | `patients/` |
| Overlays | Plain and mask-overlay PNG per slice | `overlays/<case>/` |
| Reports | One JSON per case | `outputs_region_grounded/` |
| Physician decisions | One JSON line per decision | `annotations/*.jsonl` |

Derived files and physician annotations are gitignored.

**Mask labels:** 0 background, 1 LV cavity, 2 myocardium, 3 infarct, 4 MVO. These are used for real, synthetic and any future predicted masks.

**Synthetic vs real:**

- Synthetic images are small (about 72x72, about 10 slices) and cropped around the heart. Real scans show the whole chest.
- The synthetic mask has its axes in a different order from the image (`[D,H,W]` vs `[H,W,D]`) and can have a different slice count. Code that reads both fixes this.
- Synthetic intensities are about -1 to 1. Real ones are in the thousands.
- Synthetic cases have no clinical data, so their reports use only the image.
- A synthetic file is named after its source real case. Nobody has confirmed it shows the same patient.
- About 47 pathological cases can be used as inputs. LeFusion's built-in test split (10 cases) is excluded.

## 3. Pipeline

Run scripts from the folder that holds the EMIDEC data. Most use relative paths.

| Stage | What happens | Scripts |
|---|---|---|
| 1. Prep | Parse clinical files (must be read as `cp1252`), compute infarct and MVO measurements, render overlay PNGs | `data_pipeline/`: `patient_schema.py`, `measurements.py`, `build_patient_jsons.py`, `overlay_renderer.py`, `overlay_renderer_synthetic.py` |
| 2. Synthetic generation | LeFusion on Modal. Only `emidec.pt` is used, not `diffmask.pt`. | `synthetic_generation/`: `modal_lefusion.py`, `trigger_lefusion.py`, `check_lefusion_result.py` |
| 3. Synthetic checks | Geometry, copy-vs-new, slice-to-slice smoothness | `synthetic_generation/check_*.py`, `plot_consistency_comparison.py`; `verification/slice_consistency_metrics.py` |
| 4. Reports | Gemini or MedGemma writes a report from the middle slice's plain and overlay images | `report_generation/` |
| 5. Claim checks | Compare report claims with the mask | `verification/` |
| 6. Physician review | Web app to accept, reject and edit | `physician_tool/` |
| 7. Fine-tuning and final test | **Not built yet** | |

### Generation details (stage 2)

`generate_synthetic_emidec(batch_size, max_cases, reuse_when_exhausted)` runs on an A100. It skips finished cases and empties its staging folder each run. With `reuse_when_exhausted=True` it reuses real cases under new `__dup` filenames when the unused ones run out. `synthetic_generation/README.md` lists the setup problems already solved.

### Report generation (stage 4)

- **Current method:** `prompts_region_grounded.py`. Every Findings sentence ends with `[region: lv_cavity | myocardium | infarct | mvo]`. Reports have Technique, Findings and Impression. Run `run_region_grounded_poc.py` (real cases) or `run_region_grounded_synthetic.py` (synthetic). Both need `GEMINI_API_KEY` and skip finished cases.
- **Replaced method:** `prompts.py`, `run_gemini.py`, `run_medgemma.py`, `compare_baselines.py`. Tags pointed to measurement fields, not image regions.
- **MedGemma tests:** Gemini tagged 8 of 8 cases correctly. MedGemma-4B got 0 of 8 zero-shot. One example fixed the format but it copied the example's content. Two examples made it worse. Prompting fixes format, not image reading, at this size.
- **Format checkers:** `recheck_any_folder.py <dir>` is the reliable one.

### Claim checks (stage 5)

- `verify_slice_claims.py <case> <slice>`: how much of the heart wall a region covers, whether it forms one arc or scattered pieces, and where it sits along the arc (0 edge, 0.5 centre, 1 edge).
- `assign_aha17_segment.py <case> <slice>`: AHA 17-segment location. Top of image = anterior (checked with the sternum). Slice 0 = base is assumed.
- `check_location_claims_batch.py`: compares the wall named in every report with the computed wall. It ignores basal/mid/apical because of the unchecked slice order.
- `render_case_bullseye.py <case>`, `spot_check_reports.py`: bullseye plot and a check of four known cases.

This has caught an MVO called "central" that sat near one edge (0.81), and an "anteroseptal" claim that was really mid anterior.

### Physician tool (stage 6)

A Flask app with two modules:

- **Module 1, image and mask check.** Real and synthetic cases in one list. Scroll slices, toggle the mask, draw a box, accept or reject, comment. Keys: `A`, `R`, arrows, `Enter`.
- **Module 2, report check.** A report next to its slice. Flag sentences, or use the callout view: one editable card per finding, tied to a region marker on the image. Exports include flagged sentences from rejected reports, for error analysis.

`modal_physician_tool.py` deploys it on Modal with basic auth (secret `physician-tool-auth`, volume `physician-tool-data`). If the live app shows zero cases, run `modal run physician_tool/modal_physician_tool.py` to list the volume.

## 4. Current state

**Done:** data prep and overlays (real and synthetic); region-grounded Gemini reports on real cases; claim-checking tools; LeFusion generation scaled to 103 cases; physician tool with both modules, deployed; MedGemma prompt tests (negative result).

**In progress (uncommitted):** `overlay_renderer_synthetic.py`, `run_region_grounded_synthetic.py`, and a `physician_tool.py` change so callouts work for synthetic cases.

**Open:**

1. **Field of view.** Synthetic images are cropped to the heart, real ones are not. Train as is, or crop the real images?
2. **Same-patient pairing is unverified.** After fixing axes and resampling, real P004 vs synthetic P004 matched 73.2%, but two unrelated real patients (P004 vs P019) matched 97.6%. That is backwards for a true pair. Synthetic P001 also looked different from real P001. The proper test (crop the real image to the heart, compare against the source and several other patients) has not been run.
3. **Is the synthetic geometry new?** It may be new texture on the real mask. `diffmask.pt` is unexplored.
4. **AHA17 base/apex order** is assumed.
5. **Few synthetic cases checked.** Real contiguity scores range from 0.591 to 0.739. The one synthetic case scanned over the whole volume scored 0.741. A low score on one slice of P001 (0.495) was a single-slice fluke.
6. **Cardiologist annotation protocol** is not started, and no physician has reviewed reports at scale.
7. **Evaluation metrics.** BLEU, ROUGE and BERTScore need reference reports that do not exist. Dice and IoU need predicted masks, but the pipeline only uses ground-truth masks.
8. **Other generators to test:** NVIDIA `NV-Generate-CTMR` (no masks, cardiac not listed) and `github.com/SoufianeBH/Paired-Image-Segmentation-Synthesis` (LGE image and mask together).

## 5. How to run

**Setup**

- `pip install -r requirements.txt`. LeFusion's pinned dependencies live inside `modal_lefusion.py` and run only on Modal.
- EMIDEC in release layout: `Case_XXXX/Images/` and `Case_XXXX/Contours/`, clinical `.txt` files at the root.
- `GEMINI_API_KEY`, `modal setup`, and Modal secret `huggingface-secret` (for MedGemma).
- Modal names are hardcoded: apps `lefusion-emidec-poc`, `medgemma-emidec-poc`, `physician-tool`; volumes `cardiac-data`, `physician-tool-data`.

```bash
# 1. prep
python data_pipeline/scan_all_cases.py
python data_pipeline/build_patient_jsons.py
python data_pipeline/overlay_renderer.py Case_P019 Case_P004

# 2. synthetic data (deploy, never `modal run`)
modal deploy synthetic_generation/modal_lefusion.py
python synthetic_generation/trigger_lefusion.py        # set max_cases inside the script
python synthetic_generation/check_lefusion_result.py <call_id>
modal volume get cardiac-data LeFusion_output/Image ./synthetic_cases/Image
modal volume get cardiac-data LeFusion_output/Mask  ./synthetic_cases/Mask
export SYNTHETIC_CASES_DIR=./synthetic_cases
python data_pipeline/overlay_renderer_synthetic.py Case_P001

# 3. reports
export GEMINI_API_KEY=...
python report_generation/run_region_grounded_poc.py
python report_generation/run_region_grounded_synthetic.py
python report_generation/recheck_any_folder.py outputs_region_grounded

# 4. checks
python verification/verify_slice_claims.py Case_P019 5
python verification/check_location_claims_batch.py

# 5. physician tool, http://localhost:5050
export REAL_CASES_DIR=/path/to/emidec REPORTS_DIR=./outputs_region_grounded REPORT_OVERLAYS_DIR=./overlays
python physician_tool/physician_tool.py
```

`modal run` creates a temporary app that stops when the local script ends, which cancels the background job. Always use `modal deploy`.

These commands come from reading the scripts. I did not run them.

## 6. Working principles

- **Test on a known answer first.** This caught wrong labels, wrong slice picking, the mask axis bug and a miscounting report checker.
- **Check outputs, do not trust them.** Report claims are measured against the mask. Synthetic data is checked for copying, smoothness and plausible geometry.
- **State sample sizes.** Results from a few cases are called preliminary.
- **Keep failed experiments.** Replaced prompts and the MedGemma runs stay in the repo.
- **Safe to rerun.** Scripts skip finished work. Physician decisions are append-only.
- **No patient data in git.**
- **Not config-driven yet.** Paths use environment variables. Settings like `max_cases` are hardcoded in `trigger_lefusion.py`. There is no config file.
- **Keep test patients out of synthetic inputs.**

## 7. Glossary

- **EMIDEC:** public cardiac MRI dataset with infarct contours.
- **LGE:** late gadolinium enhancement. Scar tissue looks bright.
- **LV cavity / myocardium:** the blood-filled chamber of the left ventricle / the muscle wall around it.
- **Infarct:** dead heart muscle after a blocked blood supply (label 3).
- **MVO:** microvascular obstruction, a blocked area inside the infarct (label 4).
- **AHA17:** the standard 17-segment map of the left ventricle.
- **Basal / apical:** near the valve / near the tip.
- **Angular coverage:** percent of the heart wall ring that a region covers.
- **Contiguity score:** how much a region forms one smooth arc instead of scattered pieces.
- **Region-grounded report:** each Findings sentence ends with `[region: <name>]`.
- **LeFusion:** a diffusion model that generates synthetic lesion MRI and masks.
- **Conditioning case:** the real case LeFusion uses as input.
- **Gold / silver:** expert-checked real data / unchecked synthetic data.
- **MedGemma-4B-IT:** a small open medical VLM used as a baseline.
- **Callout view:** one card per finding, tied to a marker on the image.
- **Modal:** the cloud GPU platform used here.

## 8. README/code mismatches

1. The old README said only Module 1 of the physician tool exists. The code also has Module 2. (README now fixed.)
2. The handoff doc says generation runs on A10G. It now uses an A100. A comment in `modal_lefusion.py` still mentions switching.
3. The handoff doc lists 6 synthetic cases. There are 103. `trigger_lefusion.py` has a hardcoded "91 already exist" message that will go stale.
4. Report images show MVO in yellow. `physician_tool.py` shows MVO in green and myocardium in yellow.
5. `synthetic_generation/README.md` is written for one named collaborator and says the setup uses "my own" Modal account.
6. `README.md` and `docs/PROJECT_HANDOFF_CONTEXT.md` name supervisors and an institution. The handoff doc is a prompt for continuing a conversation, not a neutral project doc. Its Windows paths and Python 3.13 do not match this checkout.
7. `physician_tool/test.py` is one `print("this is a test")` line.
