# Cardiac MRI Grounded VLM

Research code for cardiac MRI reports that can be checked against the scan: region-grounded report generation, synthetic data, geometry checks, and physician review. For a plain-language summary of the whole project, see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

## Repo structure

```
data_pipeline/         EMIDEC loading, measurement extraction, overlay rendering
report_generation/     Prompt scaffolds + runners for Gemini/MedGemma report generation
verification/          Quantitative geometry checks (angular coverage, contiguity, AHA17)
synthetic_generation/  LeFusion synthetic MRI+mask generation on Modal
physician_tool/        Flask web app for physician review (Module 1: image/mask, Module 2: reports) + Modal deployment
docs/                  Additional project context
```

### `data_pipeline/`
- `measurements.py` — extracts infarct volume, infarct %, MVO volume, affected slices from a mask
- `patient_schema.py` — parses EMIDEC clinical `.txt` files (note: must be read as `cp1252`, not UTF-8 — EMIDEC uses non-breaking spaces before colons, a French typographic convention, which silently breaks UTF-8 parsing)
- `overlay_renderer.py` — renders MRI + mask overlay PNGs (red=infarct, yellow=MVO)
- `scan_all_cases.py` — scans all 100 EMIDEC cases, builds a summary CSV
- `build_patient_jsons.py` — builds per-patient JSON records with measurements + clinical metadata
- `verify_labels.py` — confirms the EMIDEC label mapping against the official challenge repo
- `overlay_renderer_synthetic.py` — same overlay PNGs for synthetic (LeFusion) cases; applies the mask axis fix and per-slice intensity normalization
- `debug_sex_field.py` — diagnostic script for the clinical file encoding issue

**Verified EMIDEC label mapping:** `0=background, 1=LV cavity, 2=myocardium, 3=infarct, 4=MVO/no-reflow`

### `report_generation/`
Two generations of approach here, both present for reference:
- `prompts.py` — **superseded.** Original approach, grounding tags pointed to measurement field names (e.g. `infarct_volume_ml`), not visual regions. Carlos clarified this isn't real grounding.
- `prompts_region_grounded.py` — **current approach.** Tags in `[region: name]` format tied to actual segmentation regions. Verified against real geometry, not just format.
- `prompts_region_grounded_fewshot.py` / `_fewshot2.py` — experiments testing whether MedGemma's poor performance was a prompting issue (Karen's hypothesis). One example fixed formatting but caused content-copying; two examples made it worse. Conclusion: prompting fixes format, not reliable content understanding, at 4B scale.
- `run_region_grounded_synthetic.py` — region-grounded Gemini reports for synthetic cases (no clinical data, image + overlay only)
- `run_*.py` — runners for each experiment (Gemini and MedGemma, original and region-grounded, fewshot variants)
- `modal_medgemma.py` — Modal deployment for MedGemma-4B-IT (A10G GPU)
- `compare_baselines.py`, `recheck_region_grounding.py`, `recheck_any_folder.py` — automated checkers for grounding format compliance. **Note:** the checker logic went through several real bug fixes (markdown formatting artifacts, then numbered-list formatting artifacts breaking the sentence-counting heuristic) — see `recheck_any_folder.py` for the current, most robust version.

### `verification/`
Quantitative tools to check report claims against actual mask geometry, rather than trusting model output or eyeballing images.
- `verify_slice_claims.py` — `angular_coverage()` (what % of the ring a region covers), `contiguity_score()` (smooth solid arc vs. scattered fragments), `position_along_arc()` (is a region centered or off to one edge)
- `assign_aha17_segment.py` — computes AHA 17-segment assignment from mask geometry. **Orientation note:** top-of-image = anterior wall was confirmed using the sternum as a landmark. Which end of the slice stack is basal vs. apical was assumed, never independently confirmed.
- `check_location_claims_batch.py` — checks the wall named in every report (anterior/inferior/septal/lateral) against the AHA17 wall computed from the mask
- `spot_check_reports.py` — checks reports for P019, P004, P055, P060 against geometry already established
- `slice_consistency_metrics.py`, `scan_real_consistency.py` — slice-to-slice smoothness of cavity size/centroid on real cases (shared metric used by the synthetic checks)
- `render_case_bullseye.py` — renders a real, data-driven AHA17 bullseye plot colored by actual computed infarct/MVO involvement per segment

This tooling has already caught real issues worth knowing about: a report claiming MVO was "central" when it was actually near one edge (0.81 on a 0=edge/0.5=center/1=edge scale), and a claimed "anteroseptal" location that was actually "mid anterior."

### `synthetic_generation/`
LeFusion (`github.com/HINTLab/LeFusion`) deployed on Modal to generate synthetic EMIDEC-style cardiac MRI + masks from pretrained weights.
- `modal_lefusion.py` — the Modal app. Functions: `setup_data_and_weights`, `generate_synthetic_emidec` (A100; `max_cases` for small batches; `reuse_when_exhausted=True` reuses real cases under unique `__dup` filenames once the unused pool runs out), `analyze_synthetic_geometry`, `scan_all_slices_geometry`, `compare_synthetic_to_real_mask`, `render_synthetic_case`
- Generation notes: the staging folder is now emptied every run (before, old cases were silently reprocessed), and LeFusion's hardcoded 10-case test split is excluded from the input pool (about 47 usable cases). Local output is now 103 synthetic cases.
- `check_synthetic_consistency.py`, `plot_consistency_comparison.py` — slice-to-slice smoothness for synthetic cases, and real-vs-synthetic plots. Full details: [synthetic_generation/README.md](synthetic_generation/README.md)
- `trigger_lefusion.py` — triggers generation on the **deployed** app (not `modal run`, see note below)
- `check_lefusion_result.py` — polls a spawned job by call ID, works from any session
- `render_and_download.py`, `check_synthetic_geometry.py`, `check_all_slices.py` — inspection/visualization tools for synthetic cases
- `scan_real_case_all_slices.py`, `compare_two_real_cases.py` — matching tools for real cases, used to build fair real-vs-synthetic comparisons
- `check_real_vs_synthetic_identity.py` — checks whether a "synthetic" case is genuinely new content or effectively a copy of its real conditioning case (axis-order-corrected, resolution-resampled pixel comparison)

**Important workflow note:** use `modal deploy modal_lefusion.py` then `python trigger_lefusion.py`, never `modal run`. `modal run` creates a temporary app that gets torn down the moment the local script finishes, which silently kills any spawned background job — confirmed via a real run that showed `Status=Cancelled` with zero containers ever going live.

**Known real bugs already fixed in this pipeline** (worth knowing before assuming something new is broken): pip version conflicts with LeFusion's pinned dependencies, two packages missing from a hand-transcribed requirements list (fixed by installing LeFusion's complete original list instead), a conflict where LeFusion's pinned `typing_extensions` broke Modal's own SDK, a missing system graphics library for `opencv-python`, a nested-path issue in the preprocessed data archive, and — most subtly — the synthetic mask array is stored with a different axis order than the image array (`[D,H,W]` vs `[H,W,D]`), which is corrected in every function that touches mask data.

### `physician_tool/`
Flask web app for physician review, with two modules. Decisions are appended to JSONL files in `annotations/`.

- **Module 1, image/mask validation.** Real and synthetic cases in one list. Scroll slices, toggle the mask, optionally draw a box, accept or reject, add a comment. Reopening a case pre-fills the earlier decision. `Export accepted dataset` gives a JSON list of accepted cases. Keys: `A` accept, `R` reject, arrows to move, `Enter` to submit. Saves to `annotations/image_validation.jsonl`.
- **Module 2, report validation.** Shows a generated report next to its slice. Physicians can flag individual sentences, or use the callout view: one card per finding in a row above the image, each tied to a region computed from the mask, with editable text. Works for real and synthetic cases. Saves to `annotations/report_validation.jsonl`. Export (`/api/export_reports`) includes flagged sentences from rejected reports too, for error analysis.
- `modal_physician_tool.py` — deploys the app on Modal with basic auth: `modal deploy physician_tool/modal_physician_tool.py`. Needs volume `physician-tool-data` (`real_cases/`, `synthetic_cases/`, `outputs_region_grounded/`, `overlays/`) and secret `physician-tool-auth` (`BASIC_AUTH_USER`, `BASIC_AUTH_PASSWORD`). If the live app shows zero cases, run `modal run physician_tool/modal_physician_tool.py` to list the volume contents.

Run locally:
```bash
pip install flask nibabel numpy matplotlib scipy
export REAL_CASES_DIR=/path/to/emidec/cases
export SYNTHETIC_CASES_DIR=/path/to/downloaded/synthetic/cases
export REPORTS_DIR=/path/to/outputs_region_grounded       # Module 2
export REPORT_OVERLAYS_DIR=/path/to/overlays              # Module 2
python physician_tool/physician_tool.py
```
Then open `http://localhost:5050`.

Synthetic cases must be downloaded first, since they live on the Modal volume:
```bash
modal volume get cardiac-data LeFusion_output/Image ./synthetic_cases/Image
modal volume get cardiac-data LeFusion_output/Mask ./synthetic_cases/Mask
```

## Open questions — not yet resolved

1. **Field-of-view mismatch between real and synthetic images.** Synthetic images are tightly cropped to almost just the heart; real EMIDEC scans show the whole chest. Sent to Karen/Carlos as an open question (train on this as-is, or crop real images to match?). Awaiting reply.
2. **Same-patient identity pairing is unverified.** LeFusion names synthetic output files after their real conditioning case (e.g. synthetic `Case_P004.nii.gz` is assumed to correspond to real `Case_P004`), but this was never independently confirmed. A pixel comparison (axis-corrected, resampled) showed real-P004-vs-synthetic-P004 matching at only 73%, while two totally unrelated real patients matched at 97% — backwards from what you'd expect if genuinely paired. A proper controlled test (crop real image to heart-only region first, then compare against both its supposed source and several random other patients) has not been done.
3. **AHA17 base-vs-apex slice ordering** was assumed, not confirmed, unlike the anterior/posterior orientation (confirmed via the sternum landmark).
4. **Cardiologist annotation protocol** — not started.
5. **Karen's requested evaluation metrics** (BLEU/ROUGE/BERTScore for text, Dice/IoU/precision/accuracy for segmentation) — flagged back to her as needing clarification: text metrics need a reference report that doesn't exist yet, segmentation metrics need a predicted mask, but the pipeline currently only uses ground-truth masks as input.
6. **Synthetic data quality checked on only a handful of cases so far.** The contiguity metric showed real cases range 0.591–0.739 and the one synthetic case checked thoroughly (whole-volume) scored 0.741 — no red flag yet, but sample size is small.
7. **Whether LeFusion's separate DiffMask component needs to be incorporated.** The current pipeline uses only `emidec.pt`. LeFusion also ships `diffmask.pt` ("the mask generator") which may be necessary for genuinely new synthetic pathology geometry, rather than new image texture conditioned on a real mask. Not yet investigated.
8. **3D generation methods to evaluate**, per the latest meeting: NVIDIA's `NV-Generate-CTMR` (note: their MR model is image-only, no paired masks, and cardiac isn't in its listed supported body regions — CT models support pairs but not MR) and a more promising candidate found via literature search, `github.com/SoufianeBH/Paired-Image-Segmentation-Synthesis` (LGE-specific, joint image+mask synthesis, tested with 200 synthetic volumes matching the project's own target number). Neither has been tested yet.

## Environment setup

- Modal account, environment `main`, persistent volume `cardiac-data`
- Deployed Modal apps: `medgemma-emidec-poc`, `lefusion-emidec-poc`, `physician-tool` (extra volume `physician-tool-data`, secret `physician-tool-auth`)
- Gemini API: if using a KAUST Google account, you may hit an org policy blocking plain API keys (`API_KEY_SERVICE_BLOCKED` or requiring service-account binding) — using a personal Gmail account via [aistudio.google.com/apikey](https://aistudio.google.com/apikey) avoids this entirely
- EMIDEC dataset expected in the same folder structure as the official release: `Case_XXXX/Images/Case_XXXX.nii.gz` and `Case_XXXX/Contours/Case_XXXX.nii.gz`, plus clinical `.txt` files at the root

## Working principles established on this project

- **Verify before trusting.** Every script here was tested against synthetic data with a known correct answer before being run on real data. This caught real bugs (wrong label mapping, wrong slice auto-picking, an axis-order bug that made a rendered mask look like broken noise, a checker that mis-scored real model output due to markdown/formatting artifacts) before they wasted compute or led to wrong conclusions.
- **Don't accept results at face value.** Cross-check anything suspicious rather than smoothing it over.
- **State sample sizes and caveats honestly.** Findings from n=1 or n=3 are preliminary, not conclusive, however clean they look.
