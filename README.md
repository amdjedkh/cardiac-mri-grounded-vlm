# Cardiac MRI Grounded VLM

Research project at KAUST building a clinically grounded vision-language model for 3D cardiac MRI. Target venue: CVPR 2027 (MICCAI as a realistic backup). Supervisors: Dr. Karen Sanchez, Dr. Carlos Hinojosa, PI Prof. Bernard Ghanem.

**Core idea:** a cardiac VLM whose reports are grounded in auditable quantitative measurements and cardiologist corrections. Investigating whether controlled synthetic 3D cardiac MRI can improve grounded report generation on unseen real patients.

**Three hypotheses:**
- **H1 (Grounding):** mask-derived quantitative measurements should improve clinical correctness and reduce unsupported statements vs. image-only generation.
- **H2 (Expert supervision):** cardiologist-corrected reports should provide better supervision than unreviewed LLM-generated reports.
- **H3 (Synthetic utility):** controlled synthetic 3D cardiac MRI should improve grounded report generation when appropriately filtered and mixed with real data.

## Current direction (as of the latest supervisor meeting)

The project is moving from a 2D proof-of-concept toward a proper 3D pipeline. Three parallel workstreams:
1. **3D synthetic generation** — investigate methods that generate a full 3D volume with anatomically consistent slices, not independent 2D slices. Target: ~200 synthetic 3D cases for physician validation.
2. **Physician evaluation tool** — a lightweight web tool (not CVAT-scale) for a cardiologist to review synthetic MRI/masks and generated reports, and produce a clean "accepted" dataset.
3. **Report generation at scale** — generate Gemini reports for all real EMIDEC cases (currently blocked on confirming Gemini API credits with Carlos).

## Repo structure

```
data_pipeline/         EMIDEC loading, measurement extraction, overlay rendering
report_generation/     Prompt scaffolds + runners for Gemini/MedGemma report generation
verification/          Quantitative geometry checks (angular coverage, contiguity, AHA17)
synthetic_generation/  LeFusion synthetic MRI+mask generation on Modal
physician_tool/        Flask web app for physician review (Module 1: image/mask validation)
docs/                  Additional project context
```

### `data_pipeline/`
- `measurements.py` — extracts infarct volume, infarct %, MVO volume, affected slices from a mask
- `patient_schema.py` — parses EMIDEC clinical `.txt` files (note: must be read as `cp1252`, not UTF-8 — EMIDEC uses non-breaking spaces before colons, a French typographic convention, which silently breaks UTF-8 parsing)
- `overlay_renderer.py` — renders MRI + mask overlay PNGs (red=infarct, yellow=MVO)
- `scan_all_cases.py` — scans all 100 EMIDEC cases, builds a summary CSV
- `build_patient_jsons.py` — builds per-patient JSON records with measurements + clinical metadata
- `verify_labels.py` — confirms the EMIDEC label mapping against the official challenge repo
- `debug_sex_field.py` — diagnostic script for the clinical file encoding issue

**Verified EMIDEC label mapping:** `0=background, 1=LV cavity, 2=myocardium, 3=infarct, 4=MVO/no-reflow`

### `report_generation/`
Two generations of approach here, both present for reference:
- `prompts.py` — **superseded.** Original approach, grounding tags pointed to measurement field names (e.g. `infarct_volume_ml`), not visual regions. Carlos clarified this isn't real grounding.
- `prompts_region_grounded.py` — **current approach.** Tags in `[region: name]` format tied to actual segmentation regions. Verified against real geometry, not just format.
- `prompts_region_grounded_fewshot.py` / `_fewshot2.py` — experiments testing whether MedGemma's poor performance was a prompting issue (Karen's hypothesis). One example fixed formatting but caused content-copying; two examples made it worse. Conclusion: prompting fixes format, not reliable content understanding, at 4B scale.
- `run_*.py` — runners for each experiment (Gemini and MedGemma, original and region-grounded, fewshot variants)
- `modal_medgemma.py` — Modal deployment for MedGemma-4B-IT (A10G GPU)
- `compare_baselines.py`, `recheck_region_grounding.py`, `recheck_any_folder.py` — automated checkers for grounding format compliance. **Note:** the checker logic went through several real bug fixes (markdown formatting artifacts, then numbered-list formatting artifacts breaking the sentence-counting heuristic) — see `recheck_any_folder.py` for the current, most robust version.

### `verification/`
Quantitative tools to check report claims against actual mask geometry, rather than trusting model output or eyeballing images.
- `verify_slice_claims.py` — `angular_coverage()` (what % of the ring a region covers), `contiguity_score()` (smooth solid arc vs. scattered fragments), `position_along_arc()` (is a region centered or off to one edge)
- `assign_aha17_segment.py` — computes AHA 17-segment assignment from mask geometry. **Orientation note:** top-of-image = anterior wall was confirmed using the sternum as a landmark. Which end of the slice stack is basal vs. apical was assumed, never independently confirmed.
- `render_case_bullseye.py` — renders a real, data-driven AHA17 bullseye plot colored by actual computed infarct/MVO involvement per segment

This tooling has already caught real issues worth knowing about: a report claiming MVO was "central" when it was actually near one edge (0.81 on a 0=edge/0.5=center/1=edge scale), and a claimed "anteroseptal" location that was actually "mid anterior."

### `synthetic_generation/`
LeFusion (`github.com/HINTLab/LeFusion`) deployed on Modal to generate synthetic EMIDEC-style cardiac MRI + masks from pretrained weights.
- `modal_lefusion.py` — the Modal app. Functions: `setup_data_and_weights`, `generate_synthetic_emidec` (supports `max_cases` for fast small batches), `analyze_synthetic_geometry`, `scan_all_slices_geometry`, `compare_synthetic_to_real_mask`, `render_synthetic_case`
- `trigger_lefusion.py` — triggers generation on the **deployed** app (not `modal run`, see note below)
- `check_lefusion_result.py` — polls a spawned job by call ID, works from any session
- `render_and_download.py`, `check_synthetic_geometry.py`, `check_all_slices.py` — inspection/visualization tools for synthetic cases
- `scan_real_case_all_slices.py`, `compare_two_real_cases.py` — matching tools for real cases, used to build fair real-vs-synthetic comparisons
- `check_real_vs_synthetic_identity.py` — checks whether a "synthetic" case is genuinely new content or effectively a copy of its real conditioning case (axis-order-corrected, resolution-resampled pixel comparison)

**Important workflow note:** use `modal deploy modal_lefusion.py` then `python trigger_lefusion.py`, never `modal run`. `modal run` creates a temporary app that gets torn down the moment the local script finishes, which silently kills any spawned background job — confirmed via a real run that showed `Status=Cancelled` with zero containers ever going live.

**Known real bugs already fixed in this pipeline** (worth knowing before assuming something new is broken): pip version conflicts with LeFusion's pinned dependencies, two packages missing from a hand-transcribed requirements list (fixed by installing LeFusion's complete original list instead), a conflict where LeFusion's pinned `typing_extensions` broke Modal's own SDK, a missing system graphics library for `opencv-python`, a nested-path issue in the preprocessed data archive, and — most subtly — the synthetic mask array is stored with a different axis order than the image array (`[D,H,W]` vs `[H,W,D]`), which is corrected in every function that touches mask data.

### `physician_tool/`
Flask web app for physician review. **Module 1 only (image/mask validation) is built.** Module 2 (report review) is not yet built — it's blocked on having real generated reports to review, which depends on the report-generation-at-scale workstream.

Run with:
```bash
pip install flask nibabel numpy matplotlib scipy
export REAL_CASES_DIR=/path/to/emidec/cases
export SYNTHETIC_CASES_DIR=/path/to/downloaded/synthetic/cases
python physician_tool/physician_tool.py
```
Then open `http://localhost:5050`.

Workflow: physician reviews one case at a time (real or synthetic, mixed in one list), scrolls through slices, toggles the mask overlay on/off, optionally marks a specific region with a bounding box, accepts or rejects, adds a comment, and moves to the next case. Revisiting an already-reviewed case pre-fills the previous decision so it can be changed. A summary bar shows live counts, and an "Export accepted dataset" button produces a JSON manifest of every accepted case with its file paths — this is the actual clean input list for building the final training set. Keyboard shortcuts: `A` accept, `R` reject, arrow keys for slice/case navigation, `Enter` to submit.

Synthetic cases must be downloaded locally first, since they live on the Modal volume:
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
- Deployed Modal apps: `medgemma-emidec-poc`, `lefusion-emidec-poc`
- Gemini API: if using a KAUST Google account, you may hit an org policy blocking plain API keys (`API_KEY_SERVICE_BLOCKED` or requiring service-account binding) — using a personal Gmail account via [aistudio.google.com/apikey](https://aistudio.google.com/apikey) avoids this entirely
- EMIDEC dataset expected in the same folder structure as the official release: `Case_XXXX/Images/Case_XXXX.nii.gz` and `Case_XXXX/Contours/Case_XXXX.nii.gz`, plus clinical `.txt` files at the root

## Working principles established on this project

- **Verify before trusting.** Every script here was tested against synthetic data with a known correct answer before being run on real data. This caught real bugs (wrong label mapping, wrong slice auto-picking, an axis-order bug that made a rendered mask look like broken noise, a checker that mis-scored real model output due to markdown/formatting artifacts) before they wasted compute or led to wrong conclusions.
- **Don't accept results at face value.** Cross-check anything suspicious rather than smoothing it over.
- **State sample sizes and caveats honestly.** Findings from n=1 or n=3 are preliminary, not conclusive, however clean they look.
