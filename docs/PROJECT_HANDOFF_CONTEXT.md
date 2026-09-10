# Cardiac MRI Grounded VLM Project — Full Handoff Context

This document is a complete handoff from a previous, very long conversation. Read this fully before responding to anything. The goal is for you to continue exactly as if you were the same assistant who did all of this work, not to start fresh.

## Who I am and the project

I'm a Master's student doing a research internship at KAUST. Supervisors: Dr. Karen Sanchez (direct supervisor), Dr. Carlos Hinojosa (co-supervisor), Prof. Bernard Ghanem (PI). Target venue: CVPR 2027, with MICCAI flagged as a realistic backup given the project's actual shape (small real dataset, expert-validation-heavy).

**Project goal:** a clinically grounded cardiac vision-language model for 3D cardiac MRI. Core idea: build a cardiac VLM whose reports are grounded in auditable quantitative measurements and cardiologist corrections, and investigate whether controlled synthetic 3D cardiac MRI can improve grounded report generation on unseen real patients.

**Three formal hypotheses from the team's research white paper:**
- H1 (Grounding): mask-derived quantitative measurements should improve clinical correctness and reduce unsupported statements vs. image-only generation.
- H2 (Expert supervision): cardiologist-corrected reports should provide better supervision than unreviewed LLM-generated reports.
- H3 (Synthetic utility): controlled synthetic 3D cardiac MRI should improve grounded report generation when appropriately filtered and mixed with real data.

**Pipeline concept:** Gold Standard (real EMIDEC + expert-validated reports, small, high quality) + Silver Standard (synthetic MRI + masks + synthetic reports, larger, no physician validation) → VLM fine-tuning → evaluated on an untouched real test set.

## Technical environment

- Windows machine, PowerShell. EMIDEC dataset folder: `D:\Downloads\emidec-dataset-1.0.1 (1)\emidec-dataset-1.0.1\`
- Python 3.13 locally (`C:\Python313\python.exe`)
- Modal account: `khamjed123`, environment `main`. Persistent volume: `cardiac-data`.
- Deployed Modal apps: `medgemma-emidec-poc` (MedGemma-4B-IT on A10G GPU), `lefusion-emidec-poc` (LeFusion synthetic generation on A10G, multiple functions)
- Gemini API: had to work around a KAUST Google account restriction (org policy blocking plain Gemini API keys); resolved using a personal Gmail account via AI Studio (aistudio.google.com/apikey)
- LeFusion repo: `github.com/HINTLab/LeFusion`. Pretrained weights + preprocessed EMIDEC data from HuggingFace (`YuheLiuu/LeFusion_Pretrained_model`, `YuheLiuu/LeFusion_Preprocessed_Data`)

## What has been built and completed, in order

### 1. EMIDEC data pipeline (foundational)
Scripts: `measurements.py`, `patient_schema.py`, `overlay_renderer.py`, `scan_all_cases.py`, `build_patient_jsons.py`. Two real bugs found and fixed early: EMIDEC label mapping needed verification against the official challenge repo (correct mapping: 0=background, 1=LV cavity, 2=myocardium, 3=infarct, 4=MVO), and clinical text files had a non-breaking-space encoding issue requiring cp1252 parsing instead of UTF-8. Eight cases selected for detailed work: N006 (normal), P055 (small infarct), P019 (large infarct + MVO), P060, P001, P002, P004, P007.

### 2. Original baseline POC (measurement-grounded, superseded)
`prompts.py`, `run_gemini.py`, `run_medgemma.py`, `modal_medgemma.py`, `compare_baselines.py`. Grounding tags pointed to measurement field names (e.g. `infarct_volume_ml`), not visual regions. Result: Gemini 8/8, MedGemma 1/8. **This approach was later corrected** — Carlos clarified that real grounding means linking text to an actual visual region, not a data field.

### 3. Corrected region-grounding (the real approach going forward)
`prompts_region_grounded.py` (tags in `[region: name]` format tied to actual segmentation regions), `run_region_grounded_poc.py`. Result: Gemini 8/8 fully and correctly tagged, verified not just by format but by checking claims against real geometry.

### 4. Quantitative verification tooling (used constantly throughout)
`verify_slice_claims.py`: `angular_coverage()`, `contiguity_score()`, `position_along_arc()`, polar diagram rendering. This caught two real hallucinations: P019's "central" MVO claim was actually off-center (0.81 on a 0=edge/0.5=center/1=edge scale), and P004's "anteroseptal" location claim was wrong (actual: mid anterior, confirmed by using the sternum as an anatomical landmark to verify image orientation). `assign_aha17_segment.py` computes AHA 17-segment assignment from mask geometry. `render_case_bullseye.py` renders a real, data-driven AHA17 bullseye plot colored by actual computed infarct/MVO involvement per segment (not a generic reference diagram).

### 5. MedGemma investigation (direct response to Karen's question)
Karen hypothesized MedGemma's poor performance might be a prompt/decoding issue, not a real capability gap. Tested properly across three configurations:
- Original prompt (`run_medgemma_region_grounded_poc.py`): 0/8 correctly grounded — tags leaked outside the Findings section, and in half the cases the Impression duplicated the Findings verbatim. (Required first fixing real bugs in the `check_region_grounding()` checker itself — it initially mis-scored things due to markdown formatting artifacts, then a numbered-list formatting artifact; both fixed and regression-tested against all known real outputs.)
- One worked example (`prompts_region_grounded_fewshot.py`): fixed the format completely (3/3), but MedGemma started copying the example's wording onto real cases regardless of actual content — including wrongly saying MVO was absent on a case where it's confirmed present.
- Two contrasting examples (`prompts_region_grounded_fewshot2.py`): made it worse, not better — all three test cases came back with nearly identical templated text, including the normal case falsely claiming an infarct was present.
- **Conclusion reported to Karen/Carlos:** formatting is fixable with prompting; actually reading and describing real image content is not reliable at 4B scale with prompting alone — likely needs fine-tuning, not just better prompts.

### 6. Code sent to supervisors
All relevant scripts had comments and docstrings stripped (via an AST-based Python script, not manual editing) before being sent to Karen over Discord, split into small labeled batches so she knows which file does what.

### 7. AHA17 label-consistency question answered
Carlos asked whether a separate label set is needed for AHA17. Proposal sent: no new label set — AHA17 gets computed deterministically on top of whatever mask already exists (real, synthetic, or model-predicted), using the same EMIDEC 4-class labels throughout, so label consistency across real/synthetic/predicted data is never broken.

### 8. LeFusion synthetic data pipeline (major infrastructure effort)
`modal_lefusion.py` deploys LeFusion (verified against the real HINTLab/LeFusion repo — exact inference command from `emidec_inference.sh`, exact pinned `requirements.txt`, not guessed). Real bugs found and fixed, in order, each confirmed via an actual failed run before being fixed (not preemptively guessed):
1. Modern pip (24.1+) rejects `pytorch-lightning==1.6.4`'s malformed version metadata — fixed by installing `pip==22.3.1` first, matching LeFusion's own README instruction I'd initially missed.
2. Two missing Python packages (`blobfile`, `rotary-embedding-torch`) — both were actually present in the real `requirements.txt`, dropped during my own transcription; fixed by installing the complete original list instead of a hand-picked subset.
3. LeFusion's pinned `typing_extensions==4.2.0` broke Modal's own SDK, which needs a modern version to run inside the same container — fixed by explicitly reinstalling a modern version after LeFusion's own packages.
4. Missing system library `libgl1`/`libglib2.0-0` needed by `opencv-python` in a headless container.
5. The preprocessed data `.tar` archives extract into a deeply nested path matching the original author's own directory structure, not a clean top-level folder as the README implied — fixed with a `glob` search + symlink to a clean path.
6. `modal run` creates an ephemeral app that gets torn down the moment the local script finishes, silently killing spawned background jobs (confirmed via a real run showing `Status=Cancelled`, 0 containers ever went live). Fixed by switching to `modal deploy` + a separate `trigger_lefusion.py` (uses `.spawn()` on the deployed, persistent app) + `check_lefusion_result.py` (polls a call ID from any session, resilient to the user's own connection dropping mid-run, which happened once due to travel wifi).
7. The original 30-minute function timeout was too short for a full ~50-case directory run (confirmed via a real `FunctionTimeoutError`) — bumped to 3 hours, and added a `max_cases` parameter for fast small-batch runs (also fixed to skip already-generated cases rather than wastefully regenerating them).
8. The synthetic mask array is stored with a different axis order than the image array (mask as `[D,H,W]`, image as `[H,W,D]`) — this caused the very first rendered mask overlay to look like a broken thin vertical strip instead of a proper ring shape. Fixed by detecting the axis permutation via matching shape dimensions and transposing.
9. The slice auto-picker originally chose whichever slice had the most *total* foreground (cavity+myocardium+infarct+MVO combined), which could skip past a slice that actually had pathology if another slice had more total anatomy — fixed to specifically target infarct/MVO presence.

**Working Modal functions in `modal_lefusion.py`:** `setup_data_and_weights`, `generate_synthetic_emidec` (main generation, supports `max_cases`), `analyze_synthetic_geometry` (single-slice contiguity check), `scan_all_slices_geometry` (whole-volume contiguity scan), `compare_synthetic_to_real_mask` (pixel-level identity check with axis-fix + resampling), `render_synthetic_case` (overlay PNG rendering).

**Local scripts:** `trigger_lefusion.py`, `check_lefusion_result.py`, `render_and_download.py`, `check_synthetic_geometry.py`, `check_all_slices.py`, `scan_real_case_all_slices.py` (real-case whole-volume scanner, mirrors the synthetic one), `compare_two_real_cases.py` (unrelated-real-patient baseline), `check_real_vs_synthetic_identity.py`.

**Generated so far:** synthetic cases `Case_P001`, `P002`, `P003`, `P004`, `P010`, `P022` (plus leftover files from an earlier killed full-directory run — the output folder likely contains ~50 total files, meaning the full 50-case preprocessed pathological dataset may already be fully covered).

### 9. Synthetic data quality investigation (rigorous, multi-stage)
- Built a "contiguity" metric (does a region form one smooth arc, or scattered fragments) and validated it against synthetic geometry with known correct answers before trusting it on real data.
- Real whole-volume baselines: P019=0.739, P004=0.730, P055=0.711, P060=0.591 (natural real-world range: 0.591–0.739).
- Synthetic P001's single auto-picked slice looked like a concerning outlier (0.495), but the whole-volume average was 0.741 — right in the real range. **This was a false alarm caused by checking only one unlucky slice out of ten**, resolved by building a whole-volume scanner.
- Investigated a second concerning signal: real P004 and "synthetic" P004 produced suspiciously matching contiguity numbers. Built a direct pixel-level comparison (with the axis-order fix and proper resampling to a common resolution, since the two are at different native resolutions). Result: **73.2% pixel match — not identical, so not a literal copy-through**, but also lower than a control comparison between two totally unrelated real patients (P004 vs P019), which matched at **97.6%**. That's backwards from what you'd expect if P004-synthetic is genuinely paired with P004-real.
- A visual side-by-side (real vs. synthetic overlay images) showed the synthetic image is tightly cropped to almost just the heart, while the real scan shows the whole chest — a genuine, structural field-of-view difference (LeFusion generates on a small fixed 72×72 canvas), not fake or copied data, but a real gap worth resolving before mixing into training.
- **The user then pointed out that for a second case (P001), the real and synthetic infarct patterns look genuinely different in shape, not just zoomed differently** — this reopened doubt about whether the filename-based "same patient" pairing assumption is actually correct. **This was never rigorously verified — it was assumed from LeFusion's output naming convention.** This is a live, unresolved uncertainty, not a resolved one.
- Given upload limits reached, the final Discord messages to Karen/Carlos were relabeled to explicitly **not claim same-patient pairing** ("not confirmed to be the same patient — filenames match by convention, but pairing has not been independently verified"), and the field-of-view mismatch was sent as an open question needing their input, not something explained away.

### 10. Other resolved items
- CURE gap analysis (from early in the project): CURE (the team's own prior chest X-ray grounded report generation paper) only grounds via 2D bounding boxes, never masks, and needs real radiology reports to mine phrase-to-region links from — which EMIDEC doesn't have. This is why the project can't just copy CURE's recipe directly.
- Exhaustive, repeated verification (three separate search passes) confirmed no public dataset anywhere pairs cardiac MRI segmentation with real grounded reports — this is a field-wide gap, not EMIDEC-specific.
- Karen's requested new evaluation metrics (BLEU/ROUGE/BERTScore for text, Dice/IoU/precision/accuracy for segmentation) were flagged back to her as needing clarification rather than silently building the wrong thing: text metrics need a reference report that doesn't exist yet (no real EMIDEC reports), and segmentation metrics need a predicted mask, but the current pipeline only uses ground-truth masks as input, never predicts one.

## Open items — genuinely unresolved, not just "next steps"

1. **Field-of-view mismatch between real and synthetic images** — sent to Karen/Carlos as an explicit open question (train on this as-is, or crop real images to match synthetic's tight framing?). Awaiting their reply.
2. **Same-patient identity pairing is unverified.** The 73% vs 97% pixel-match numbers are suspicious (paired should score higher than unrelated, not lower), and a second case (P001) showed genuinely different-looking patterns between real and "matching" synthetic. A proper controlled test would be: crop the real image to the heart-only region first (removing the field-of-view confound), then compare synthetic-P004 against real-P004 *and* against several other random real patients, to see if it actually matches its supposed source patient better than random. This has not been done yet.
3. **AHA17 base-vs-apex slice ordering** (which end of the slice stack is basal vs. apical) was assumed, never independently confirmed — unlike the top=anterior orientation, which *was* confirmed using the sternum as a landmark.
4. **Cardiologist annotation protocol** — not started at all.
5. **Karen's new evaluation metrics** — clarifying question sent, awaiting reply before building anything.
6. **Synthetic data quality checked on only a handful of cases so far** — want to grow the sample significantly before treating any conclusion (positive or negative) as solid.
7. **Whether LeFusion's separate DiffMask model needs to be incorporated.** LeFusion ships two components: the main model (`emidec.pt`, used so far) and `diffmask.pt`, described as "the mask generator." There's a real possibility the current pipeline only generates new *image* texture conditioned on the *real* mask geometry (not genuinely new pathology shapes), and that true new synthetic geometry requires running DiffMask too. This theory was raised but never investigated or confirmed.

## Working style and preferences established in this conversation — please follow these exactly

- **Verify before shipping, always.** Every script/function built in this project was tested against synthetic data with a known correct answer *before* being handed to the user to run for real. This caught many real bugs before wasting the user's time or Modal compute. Continue this discipline without exception.
- **Don't accept pasted results at face value.** Cross-check real outputs for consistency (this caught the axis-order bug, the wrong slice-picker, the P004 identity confusion, and more). If something looks suspicious, dig into it rather than smoothing it over.
- **Be honest about mistakes immediately and specifically**, without excessive self-flagellation. Several real mistakes happened in this conversation (broken `str_replace` edits that orphaned function bodies and silently created dead code, an overclaimed "identical data" finding that had to be walked back, describing a genuine field-of-view difference as mere "framing" when the user correctly pushed back that it was more significant than that). In each case, the right move was: acknowledge plainly, explain what actually happened, fix it, verify the fix, move on.
- **Never overclaim.** State sample sizes and caveats honestly. A finding based on n=1 or n=3 should be presented as preliminary, not conclusive, even if it points a particular direction.
- **Discord messages to supervisors** must be: short, plain/easy English, split into multiple separate short messages rather than one long block, no em-dashes or en-dashes anywhere, casual/lowercase tone that doesn't read as AI-generated, and framed humbly and tentatively ("i think", "i suggest", "wanted to check") since the user is being supervised, not leading the research — never overconfident phrasing like "confirmed" or "clearly."
- **Code sent to supervisors** should have all comments and docstrings stripped first (there's an AST-based stripping approach already used successfully for this).
- **The user often loses PowerShell context** (closes the terminal, forgets what was run). When asked for instructions again, always give the complete sequence from scratch, including `cd` into the EMIDEC folder, rather than assuming prior steps are remembered.
- **Explain technical findings in plain, simple language with concrete analogies** when the user seems confused — they are a capable student but not always fluent in the specific statistical/technical vocabulary being used, and appreciate a "here's what this actually means in plain terms" pass alongside the technical detail.
- **Be proactive**: build and fix things directly once direction is reasonably clear, rather than asking permission at every step — but flag real uncertainty or open questions honestly rather than plowing ahead on a shaky assumption.

## What to do first in this new conversation

Don't take any action yet. Just confirm you've read and understood this context, and ask what the user wants to work on next.
