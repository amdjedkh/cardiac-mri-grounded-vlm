# Experiment log

## 2026-11-02: LeFusion lesion inserts into train normal hearts (insert run v1)

### Hypothesis
Real EMIDEC lesion shapes are few (31 distinct shapes among the 54 train pathological cases at footprint IoU 0.6).
Putting real train lesion masks into train NORMAL hearts, with a random angle / width / depth / slice shift, and
letting LeFusion paint the lesion texture, adds new lesion shapes and new heart + lesion combinations. A segmentation
model trained on real + these volumes should then do better on the fixed 20-case test set than one trained on real
data alone. Not tested yet; this entry covers generation and image checks only.

### Settings
- Generator: LeFusion (HINTLab) through `synthetic_generation/modal_lefusion.py::generate_staged`, A100, jump_length 2, jump_n_sample 2 (896-step loop), batch 1.
- No random seed: LeFusion's inference has no seed option, so a rerun gives different pixels. The masks are reproducible
  (builder seed 42, rebuilt byte for byte from the volume inputs, 400 of 400 files identical); the generated images are not.
- Inputs: `synthetic_generation/build_staged_test.py --mode full --per-heart 8`. Crop = bounding box of labels > 0 + 5 px in-plane, all slices; LeFusion pads to 72 x 72 x 10 and rescales to [-1, 1].
- Library: the 54 train pathological lesions. Targets: train normal hearts only. No test-split patient anywhere (checked on the staged lists and on the outputs with `splits/check_split_leakage.py`).
- n = 200 volumes, 25 normal hearts x 8 lesions. Case_N016 excluded: 4 slices, no library lesion could be placed (26 train normals, 25 used). 43 of the 54 library lesions were used, each 1 to 9 times.
- Run as 3 jobs (67 / 67 / 66 volumes). Backup: `~/backups/full_insert_v1/` with SHA256 of every file in `~/private_notes/synthetic_file_hashes.sha256` (800 .nii.gz + MANIFEST.json + staged_files.txt).
- Checks: `experiments/validate_full.py` (200 volumes, 0 problems: output image finite, output mask equals the padded input mask, no test patient); `experiments/contrast_report.py`; `experiments/montage.py`.

### Results (contrast = mean lesion - mean healthy wall, [-1, 1] scale)
| set | n | mean | median | min | max |
|---|---|---|---|---|---|
| 200 inserts, infarct + MVO | 200 | +0.146 | +0.139 | +0.030 | +0.275 |
| real train pathological | 54 | +0.231 | +0.220 | +0.099 | +0.556 |
| inserts, infarct voxels only | 200 | +0.149 | +0.146 | +0.030 | +0.278 |
| real, infarct voxels only | 54 | +0.247 | +0.237 | +0.099 | +0.583 |
| inserts, MVO voxels only | 129 | +0.121 | +0.120 | -0.029 | +0.293 |
| real, MVO voxels only | 32 | +0.106 | +0.116 | -0.117 | +0.396 |

- Insert contrast is about 63% of real (0.146 / 0.231). The shortfall is in infarct (0.149 vs 0.247); MVO is close to real.
- 157 of 200 are inside the real range (>= 0.099); 43 are below it; 8 are below 0.05. Nothing was removed.
- By size tertile the inserts get brighter with size (0.126 / 0.152 / 0.160); the real cases go the other way (0.251 / 0.223 / 0.219).
- Mean absolute change outside the lesion, per volume: 0.024 (range 0.020 to 0.028), so LeFusion leaves the rest of the image close to the input but not identical.
- Montages (random 12, faintest 8, brightest 8): the faintest are mostly thin infarct-only lesions, the brightest mostly infarct + MVO.
- jump 5/5 was 2.6x slower with no measurable contrast gain on the 6-volume test.

### Cost and time
- Container time from file times on the volume (CET): job 1 22:02 to 22:34 (33 min), job 2 22:35 to 23:02 (28 min), job 3 rerun 10:14 to 10:41 on 2 Nov (28 min). Sum 89 min.
- Job 2 took 88 min on the local log but its outputs were all written within 28 min; the rest was the local client waiting. The first attempt at job 3 was cancelled after 6 of 66 volumes and the rerun overwrote them (all 66 files carry rerun times). Cause not proven: the pattern fits the laptop sleeping and losing its connection to Modal.
- Estimate at about $2.10/h for an A100-40GB (rate from memory, not checked): about $3.1 for the three useful runs plus $0.2 to $2.1 for the cancelled attempt, so about $3.3 to $5.2. The exact figure is on the Modal billing page.
- Modal billing total = [I will fill in]

### Limitations
- Contrast is about 63% of real, so the lesions are fainter than real ones and the size trend is reversed.
- LeFusion's pretrained weights probably saw the test patients (they were trained on EMIDEC without our split). The inputs are test-free, but the texture model may not be.
- Only 25 hearts and 43 library lesions; the combination count is large but the underlying anatomy and lesion shapes are few.
- Lesions are inserted into normal hearts, so the images contain no real scar tissue characteristics beyond what LeFusion learned.
- No seed: the generated images cannot be regenerated exactly. The backup is the only copy of this set.
- The crop uses the ground-truth label box (see the plan below).

## 2026-11-03: reference recomputed from train cases only; insert list v2
- The accept rules of `mask_diversity.evaluate()` compare each insert with a "real" reference (`synthetic_generation/mask_diversity_reference.json`). The reference used for the insert run was computed from every `Case_P*` folder (67 cases), so it **included the test patients**; the contiguity maximum (0.934) and the MVO-touch minimum (0.968) came partly from them.
- It is now computed from the 54 train pathological cases only (`compute_reference` reads `splits/split_v1.json`): contiguity 0.3665 to 0.8882 (was 0.3665 to 0.9343), MVO touch minimum 0.9735 over 32 cases (was 0.9683 over 40). A 72 x 72 x 10 crop does not change the score (difference 0 on all 54 train cases), so the full-size reference is valid at the insert resolution.
- Re-applying the reference-dependent rules to the 200 inserts (`experiments/reapply_reference.py`): 0 rejected under the old reference, 3 under the new one, all on MVO touch: Case_N030__insP02 (0.971), insP03 (0.970), insP04 (0.969), against a minimum of 0.974. Contiguity passes for all 200.
- `experiments/insert_list_v2.txt` marks those 3 as excluded and the other 197 as included. No file was deleted. The segmentation comparison uses the 197.
- The old reference file is kept at `~/private_notes/mask_diversity_reference_67cases_old.json` (it is also in git history).
- The other accept rules (area ratio, lesion leaving the wall, lesion lost) do not use the reference and were not repeated.

## 2026-11-03: rule-based report baseline and the circularity of its location score
- Baseline: `experiments/rule_based_report.py`, a fixed template filled from the expert mask of the slice shown to Gemini, 80 train + 20 test cases, no AI, no training.
- **Its score on the existing location check is circular.** The wall location is computed by `assign_segment()`, the function the check compares against, so the match is 66 of 66 by construction. It is a ceiling for that check, not accuracy.
- Independent check (`experiments/location_check_independent.py`, separate code, pixel-wise share of infarct pixels inside the claimed sectors, right if >= 0.5, declared before scoring): rule-based 41 of 66 right (62%, mean share 0.63); Gemini 20 of 39 (51%, mean 0.54). On the 39 cases where Gemini names a wall: rule-based 26 of 39 (mean share 0.67), Gemini 20 of 39 (0.54). The rule-based text uses the arc-midpoint segment, which misses much of a wide lesion.
- Other existing checks (format, negative statements, presence on the slice) are 100% for both and do not separate them. Gemini names a checkable wall in only 39 of 100 reports.
- Caveats: the sector orientation (image top = anterior, right = septal) is assumed in both checks; ring level is not checked; test cases use the older 100-report folder.

## 2026-11-03: two-sided location check, shuffle baseline, rule-based v2, new Gemini reports
Scores are in `~/backups/full_insert_v1/reports/location_scores_v2.csv` (`experiments/score_reports.py`). Denominators are "right / checkable"; a report is checkable only if it names a wall in the [region: infarct] sentence AND the slice shown has infarct pixels.

Checks (thresholds declared before scoring, none tuned):
- **existing**: wall words vs the arc-midpoint AHA segment (`check_location_claims_batch` logic).
- **recall-only** (independent, `location_check_independent.py`): share of infarct pixels in the claimed sectors (lenient word map) >= 0.5.
- **two-sided** (independent, from `docs/claim_geometry_score_design.md`): strict word map; right only if EVERY claimed sector holds >= 10% of the infarct pixels AND the claimed sectors together hold >= 80%.
- **shuffle**: each report paired with another case's mask slice, 1000 seeded permutations without fixed points; chance = mean of right / checkable, 5th to 95th percentile in brackets.

| source (100 cases) | existing | recall-only | two-sided |
|---|---|---|---|
| rule-based v1 | 66/66 = 1.00 (chance .27) | 41/66 = .62 (.26) | 4/66 = .06 (.02) |
| rule-based v2 (sectors >= 15%) | 9/66 = .14 (.05) | 66/66 = 1.00 (.55) | 65/66 = .98 (.09) |
| Gemini old100 (September folder) | 25/50 = .50 (.29) | 28/50 = .56 (.33) | 9/50 = .18 (.07) |
| Gemini new100 (pinned config, 2 Nov) | 24/37 = .65 (.33) | 21/37 = .57 (.30) | 8/37 = .22 (.07) |

- The 20 test cases separately (right / checkable; chance in brackets), existing | recall-only | two-sided: rule-based v1 13/13 | 9/13 | 3/13; rule-based v2 4/13 | 13/13 | 13/13; Gemini old 5/10 | 7/10 | 2/10; Gemini new 7/8 | 7/8 | 4/8. Chance levels on 20 cases are wide (for example 0 to 0.67); these are small counts, not evidence of a difference.
- Rule-based v1 on the existing check (66/66) is circular. Rule-based v2 on the independent checks is also close to a ceiling: v2 names the sectors from the same pixel shares the independent check uses, so 65/66 is not independent evidence. v2 on the existing check (9/66) shows a defect of that check: it requires every named word to sit inside the single arc-midpoint segment, so a complete multi-sector description scores wrong.
- Gemini beats shuffled chance on every check (for example two-sided .22 vs .07), but the two-sided score is low: most Gemini location claims name one wall for a lesion that spans several sectors. Wilson 95% intervals are wide (24/37: .49 to .78; 25/50: .37 to .63), so new vs old is not a demonstrated difference; the new folder has fewer checkable reports (37 vs 50).
- Other existing checks (format, negative statements, presence) are 100% or near it for all four sources and do not separate them.
- **Non-determinism**: with the same pinned settings (vertex, gemini-3.1-flash-lite, temperature 0, seed 42), 71 of the 80 train reports of the new run are identical to the 80-case run (`run_train_v1`); 9 differ. The old September folder matches the new one on 0 of the 20 test cases (earlier settings, no record of them).
- New reports: `EMIDEC/run_all100_pinned/outputs_region_grounded/` (100 records, each with `generation_settings`: vertex, gemini-3.1-flash-lite, 0.0, 42; none off-config). Old folders untouched. Script: `experiments/run_all100_pinned.py`.

## 2026-11-03: nnU-Net 2D trial, arm 1 (one run, 20 epochs, L4)
- Setup: Dataset501_EMIDECArm1 (80 train volumes, 72 x 72 x 10, [-1, 1], no normalization by nnU-Net), nnU-Net 2.8.1, torch 2.15.0, trainer `nnUNetTrainer_20epochs`, fold `all` (no validation split), L4 GPU, 8 CPUs. No test case, no insert.
- Epoch time: first epoch 98.0 s (warm-up), then 8.60 s on average (8.25 to 9.51 s) over the next 19. The 20 epochs took 261.5 s; the training command took 360.3 s in all (about 99 s fixed: start-up, planning, final pass over the 80 training volumes). Preprocessing took 19.1 s on CPU. Wall time of the GPU step as seen from here was about 19 minutes, mostly waiting and container start, which I could not separate.
- Extrapolation (98 s first epoch + 8.60 s per further epoch + 99 s fixed): 250 epochs about 2,340 s (39 min); 1000 epochs about 8,790 s (2.4 h). At about $0.80/h for an L4 (rate from memory, not checked) that is about $0.5 and $2.0 per run; 9 runs (arms 1 to 3, 3 seeds) about $4.7 at 250 epochs and $17.6 at 1000. Arms 2 to 5 may be slightly slower with the custom insert-mixing trainer, which is not built yet. Exact cost: Modal billing page.
- The "validation" Dice from `-f all` is on the TRAINING cases (mean foreground Dice 0.819; LV 0.964, myocardium 0.891, infarct 0.643, MVO 0.778 after 20 epochs). It only shows that the pipeline learns; it is not a performance number and no test case was used.
- A first launch failed at container start (config.py was not shipped into the image); fixed with `add_local_file`; no compute was billed for the failed start. Two launch chains briefly ran at once; both were stopped before any container ran, and one chain produced the single reported run.

## 2026-11-03: declarations (before any arm run)
- 1000 epochs for every arm (decided from the trial: 8.6 s/epoch, about 2.4 h and about $2 per run on an L4; not revisited after results).
- Report-side location scoring: two-sided check primary, recall-only secondary, the existing check a legacy column. Its defect is documented in README.md: it requires every wall word to fall inside the single arc-midpoint segment, so complete multi-sector descriptions score wrong (rule-based v2 9/66 vs 65/66 two-sided).
- Plan: `experiments/SEGMENTATION_COMPARISON_PLAN.md` (declarations section).

## 2026-11-04: launch record, arms 1-3 x seeds 0,1,2 (1000 epochs, one L4 job each)
Authorised by the user ("Go for the 9 runs"). Recorded BEFORE launch.
- Code: branch `launch-arms-1-3`, launcher commit `cbb81ecf87815bfe5a25567bef29b5147fb760a1` (parent `926eacf`, tip of the PR stack #11-#29).
- Software on Modal: nnunetv2 2.8.1 (pip pin), torch resolved by pip at image build (local venv: torch 2.15.0; the Modal container's torch version is printed in each job log and is recorded below when read), python 3.11, L4, 8 CPU, 16 GB.
- Datasets (sha256 of the sorted per-file sha256 list, first 16 hex; uploaded to volume `cardiac-data` under `nnunet/nnUNet_preprocessed/`): Dataset501 raw 3c2ba7f5ef0c8a42 (161 files), preprocessed a5981e11a416f50c (323 files); Dataset502 raw 561365462b21d2d1 (555 files), preprocessed b608c36bbd4a928e (1111 files). Preprocessed 501 on the volume was replaced by the local copy.
- Seeds: 0, 1, 2 (env EMIDEC_SEED), same in every arm.
- Jobs and exact command per job (`nnUNetv2_train <dataset> 2d all -tr <trainer> -device cuda --c`, env EMIDEC_SEED, EMIDEC_SYN_FRACTION, nnUNet_results=/cardiac-data/nnunet/results/arm{A}_seed{S}):
  arm 1: 501, nnUNetTrainerEmidec, syn 0; arm 2: 502, nnUNetTrainerEmidec, syn 0.3333; arm 3: 501, nnUNetTrainerEmidecStrongAug, syn 0; each for seeds 0, 1, 2.
- Launcher: `modal run --detach experiments/nnunet/modal_nnunet.py::run_arms --arms 1,2,3 --seeds 0,1,2` (spawns 9 calls). Changes vs PR #28: spawn instead of starmap, `--c` added, volume commit every 120 s so logs are readable, job raises on non-zero exit or missing checkpoint_final.pth (retries=1).
