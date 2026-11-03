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
