# Segmentation comparison plan

Status: declared before any arm is trained (revised 2026-11-03: no validation fold, 197 inserts, fixed epoch budget). Only a one-run pipeline and cost check on arm 1 is planned; it is not a result and uses no test case.

## Question
Does adding the 197 accepted LeFusion inserts improve infarct segmentation on the fixed 20-case test set, compared with real data only?

## Declarations of 2026-11-03 (made before any arm run)
- **Epoch budget: 1000 epochs of 250 iterations for every arm.** Decided after the one-run trial (8.6 s per epoch on an L4, about 2.4 h and about $2 per run). Not revisited after seeing results.
- **Location metric for the report side: the two-sided check is primary** (right only if every claimed sector holds >= 10% of the infarct pixels and the claimed sectors together hold >= 80%; `experiments/location_check_independent.py`). The recall-only version (>= 50% of the infarct pixels in the claimed sectors) is secondary. The existing check (`check_location_claims_batch`) is kept only as a **legacy column**: it compares every named wall word with one arc-midpoint segment, so a correct multi-sector description scores wrong (rule-based v2: 9 of 66 on it vs 65 of 66 on the two-sided check). Every location number is reported next to its shuffled-mask chance level.
- These declarations do not change the primary segmentation metric below.

## Primary metric (declared now)
Infarct Dice on the 13 pathological test cases (all 13 have an infarct label), arm 2 (real + 197 inserts) minus arm 1 (real only), evaluated on the label-box crop.
- For each seed and each of the 13 cases, take the Dice difference arm 2 - arm 1 (same seed in both arms, so the two models are paired). Average over seeds per case.
- Patient-level bootstrap: resample the 13 patients with replacement 10,000 times, take the mean difference each time, report the 95% percentile interval. Also report every per-seed difference.
- The result is called an improvement only if the interval excludes 0. Everything else below is secondary and exploratory; no p-value or claim is based on it.

## Arms and priority
Run in this order; do not start the next group before the previous group's results are looked at.
1. Group 1: arm 1 (real only), arm 2 (real + 197 inserts), arm 3 (real + stronger classic augmentation).
2. Group 2: arm 5, control: real + a random 154 of the 197 inserts (seeded).
3. Group 3: arm 4: real + the 154 of the 197 inserts with contrast >= 0.099 (the weakest real train lesion).
Arm 5 is there so "brighter" can be separated from "fewer". Arm 4 vs arm 5 is the comparison that answers it.

## Fixed for every arm
- Same space: 72 x 72 x 10 volumes, per-volume min-max rescale to [-1, 1] (nnU-Net channel normalization set to none so it does not z-score), the same network (nnU-Net v2, 2D), the same epoch budget and optimizer (below). All volumes are written with the same nominal spacing (1.5625 x 1.5625 x 10 mm, as in LeFusion's files), so no arm gets resampled differently; native spacing differs by case (1.37 to 1.56 mm in plane).
- Same 20 test cases (`splits/split_v1.json`: 13 pathological, 7 normal), never used for training, for choosing a setting, or for stopping.
- Augmentation: every arm uses the same on-the-fly augmentation (nnU-Net's standard trainer augmentation plus the crop-centre jitter below). Arm 3 differs only by the stronger classic augmentation below, declared before any run and implemented in `nnUNetTrainerEmidecStrongAug` (`experiments/nnunet/nnUNetTrainerEmidec.py`). Arm 3 adds no extra volumes and trains for the same number of iterations. Arms 2, 4 and 5 use the standard augmentation only, not the stronger one.
- Mixing: arms 2, 4 and 5 draw 1/3 of every batch from the inserts and 2/3 from real training volumes. Arms 1 and 3 are 100% real. The epoch length (iterations per epoch) is the same in all arms, so the number of insert files does not change training length.
- Seeds: at least 3 per arm (the same seeds in every arm), reported as mean +- std.

## Training set, epoch budget and checkpoint (no validation fold)
- Training set: all 80 train patients (54 pathological, 26 normal) for every arm; arms 2, 4 and 5 add inserts as the 197 files marked `included` in `experiments/insert_list_v2.txt` (the 3 marked `excluded` fail the train-only reference rules; no file is deleted). Arm 4 uses the 154 of them with contrast >= 0.099; arm 5 a seeded random 154 of the 197. Every insert is built from train patients only (checked), so nothing needs to be held out for leakage.
- No validation fold: nothing is selected, tuned or early-stopped on held-out data, so a validation split is not needed, and the test set is not used for any choice either. Settings are fixed before the first arm.
- Epoch budget, declared now: 1000 epochs of 250 iterations per run (nnU-Net's default), the same for every arm, so the insert count does not change training length. The budget is changed only before any arm is launched, for all arms together, if the measured cost makes it unaffordable (the cost is measured and reported first, see Cost).
- Evaluation: the FINAL checkpoint (`checkpoint_final.pth`) only; no best-epoch selection.
- Consequence: without a held-out fold a diverged or broken run is only visible from the training loss and the training-set fit. A run is repeated only for a software failure (crash, NaN loss), never because its test result is disappointing, and any repeat is reported.

## Why one fixed split and not cross-validation
- The 200 inserts were generated from the train patients of the fixed split. A different outer split would need new inserts for each fold (about 1.5 h of A100 time and roughly $3 to $5 per set of 200), otherwise a fold's test patients would already be inside the inserts.
- LeFusion's pretrained weights probably saw the test patients (they were trained on EMIDEC without our split). So a held-out fold is clean only in the inputs we choose, not in the generator; extra folds would multiply the cost without making the estimate cleaner.
- The uncertainty is handled instead with repeated seeds and the patient-level bootstrap on the 13 pathological test patients. The price is a wide interval from a small test set, and results for one split only; this is stated wherever numbers are reported.

## Crop
The crop recipe is the bounding box of the ground-truth labels (labels > 0) plus 5 px in plane, all slices, then centre pad / crop to 72 x 72 x 10.
- Training, all arms: random crop-centre jitter (uniform in-plane shift of up to +-8 px, zero padded) so the models do not learn that the heart is always centred. Same jitter in every arm.
- Evaluation, same trained models, two crops: (a) the label-box crop (the primary metric uses this one; it uses the ground-truth contour to place the heart, so absolute Dice is optimistic and not deployment performance); (b) a fixed centre crop, a 72 x 72 window at the centre of the full slice, no label information. (b) is secondary. The comparison between arms is fair in both because all arms see the same crops.

## Secondary metrics (exploratory)
Dice for myocardium and MVO; infarct Dice with the centre crop; arm 3 vs arm 1; arm 4 vs arm 5; arm 4 and arm 5 vs arm 2; lesion-level detection (a predicted lesion pixel in a slice with a true lesion); volume error. Per-case values are reported next to every mean.

## nnU-Net setup notes
- Dataset 501, configuration 2d, channel name `noNorm` so [-1, 1] values are kept, labels 0 background, 1 LV cavity, 2 myocardium, 3 infarct, 4 MVO.
- Training on all 80 patients with `-f all` (no validation split).
- Custom trainer subclasses for the jitter, the 1/3 mixing and the stronger augmentation (arm 3); epoch length fixed.
- Reduced-epoch trainer (`nnUNetTrainer_*epochs`) only for the pipeline and cost check on arm 1, never for reported results. With `-f all` nnU-Net's own per-epoch and final "validation" is computed on the training cases, so it only shows that the model fits the data.

## Arm 3 stronger augmentation (declared 2026-11-03, before any run)
Everything else is nnU-Net's standard training pipeline plus the +-8 px crop-centre jitter used by all arms. Defaults in brackets.
| transform | stronger setting |
|---|---|
| rotation | p 0.5 [0.2]; range unchanged (isotropic patch, +-180 degrees) |
| scaling | p 0.5 [0.2]; (0.6, 1.6) [(0.7, 1.4)] |
| elastic deformation | p 0.3 [0]; scale (0.1, 0.25) of the edge length; magnitude (0, 5) px |
| Gaussian noise | p 0.3 [0.1]; variance (0, 0.15) [(0, 0.1)] |
| Gaussian blur | p 0.4 [0.2] |
| brightness | p 0.4 [0.15]; multiplier (0.6, 1.4) [(0.75, 1.25)] |
| contrast | p 0.4 [0.15]; (0.6, 1.4) [(0.75, 1.25)] |
| low resolution | p 0.4 [0.25] |
| gamma, inverted | p 0.25 [0.1]; (0.6, 1.7) [(0.7, 1.5)] |
| gamma | p 0.5 [0.3]; (0.6, 1.7) [(0.7, 1.5)] |
| mirror | unchanged |

## Implementation (built and tested locally on CPU, 2026-11-03)
- Trainers: `nnUNetTrainerEmidec` (arms 1, 2, 4, 5) and `nnUNetTrainerEmidecStrongAug` (arm 3), nnU-Net 2.8.1, 2d configuration, fold `all`, 1000 epochs x 250 iterations, batch 24, patch 64 x 64 (plans of Dataset501, transferred unchanged to Datasets 502, 504, 505, so every arm has the identical network).
- Seed: environment variable `EMIDEC_SEED`; seeds python, numpy, torch and the augmenter workers before the network is built. Not bit-exact (the multi-threaded augmenter delivers batches in non-deterministic order; GPU kernels are not forced deterministic), so two runs with the same seed can differ slightly; the seeds exist to vary initialisation and sampling in a recorded way.
- Mixing (arms 2, 4, 5): `EMIDEC_SYN_FRACTION` = 1/3; every batch holds exactly 8 insert samples and 16 real ones, in shuffled positions (nnU-Net forces foreground in the last third of a batch; the shuffle keeps inserts from filling those slots, tested: mean 2.6 to 2.8 of 8, 2.67 expected). Inserts are the cases whose id contains `__ins`. Real and insert volumes are drawn with replacement.
- Datasets: 501 = 80 real (arms 1, 3); 502 = 80 real + 197 inserts (arm 2); 504 = 80 real + the 154 bright inserts (arm 4); 505 = 80 real + a seeded random 154 of the 197 (arm 5; 121 of its 154 overlap with arm 4's). Insert volumes are LeFusion output images and masks (72 x 72 x 10, [-1, 1]). Lists: `experiments/nnunet/arm_lists.json`.
- Jitter: integer in-plane shift of up to +-8 px (image filled with 0, segmentation with nnU-Net's ignore value) applied to the initial patch before nnU-Net's spatial transform, same for all arms.
- Seeds for the comparison: 0, 1, 2 for every arm. Jobs for arms 1 to 3: 9 runs, one parallel GPU job each (L4).

## Cost
Measured on one reduced-epoch arm-1 run (see the experiment log). The full 1000-epoch cost per run is extrapolated from it and given before any arm is launched; arms are only launched after an explicit go.

## Needs your decision before group 1
Go for the 9 runs of arms 1 to 3 after the cost and job list are approved.
