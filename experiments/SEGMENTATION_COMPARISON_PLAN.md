# Segmentation comparison plan

Status: declared before any arm is trained (revised 2026-11-03: no validation fold, 197 inserts, fixed epoch budget). Only a one-run pipeline and cost check on arm 1 is planned; it is not a result and uses no test case.

## Question
Does adding the 197 accepted LeFusion inserts improve infarct segmentation on the fixed 20-case test set, compared with real data only?

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
- Augmentation: every arm uses the same on-the-fly augmentation (nnU-Net's standard trainer augmentation plus the crop-centre jitter below). Arm 3 differs only by adding stronger classic augmentation on top (wider rotation and scaling ranges, elastic deformation, stronger brightness / contrast / gamma and noise ranges; exact values fixed in a trainer subclass before the first run). Arm 3 adds no extra volumes and trains for the same number of iterations. Arms 2, 4 and 5 use the standard augmentation only, not the stronger one.
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

## Cost
Measured on one reduced-epoch arm-1 run (see the experiment log). The full 1000-epoch cost per run is extrapolated from it and given before any arm is launched; arms are only launched after an explicit go.

## Needs your decision before group 1
Number of seeds, whether the 1000-epoch budget is affordable at the measured cost, and the exact stronger-augmentation settings for arm 3.
