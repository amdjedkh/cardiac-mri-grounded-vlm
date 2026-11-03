# Segmentation comparison plan

Status: declared before any arm is trained. Only a one-run pipeline and cost check on arm 1 (validation fold, no test cases) has been done.

## Question
Does adding the 200 LeFusion inserts improve infarct segmentation on the fixed 20-case test set, compared with real data only?

## Primary metric (declared now)
Infarct Dice on the 13 pathological test cases (all 13 have an infarct label), arm 2 (real + 200 inserts) minus arm 1 (real only), evaluated on the label-box crop.
- For each seed and each of the 13 cases, take the Dice difference arm 2 - arm 1 (same seed in both arms, so the two models are paired). Average over seeds per case.
- Patient-level bootstrap: resample the 13 patients with replacement 10,000 times, take the mean difference each time, report the 95% percentile interval. Also report every per-seed difference.
- The result is called an improvement only if the interval excludes 0. Everything else below is secondary and exploratory; no p-value or claim is based on it.

## Arms and priority
Run in this order; do not start the next group before the previous group's results are looked at.
1. Group 1: arm 1 (real only), arm 2 (real + 200 inserts), arm 3 (real + stronger classic augmentation).
2. Group 2: arm 5, control: real + a random 157 of the 200 inserts (seeded).
3. Group 3: arm 4: real + the 157 inserts with contrast >= 0.099 (the weakest real train lesion).
Arm 5 is there so "brighter" can be separated from "fewer". Arm 4 vs arm 5 is the comparison that answers it.

## Fixed for every arm
- Same space: 72 x 72 x 10 volumes, per-volume min-max rescale to [-1, 1] (nnU-Net channel normalization set to none so it does not z-score), the same network (nnU-Net v2, 2D), the same epochs, optimizer and validation rule. All volumes are written with the same nominal spacing (1.5625 x 1.5625 x 10 mm, as in LeFusion's files), so no arm gets resampled differently; native spacing differs by case (1.37 to 1.56 mm in plane).
- Same 20 test cases (`splits/split_v1.json`: 13 pathological, 7 normal), never used for training, validation or stopping.
- Augmentation: every arm uses the same on-the-fly augmentation (nnU-Net's standard trainer augmentation plus the crop-centre jitter below). Arm 3 differs only by adding stronger classic augmentation on top (wider rotation and scaling ranges, elastic deformation, stronger brightness / contrast / gamma and noise ranges; exact values fixed in a trainer subclass before the first run). Arm 3 adds no extra volumes and trains for the same number of iterations. Arms 2, 4 and 5 use the standard augmentation only, not the stronger one.
- Mixing: arms 2, 4 and 5 draw 1/3 of every batch from the inserts and 2/3 from real training volumes. Arms 1 and 3 are 100% real. The epoch length (iterations per epoch) is the same in all arms, so the number of insert files does not change training length.
- Seeds: at least 3 per arm (the same seeds in every arm), reported as mean +- std.

## Folds and the leakage rule
One fixed validation fold of the 80 train patients (patient level, stratified by pathological / normal, seeded), used only to watch training and pick nothing from the test set. An insert is excluded from training if its heart OR its lesion_from patient is in the validation fold. An insert uses two patients (the normal heart and the source of the lesion), so a fold can remove many inserts; the number removed is reported with each run. Without this rule a validation patient's heart or lesion shape would leak into training through the inserts.

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
- Custom `splits_final.json` with the patient-level fold above (nnU-Net's own random fold split is not used).
- Custom trainer subclasses for the jitter, the 1/3 mixing and the stronger augmentation (arm 3); epoch length fixed.
- Reduced-epoch trainer (`nnUNetTrainer_*epochs`) only for the pipeline and cost check, never for reported results.

## Cost
Measured on one reduced-epoch run (see the experiment log); the full-length cost per run is extrapolated from it and given before any arm is launched.

## Needs your decision before group 1
Number of seeds, whether the full 1000-epoch length is affordable, and the exact stronger-augmentation settings for arm 3.
