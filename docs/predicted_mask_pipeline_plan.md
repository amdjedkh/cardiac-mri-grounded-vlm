# Predicted-mask report pipeline: plan (not built, nothing run)

Written 2026-11-03, before any arm is trained. Goal: replace the expert mask in the rule-based report with a mask predicted by the arm-1 model, and score the result with the two-sided location check.

## Pipeline
1. **Model.** The arm-1 `checkpoint_final.pth` (real data only, 1000 epochs), one per seed (0, 1, 2). Nothing is selected on the test set; if the arm 2 / arm 3 models exist later, the same steps run for them as a secondary analysis.
2. **Test inputs.** The 20 test cases (13 pathological, 7 normal) with the label-box crop (bounding box of the expert labels + 5 px, all slices, min-max to [-1, 1], centre pad to 72 x 72 x 10) written as nnU-Net test images. A second input set uses the fixed centre crop (no label information), as declared in the segmentation plan.
3. **Prediction.** `nnUNetv2_predict` (2d, fold all, trainer `nnUNetTrainerEmidec`, checkpoint_final) per seed, in the cropped space.
4. **Slice.** The slice Gemini was shown (the middle slice, index n // 2 of the original stack), mapped into the padded stack by the centre-pad offset.
5. **Report.** The rule-based template (`experiments/rule_based_report.py`, v1 and v2) filled from the PREDICTED mask of that slice: infarct present, sectors, wall fraction, MVO present and where. If the prediction has no cavity on that slice, no report is written and the case is listed as "not reportable" (counted, not dropped silently).
6. **Score.** Against the EXPERT mask of the same slice with the two-sided location check (primary), recall-only (secondary) and the legacy check (legacy column), each next to a shuffled-mask chance level (predicted masks of other cases, 1000 seeded permutations); presence of infarct and MVO against the expert mask; denominators always printed; per seed, then mean +- std over the 3 seeds. Also reported for reference on the same cases: rule-based from the EXPERT mask (the ceiling) and Gemini with the expert overlay.

## What is oracle and what is not
| step | oracle? |
|---|---|
| Label-box crop of the test volumes | **Oracle**: the box comes from the expert contour. The centre-crop run removes it. |
| Slice choice (middle slice) | Not oracle: fixed rule, the same one Gemini got. |
| Which cases are scored on location | **Oracle-defined**: a case counts only if the expert mask has infarct on that slice and the report names a wall (13 test cases for the expert-mask template). Cases where the predicted mask has infarct but the expert has none are reported separately as false claims. |
| Report text | Not oracle: made only from the predicted mask. |
| Scoring reference | The expert mask (this is the evaluation truth, as intended). |
| Gemini comparison reports | **Oracle input**: Gemini saw the expert overlay. A like-for-like comparison would re-render the overlays with the predicted mask and regenerate Gemini reports for the 20 test cases (cost about $0.01 to $0.05 at the earlier token estimate); proposed as a second step, not part of this one. |
| Sector orientation (image top = anterior, right = septal) | Assumed, not cardiologist-confirmed, in both the template and the checks. |
| Ring level (basal / mid / apical) | Coarse thirds of the stack; the full stack size is known at test time. |

## Declared in advance
- Primary result: the two-sided score of the rule-based v2 report built from the arm-1 predicted mask, on the 13 pathological test cases that have infarct on the shown slice in the expert mask, mean +- std over 3 seeds, next to the shuffled chance level and the expert-mask ceiling. Everything else is secondary.
- No threshold, template or sector rule is changed after seeing predicted-mask results.
- v2 names sectors from pixel shares, so with the expert mask it is a ceiling for the independent check (65/66); the predicted-mask score measures how much segmentation error costs, not the checker.

## Risks and limits
- 13 scored cases and 3 seeds: wide intervals; differences of a few cases are not evidence.
- Predicted infarct masks from 80 patients will often be fragmented or missing on normal-looking slices; false infarct claims on the 7 normal test cases are counted.
- If arm-1 Dice for infarct is low, the location score will follow it; the pipeline score should be read together with the segmentation Dice, not instead of it.
- No test case has been used by any model or choice so far; running this pipeline is the first use of the arm-1 models on the test set and is declared as part of the plan.

## Needs
Arm-1 final checkpoints (requires the go for arms 1 to 3), a test-set preparation script and a prediction job (about minutes on one GPU or on CPU), and an extension of `rule_based_report.py` to take a mask file instead of reading the expert mask.
