# Evaluation declarations

Written before any test case is used. The evaluation script is `experiments/nnunet/evaluate.py`; its SHA-256 is printed on every run and recorded in `experiments/EXPERIMENT_LOG.md`.

- **H1:** arm 2 (real + 197 inserts) gives higher infarct Dice than arm 1 on the 13 pathological test cases. I expect the effect to be small. A null or negative result will be reported in the same way.
- **Primary:** arm 2 vs arm 1, infarct Dice, label-box crop. **Secondary:** 3 vs 1, 2 vs 3, centre crop, MVO / myocardium / LV Dice, false positives on normals.
- Both crops are run once for all 9 models. The script is not changed after the first test use. All results are reported, including every seed.
- **Limits:** the label-box crop is optimistic; the generator weights probably saw test patients; 13 test cases give wide intervals.
