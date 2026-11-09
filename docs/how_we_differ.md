# How this project differs from related cardiac MRI VLM work

Status: draft of 2026-11-03. Everything marked **[paper]** is from the paper's abstract or arXiv page, read by me through a web search and page fetch; it is **from the paper, not independently verified**. I read abstracts and summary pages, not full papers or code. Two of the four names you gave could not be found (see the table); nothing is written for them from memory.

| | What it does [paper] | Data [paper] | Task [paper] | Code / weights [paper] |
|---|---|---|---|---|
| **CARA-VL** | Not found. A web search for "CARA-VL" returned no paper (only unrelated pages). | not found | not found | not found |
| **ORION-CMR** (arXiv 2609.23950, Demirel et al., Sep 2026) | Scanner-native end-to-end CMR foundation model; pipeline described as about 90 s. Reports are made by a "local large language model". | Pretrained on 12,896,733 CMR images from 9,258 studies; clinical validation on a multi-vendor cohort of 68 subjects (normal, congenital heart disease, dilated cardiomyopathy, myocardial infarction). Public benchmarks used but not named on the page. | Sequence classification, ventricular function, LGE detection, disease classification, report generation. Reported: normal vs abnormal AUC 0.96; multiclass AUC 0.88; 81.4% report agreement with expert interpretation. | Not stated on the arXiv page. |
| **CineMR** (arXiv 2610.01166, Li et al., Oct 2026) | Tool-augmented VLM: calls cardiac image-analysis tools (segmentation, phase selection, volumetry, morphometry, regional wall motion) and reasons over their output; trained with SFT then GRPO. | Multi-cohort VQA benchmark; sizes and cohorts not stated on the page I read. | Quantitative CMR assessment from cine: metric extraction, multiclass and differential diagnosis (a question-answering benchmark, not free-text report writing). pass@1 35.9% vs 1.5% for Qwen3-VL-8B and 7.0% for MedGemma-4B. | The abstract says code, benchmark and weights are available; I did not see the link. |
| **CLAIM** | Not found as a method. A search found only the "Checklist for Artificial Intelligence in Medical Imaging" (a reporting guideline) under that acronym. If you mean a specific paper, send me the title or link. | not found | not found | not found |

Related work the searches surfaced, not asked for, not read beyond the snippet (**[paper]**, unverified): CMR-CLIP (contrastive CMR-report model, code listed at github.com/Makiya11/CMRCLIP); CardioLens (multi-sequence CMR benchmark for multimodal LLMs, with finding-level report scoring against reference reports); a UK Biobank project plan for a CMR report model with a CMR-specific RadGraph for quality assurance. No search result described grounding generated sentences in segmentation masks.

## What is still new for us
This is a statement about this project relative to what I found, not an exhaustive literature review; check it before it goes in a paper or a talk.

- **Mask-grounded sentences.** Every Findings sentence carries a [region: ...] tag naming a region of the expert segmentation overlay, and the claims are checked against the expert mask geometry (AHA segment, extent, MVO position), not against another report.
- **Claim-level geometry check, planned.** `experiments/` and the design note describe a per-claim score derived from the mask. The papers above, as summarised, evaluate against expert-written reports or benchmark answers.
- **Public data with expert masks (EMIDEC, 100 cases).** Small, but every claim can be checked against a contour; the papers above use large or private cohorts and different supervision.
- **Mask-edited synthetic lesions** (LeFusion with our own mask edits, tested on a fixed patient-level split) as training data, with explicit limits documented in README.md.
- **A physician review tool** for validating images and reports.

What is **not** new or not shown by us: large-scale pretraining, scanner integration, tool-calling, or clinical validation. Our results so far are on 100 cases, one imaging task (infarct and MVO on late enhancement), and the orientation convention is still not confirmed by a cardiologist.
