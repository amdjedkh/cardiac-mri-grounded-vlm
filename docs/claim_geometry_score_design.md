# Per-claim geometry score: audit and design (not built)

## What the checks verify today
| Check (file) | Verifies | Does not |
|---|---|---|
| `check_region_grounding` (`prompts_region_grounded.py`) | Format only: every Findings sentence has a `[region: x]` tag, tag names are in the legend, no tags outside Findings, Impression not a copy of Findings. | Whether the sentence is true. |
| Location (`verification/check_location_claims_batch.py`) | The anterior / inferior / septal / lateral word(s) in the ONE sentence tagged `[region: infarct]` match the AHA segment of the infarct on the single slice shown. | Ring level (basal/mid/apical); wide lesions (it uses the arc MIDPOINT, so an 84%-of-ring infarct such as P019 becomes "mid inferolateral"); several sectors; MVO location; claims in other sentences; other slices. Needs a wall word, so only 39 of 100 Gemini reports make a checkable claim. |
| Extent / MVO position (`verification/verify_slice_claims.py`, `spot_check_reports.py`) | Computes ring coverage %, "near-circumferential" (>= 60%), MVO area % of infarct, MVO position along the arc (0 to 1), contiguity. | Compares nothing automatically: `spot_check_reports.py` prints the report next to these numbers for a human to read. No parsing of extent or MVO claims from text. |
| Negatives and numbers (`report_generation/compare_baselines.py`) | Explicit "no infarct" / "no MVO" statement when the volume has none; stated numbers within 5% of the patient JSON volumes. Written for the older measurement-field reports. | Slice-level presence; the region-grounded format; geometry. |

Missing for a per-claim score: claim extraction beyond one location sentence; claim types for extent, depth and MVO relation; multi-slice truth; a per-claim verdict and a per-report aggregate; "not checkable" handling.

## Design
Each Findings sentence already carries a region tag, so claims are parsed per tagged sentence by rules first (vocabulary lists, numbers with units), LLM parsing later if rules miss too much. The parser never sees the mask.

| Claim type | Example | Quantity from the mask | Right / wrong |
|---|---|---|---|
| Presence (infarct, MVO) | "No MVO is identified" | any label 3 / 4 pixel on the slice shown (and anywhere in the volume for volume-level claims) | right if stated presence equals mask presence |
| Location | "inferior wall", "mid anteroseptal" | per sector, share of lesion pixels on the slice, ring level from the slice position | right if every claimed sector holds >= 10% of the lesion; wrong if a claimed sector holds none; the claimed level must equal the slice's level when one is named |
| Extent | "near-circumferential", "focal", "3 segments" | ring coverage %; number of segments with >= 25% lesion | right if the claim's declared band contains the value (circumferential >= 60%, focal < 33%); numeric claims within 1 segment |
| Depth | "transmural", "subendocardial" | median lesion fraction of wall thickness over occupied 10-degree bins; lesion depth (0 endo to 1 epi) | right if inside the declared band (transmural >= 0.75, subendocardial: depth < 0.5 and fraction < 0.5) |
| MVO relation | "MVO within the infarct", "central" | share of the MVO border touching infarct; MVO position along the infarct arc | right if touch >= 0.5 for "within"; position in 0.35 to 0.65 for "central" |
| Size | "large infarct", "9.6 mL" | infarct area or volume; numbers vs the mask-derived volume | numbers within 5%; adjectives only against declared thresholds, else not checkable |

Bands and thresholds are declared before scoring, in one config, so they cannot be tuned on the results.

Output:
- per claim, one row: case, report id, sentence, claim type, parsed value, mask value, verdict (right / wrong / not checkable), margin to the nearest threshold;
- per report: claims, checkable, right, wrong, score = right / checkable, plus the share checkable (a report that says nothing scores nothing, not 100%);
- per run: mean score with a patient-level bootstrap interval, split by claim type.

Caveats to keep next to the numbers: the orientation (RV side = image right, anterior = top) is author-inspected and not cardiologist-confirmed; the ring level is a coarse thirds split of 4 to 10 slices; and any template that fills claims from the same mask functions (the rule-based baseline) matches by construction.
