"""
check_location_claims_batch.py

Extracts the wall-location claim from every generated report (e.g.
"inferior wall", "anterior-septal wall") and checks it against the REAL
AHA17 segment computed from that case's actual mask geometry -- instead of
spot-checking one case at a time.

Reuses assign_aha17_segment.py's tested assign_segment() function directly,
rather than re-implementing the angle math.

IMPORTANT CAVEAT (inherited from assign_aha17_segment.py, read before
trusting results): this assumes image-top = anterior (confirmed earlier via
the sternum landmark) AND slice 0 = base (NOT independently confirmed). A
wrong ring-level (basal/mid/apical) doesn't affect the anterior/inferior/
septal/lateral part of the match, only whether "basal" vs "mid" vs "apical"
is right -- this checker only compares the anterior/inferior/septal/lateral
wall word, not the ring level, for exactly that reason.

Usage (run from your EMIDEC folder):
    python3 check_location_claims_batch.py
"""

import os
import re
import sys
import json
import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
try:
    from assign_aha17_segment import assign_segment
except ImportError:
    print("Couldn't import assign_aha17_segment.py -- make sure it's in the same")
    print("folder as this script (or adjust sys.path above).")
    sys.exit(1)

# maps words that show up in report text to the wall-direction vocabulary
# assign_segment() actually uses -- deliberately only checks the
# anterior/inferior/septal/lateral axis, not basal/mid/apical (see caveat above)
WALL_KEYWORDS = {
    "anteroseptal": "anteroseptal", "anterior-septal": "anteroseptal",
    "inferoseptal": "inferoseptal", "infero-septal": "inferoseptal",
    "inferolateral": "inferolateral", "infero-lateral": "inferolateral",
    "anterolateral": "anterolateral", "antero-lateral": "anterolateral",
    "anterior": "anterior", "inferior": "inferior",
    "lateral": "lateral", "septal": "septal",
}


def extract_infarct_sentence(report_text: str) -> str:
    """Isolates just the ONE sentence attached to [region: infarct] -- the
    actual location claim -- rather than the whole report. Confirmed
    necessary after Case_P004 showed 'anteroseptal + anterior + septal' as
    three separate claims, when in reality it was the SAME single location
    mentioned twice (once in Findings, once restated in the Impression),
    and full-text scanning picked up leftover words from both mentions.

    Works backward from the tag's position rather than forward-splitting on
    sentence punctuation -- the tag sits AFTER the sentence's period
    ('...infarction. [region: infarct]'), so a naive forward split puts the
    tag at the START of the next chunk, not attached to the sentence that
    actually makes the claim. This was caught by a failing test before
    shipping, not assumed to work."""
    clean_text = report_text.replace("**", "").replace("*", "")
    tag_pos = clean_text.find("[region: infarct]")
    if tag_pos == -1:
        tag_pos = clean_text.find("[region:infarct]")
    if tag_pos == -1:
        return ""
    preceding_text = clean_text[:tag_pos]
    # find the start of the sentence the tag is attached to: the last
    # sentence-ending punctuation BEFORE this one, or the start of the text
    matches = list(re.finditer(r"[.!?]\s+", preceding_text))
    sentence_start = matches[-2].end() if len(matches) >= 2 else 0
    return preceding_text[sentence_start:].strip()


def extract_claimed_walls(report_text: str) -> list:
    """Returns ALL wall-direction words found in the infarct-location
    sentence specifically (not the whole report). Confirmed necessary after
    Case_P022's report said 'anterior and lateral walls' -- a compound
    claim -- but an earlier single-keyword version only captured 'anterior'
    and missed 'lateral', causing a false mismatch against the real
    'anterolateral' segment."""
    infarct_sentence = extract_infarct_sentence(report_text)
    text_lower = infarct_sentence.lower()
    found = []
    remaining = text_lower
    for keyword in sorted(WALL_KEYWORDS, key=len, reverse=True):
        if keyword in remaining:
            found.append(WALL_KEYWORDS[keyword])
            remaining = remaining.replace(keyword, "", 1)  # avoid double-counting a substring
    return found


def claim_matches_real_segment(claimed_walls: list, real_segment_name: str) -> bool:
    """A claim matches if EVERY word the report used appears in the real
    segment name -- so a compound claim like ['anterior', 'lateral'] only
    counts as a match against a real segment that reflects both directions
    (e.g. 'mid anterolateral'), not just one of them by coincidence.

    Also handles a real spelling mismatch found during testing: 'anterior'
    and 'inferior' are NOT literal substrings of the combining forms used in
    compound segment names ('anterolateral' uses 'antero-', not 'anterior')
    -- checking the plain word against a compound segment silently failed
    even after the compound-claim fix, until this was added."""
    if not claimed_walls:
        return False
    COMBINING_FORMS = {"anterior": "antero", "inferior": "infero"}
    for w in claimed_walls:
        alt = COMBINING_FORMS.get(w)
        if w in real_segment_name or (alt and alt in real_segment_name):
            continue
        return False
    return True


def get_slice_idx(report_data: dict) -> int:
    slice_str = report_data.get("slice_used", "")
    match = re.search(r"slice_(\d+)", slice_str)
    return int(match.group(1)) if match else None


def main():
    out_dir = "outputs_region_grounded"
    files = sorted(glob.glob(os.path.join(out_dir, "*.json")))

    checked, matched, mismatched, unverifiable = 0, 0, [], []

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        case_id = data.get("patient_id", os.path.basename(fpath))
        report_text = data.get("output", "")

        claimed_walls = extract_claimed_walls(report_text)
        if not claimed_walls:
            continue  # no location claim made -- nothing to check

        slice_idx = get_slice_idx(data)
        if slice_idx is None:
            unverifiable.append((case_id, "couldn't determine slice"))
            continue

        try:
            result = assign_segment(case_id, slice_idx)
            infarct_result = result.get("infarct") or {}
            if not infarct_result.get("present"):
                unverifiable.append((case_id, "no infarct segment found (mask may show no infarct on this slice)"))
                continue
            real_segment = infarct_result.get("aha_segment_name", "")
        except Exception as e:
            unverifiable.append((case_id, f"segment check failed: {e}"))
            continue

        checked += 1
        if claim_matches_real_segment(claimed_walls, real_segment):
            matched += 1
        else:
            mismatched.append((case_id, claimed_walls, real_segment))

    print(f"Reports with a checkable wall-location claim: {checked}")
    print(f"Matched the real computed segment: {matched}")
    print(f"Mismatched: {len(mismatched)}")
    print(f"Couldn't verify (error or no clear infarct segment): {len(unverifiable)}")

    if mismatched:
        print(f"\n--- Mismatches (report claim vs. real computed location) ---")
        for case_id, claimed, real in mismatched:
            claimed_str = " + ".join(claimed)
            print(f"  {case_id}: report said '{claimed_str}', real segment is '{real}'")

    if unverifiable:
        print(f"\n--- Couldn't verify ---")
        for case_id, reason in unverifiable:
            print(f"  {case_id}: {reason}")

    print(f"\nReminder: this only checks the anterior/inferior/septal/lateral")
    print(f"direction, not basal/mid/apical level (that part relies on an")
    print(f"unconfirmed slice-ordering assumption). A mismatch here is a real")
    print(f"finding worth double-checking by eye before treating it as certain.")


if __name__ == "__main__":
    main()
