"""
patient_schema.py

Defines the per-patient structured record combining:
  - EMIDEC clinical characteristics (from the dataset's own metadata file)
  - Mask-derived quantitative measurements (from measurements.py)
  - Provenance tags, since every downstream field must be traceable to real
    data vs. something a model will later draft.

One JSON file per patient, e.g. outputs/patients/P1.json

EMIDEC clinical fields, per Lalande et al. (2020), are read from the dataset's
own per-case text/metadata file. Field names below follow the challenge's
documented fields — verify against your local copy of the metadata file
before trusting the exact key names, since minor formatting differences
across EMIDEC distributions are possible.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from measurements import MeasurementResult


@dataclass
class EmidecClinicalMetadata:
    """
    Structured clinical fields as provided natively by EMIDEC. All REAL / not synthetic.

    Verified against an actual EMIDEC Case_PXXX.txt on 2026-07-23. Real file format is
    flat "Field : Value" text, e.g.:

        Case P001
        Gap between slices : 10 mm
        Sex : M
        Age : 32
        Tobacco : 1
        Overweight : N
        Arterial hypertension : N
        Diabetes : N
        Familial history of coronary artery disease: N
        ECG (ST +) : Y
        Troponin : 130
        Killip Max: 1
        FEVG : 35
        NTProBNP : 447

    Notes on fields that are NOT simple Y/N or numeric pass-through:
      - tobacco: stored as the RAW integer code from the file (e.g. 1). The file does not
        document what 0/1/2 mean (never/current/former, in unknown order) -- do not assume
        a mapping. If EMIDEC's own documentation defines this elsewhere, update
        `tobacco_code_meaning` once confirmed; until then this stays an unlabeled raw code.
      - gap_between_slices_mm: the dataset's own stated slice thickness. Cross-check this
        against the affine-derived spacing from verify_labels.py -- if they disagree, prefer
        this file-stated value, since it's the documented acquisition parameter.
      - fevg -> mapped to lvef_echo_percent (FEVG = Fraction d'Ejection du Ventricule Gauche,
        i.e. LVEF; EMIDEC does not specify echo vs MRI-derived, field kept as-is from source).
    """
    sex: Optional[str] = None                  # "M" / "F"
    age: Optional[int] = None
    gap_between_slices_mm: Optional[float] = None
    tobacco_raw_code: Optional[int] = None       # unlabeled raw code, see docstring
    tobacco_code_meaning: str = "unknown -- not documented in source file, do not assume"
    overweight: Optional[bool] = None
    arterial_hypertension: Optional[bool] = None
    diabetes: Optional[bool] = None
    family_history_cad: Optional[bool] = None
    ecg_stemi: Optional[bool] = None             # ST-elevation on ECG
    troponin: Optional[float] = None             # raw value; units not stated in source file
    killip_max: Optional[int] = None             # 1-4
    lvef_echo_percent: Optional[float] = None     # from FEVG field, see docstring
    ntprobnp: Optional[float] = None

    source: str = "EMIDEC_clinical_metadata_file"  # provenance tag


def _parse_yn(value: str) -> Optional[bool]:
    v = value.strip().upper()
    if v == "Y":
        return True
    if v == "N":
        return False
    return None


def parse_emidec_clinical_txt(path: str) -> EmidecClinicalMetadata:
    """
    Parses a real EMIDEC 'Case PXXX.txt' / 'Case NXXX.txt' clinical file into
    EmidecClinicalMetadata. Format is flat "Field : Value" or "Field: Value" lines.

    Encoding note: some EMIDEC files use a non-breaking space (U+00A0) before the
    colon on certain fields (French typographic convention -- e.g. "Sex\xa0: M"),
    confirmed present in a real Case_P019.txt. Reading as plain utf-8 corrupts that
    byte into a replacement character, silently breaking key matching on exactly
    those fields (observed: "Sex" and "Gap between slices" affected, others not,
    depending on how each line was originally typed/edited). Reading as cp1252
    (Windows-1252, the standard Windows French locale encoding) decodes that byte
    correctly to a real non-breaking-space character, which str.strip() removes.
    """
    fields = {}
    with open(path, "r", encoding="cp1252", errors="replace") as f:
        for line in f:
            line = line.replace("\xa0", " ").strip()  # normalize NBSP -> regular space, then strip
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.replace("\xa0", " ").strip().lower()
            value = value.replace("\xa0", " ").strip()
            fields[key] = value

    def get_float(*keys):
        for k in keys:
            if k in fields and fields[k] != "":
                try:
                    # strip non-numeric suffixes like "mm"
                    digits = "".join(c for c in fields[k] if c.isdigit() or c in ".-")
                    return float(digits) if digits else None
                except ValueError:
                    return None
        return None

    def get_int(*keys):
        val = get_float(*keys)
        return int(val) if val is not None else None

    def get_str(*keys):
        for k in keys:
            if k in fields:
                return fields[k]
        return None

    def get_yn(*keys):
        for k in keys:
            if k in fields:
                return _parse_yn(fields[k])
        return None

    return EmidecClinicalMetadata(
        sex=get_str("sex"),
        age=get_int("age"),
        gap_between_slices_mm=get_float("gap between slices"),
        tobacco_raw_code=get_int("tobacco"),
        overweight=get_yn("overweight"),
        arterial_hypertension=get_yn("arterial hypertension"),
        diabetes=get_yn("diabetes"),
        family_history_cad=get_yn("familial history of coronary artery disease"),
        ecg_stemi=get_yn("ecg (st +)"),
        troponin=get_float("troponin"),
        killip_max=get_int("killip max"),
        lvef_echo_percent=get_float("fevg"),
        ntprobnp=get_float("ntprobnp"),
    )


@dataclass
class PatientRecord:
    patient_id: str
    emidec_group: Optional[str] = None          # "normal" or "pathological", per EMIDEC split
    clinical: EmidecClinicalMetadata = field(default_factory=EmidecClinicalMetadata)
    measurements: Optional[dict] = None          # filled from MeasurementResult.to_dict()

    # Fields to be filled in LATER pipeline stages -- left empty/None here on purpose,
    # so it's obvious from the file alone what stage of the pipeline has been completed.
    generated_report: Optional[dict] = None       # {"model": ..., "findings": ..., "impression": ...}
    expert_validation: Optional[dict] = None       # {"reviewer": ..., "disposition": ..., "edits": ...}

    provenance: dict = field(default_factory=lambda: {
        "clinical_metadata": "real",
        "measurements": "real_derived_from_mask",
        "generated_report": "not_yet_generated",
        "expert_validation": "not_yet_validated",
    })

    def to_dict(self):
        d = asdict(self)
        return d

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def build_patient_record(
    patient_id: str,
    measurement_result: MeasurementResult,
    clinical: EmidecClinicalMetadata = None,
    emidec_group: str = "pathological",
) -> PatientRecord:
    record = PatientRecord(
        patient_id=patient_id,
        emidec_group=emidec_group,
        clinical=clinical or EmidecClinicalMetadata(),
        measurements=measurement_result.to_dict(),
    )
    return record


# ----------------------------------------------------------------------------
# Self-test: build one record using the synthetic measurement result
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from measurements import extract_measurements, DEFAULT_LABEL_MAP
    import numpy as np

    # reuse the same synthetic mask construction as measurements.py self-test
    shape = (10, 10, 8)
    mask = np.zeros(shape, dtype=int)
    yy, xx = np.mgrid[0:10, 0:10]
    dist = np.sqrt((yy - 5) ** 2 + (xx - 5) ** 2)
    cavity_ring = dist <= 2
    myo_ring = (dist > 2) & (dist <= 4)
    for s in range(2, 6):
        mask[cavity_ring, s] = DEFAULT_LABEL_MAP["lv_cavity"]
        mask[myo_ring, s] = DEFAULT_LABEL_MAP["myocardium_normal"]
    infarct_wedge = myo_ring & (xx >= 5) & (yy >= 5)
    for s in range(3, 6):
        mask[infarct_wedge, s] = DEFAULT_LABEL_MAP["infarct"]
    mvo_core = infarct_wedge & (dist <= 3.2)
    mask[mvo_core, 4] = DEFAULT_LABEL_MAP["mvo"]

    result = extract_measurements(mask_array=mask, voxel_spacing_mm=(1.5, 1.5, 8.0), patient_id="P_TEST")

    # Self-test the parser against the real Case P001.txt content (reconstructed here
    # verbatim from what was confirmed on the user's machine, to test parsing logic
    # without needing the actual file present in this environment).
    sample_txt = """Case P001
Gap between slices : 10 mm
Sex : M
Age : 32
Tobacco : 1
Overweight : N
Arterial hypertension : N
Diabetes : N
Familial history of coronary artery disease: N
ECG (ST +) : Y
Troponin : 130
Killip Max: 1
FEVG : 35
NTProBNP : 447
"""
    with open("/tmp/Case_P001_sample.txt", "w") as f:
        f.write(sample_txt)

    clinical = parse_emidec_clinical_txt("/tmp/Case_P001_sample.txt")
    print("Parsed clinical metadata from sample Case P001.txt:")
    print(json.dumps(asdict(clinical), indent=2))

    assert clinical.sex == "M"
    assert clinical.age == 32
    assert clinical.gap_between_slices_mm == 10.0
    assert clinical.tobacco_raw_code == 1
    assert clinical.overweight is False
    assert clinical.arterial_hypertension is False
    assert clinical.diabetes is False
    assert clinical.family_history_cad is False
    assert clinical.ecg_stemi is True
    assert clinical.troponin == 130.0
    assert clinical.killip_max == 1
    assert clinical.lvef_echo_percent == 35.0
    assert clinical.ntprobnp == 447.0
    print("\nParser self-test passed against real Case P001.txt format.")

    record = build_patient_record("P001", result, clinical, emidec_group="pathological")
    record.save("/tmp/P001.json")
    print("\nSaved to /tmp/P001.json.")
