"""
split_utils.py

Small helpers shared by make_split.py, check_split_leakage.py and
synthetic_generation/modal_lefusion.py (which ships this file into the Modal
container next to split_v1.json).
"""

import os
import re
import json

DEFAULT_SPLIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "split_v1.json")

_EXTENSIONS = (".nii.gz", ".nii", ".png", ".json", ".jsonl")


def source_patient(name: str) -> str:
    """'Case_P004__dup99zlas.nii.gz' -> 'Case_P004'. Synthetic outputs made by
    reusing a real case carry a __dup<suffix>; the patient is the part before it."""
    base = os.path.basename(str(name).strip())
    for ext in _EXTENSIONS:
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    base = base.split("__dup")[0]
    m = re.match(r"^Case[_ ]?([NP]\d+)", base)
    return f"Case_{m.group(1)}" if m else base


def load_split(path: str = DEFAULT_SPLIT_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def test_ids(path: str = DEFAULT_SPLIT_PATH) -> set:
    return set(load_split(path)["test"])


def filter_input_pool(filenames, held_out: set):
    """Drops every file whose source patient is in the held-out test set."""
    return [f for f in filenames if source_patient(f) not in held_out]
