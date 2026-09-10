"""
debug_sex_field.py

Diagnoses why a specific field (e.g. sex) might come out blank for one case
but not others. Prints raw line content with repr() to reveal hidden
characters, plus the exact parsed fields dict.

Usage (PowerShell):
    python debug_sex_field.py "Case P019.txt"
"""

import sys

path = sys.argv[1] if len(sys.argv) > 1 else "Case P019.txt"

print(f"--- Raw lines of {path} (repr, to reveal hidden characters) ---")
with open(path, "rb") as f:
    raw_bytes = f.read()

print(f"First 20 bytes (hex): {raw_bytes[:20].hex()}")
print(f"File length: {len(raw_bytes)} bytes\n")

with open(path, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        print(f"line {i}: {repr(line)}")

print("\n--- Parsed fields dict (as patient_schema.py would build it) ---")
fields = {}
with open(path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip().lower()] = value.strip()

for k, v in fields.items():
    print(f"  key={repr(k)}  value={repr(v)}")

print("\n--- Direct lookup test ---")
print(f"'sex' in fields: {'sex' in fields}")
if "sex" in fields:
    print(f"fields['sex'] = {repr(fields['sex'])}")
