"""
trigger_lefusion.py

Run this AFTER deploying (modal deploy modal_lefusion.py), not with
"modal run". This connects to the already-deployed, persistent app and
spawns the job there -- since the app is deployed rather than ephemeral,
the job survives regardless of what this local script does afterward,
including your connection dropping or this script being closed.

Usage:
    modal deploy modal_lefusion.py    (one-time, or after any code change)
    python trigger_lefusion.py
"""

import modal

print("Connecting to deployed app...")
setup_fn = modal.Function.from_name("lefusion-emidec-poc", "setup_data_and_weights")
gen_fn = modal.Function.from_name("lefusion-emidec-poc", "generate_synthetic_emidec")

print("Step 1: ensuring data + weights are ready (waits for this to finish, it's usually quick if already cached)...")
setup_fn.remote()

print("\nStep 2: spawning LeFusion inference on the deployed app (detached)...")
print("Requesting 12 MORE cases (91 already exist -> 103 total, target 100+).")
call = gen_fn.spawn(batch_size=1, max_cases=12, reuse_when_exhausted=True)

print(f"\nJob submitted. Call ID: {call.object_id}")
print("This job now runs independently of this script and your connection.")
print(f"Check on it anytime with:\n  python check_lefusion_result.py {call.object_id}")
