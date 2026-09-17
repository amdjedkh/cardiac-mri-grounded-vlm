# synthetic_generation/

Generates synthetic EMIDEC-style cardiac MRI + masks by running [LeFusion](https://github.com/HINTLab/LeFusion)'s pretrained weights on [Modal](https://modal.com), plus a set of scripts to inspect, verify, and visualize the results.

This doc is written so someone with **no prior exposure to this pipeline** (Carlos, running on his own compute) can get it working without rediscovering the same failures we already hit. Everything below reflects the actual state of the code in this folder — nothing here changes how the scripts behave.

## What's in this folder

**The Modal app**
- `modal_lefusion.py` — defines the Modal app (`lefusion-emidec-poc`). Contains the container image spec (pinned dependencies), and every remote function: `setup_data_and_weights` (downloads LeFusion's data/weights into the volume once), `generate_synthetic_emidec` (runs actual inference), `analyze_synthetic_geometry`, `scan_all_slices_geometry`, `compare_synthetic_to_real_mask`, `render_synthetic_case`, `check_synthetic_consistency`. You don't run this file directly — you `modal deploy` it (see Workflow below).

**Triggering and checking jobs**
- `trigger_lefusion.py` — connects to the *deployed* app and spawns a generation job. Prints a call ID.
- `check_lefusion_result.py <call_id>` — polls a spawned job by that call ID. Works from a different machine/session than the one that started the job, since the job runs detached on Modal's side.

**Inspecting and visualizing synthetic output**
- `render_and_download.py <case>.nii.gz [slice]` — renders one synthetic case (image + mask overlay) as a PNG and saves it locally.
- `inspect_lefusion_output.py` — lists what's actually in the output folder on the volume, with timestamps. Useful for telling genuinely-new files from leftovers of an earlier/killed run.

**Verifying synthetic quality (geometry)**
- `check_synthetic_geometry.py <case>.nii.gz` — angular coverage + contiguity score for infarct/MVO on one auto-picked slice, printed against known real-case baselines.
- `check_all_slices.py <case>.nii.gz` — same contiguity check across *every* slice with infarct, not just one, to see if a low score is a real pattern or a one-slice fluke.
- `scan_real_case_all_slices.py <Case_ID>` — the matching per-slice scan for a **real** EMIDEC case, so real and synthetic numbers are computed the exact same way.

**Verifying synthetic quality (is it real signal or a copy?)**
- `check_real_vs_synthetic_identity.py <case>.nii.gz` — checks whether a "synthetic" mask is actually just its real conditioning mask resampled, not new geometry. Was written specifically because real and "synthetic" P004 produced byte-identical contiguity numbers, which shouldn't happen by chance.
- `compare_two_real_cases.py <Case_A> <Case_B>` — compares two *different* real patients with the same resample-then-compare method, to get a baseline for "how much do two unrelated hearts naturally overlap." Needed to judge whether the identity-check number above is actually suspicious.

**Verifying synthetic quality (slice-to-slice consistency)**
- `check_synthetic_consistency.py <case>.nii.gz` — checks whether cavity size/centroid change smoothly from slice to slice (a real heart tapering) or erratically (independent, uncoordinated slices). Same metric as `verification/scan_real_consistency.py`, so the numbers are directly comparable.
- `plot_consistency_comparison.py --real <Case_ID> [<Case_ID> ...] --synthetic <case>.nii.gz [<case>.nii.gz ...]` — plots cavity-size-per-slice curves for the given real and synthetic cases side by side. Smooth curves = good; jagged zig-zags = bad. Run from your EMIDEC folder (needs local real cases). Both flags are optional and fall back to a small default set of 4 real + 4 synthetic cases if omitted.

The shared metric logic (`compute_slice_profile`, `compute_consistency_metrics`) lives in `verification/slice_consistency_metrics.py`, imported by both the real-case script and `plot_consistency_comparison.py`.

## Setup

### 1. Dependencies

Local machine (to run the trigger/check/render/analysis scripts):
```bash
pip install modal numpy nibabel scipy matplotlib
```
Then authenticate: `modal setup` (opens a browser flow, one-time per machine).

The heavy LeFusion dependency stack (torch, monai, pytorch-lightning, etc.) does **not** need to be installed locally — it only runs inside the Modal container, built automatically from the pinned list in `modal_lefusion.py`. You do not need a GPU locally, and you do not need to clone the LeFusion repo yourself; the container does that too.

### 2. Modal account and volume

You need a Modal account and a persistent volume named `cardiac-data` (or whatever you rename it to — see "Adapting this to your own setup" below):
```bash
modal volume create cardiac-data   # skip if it already exists on your account
```

### 3. First deploy

```bash
modal deploy modal_lefusion.py
```
This builds the container image (installs the pinned dependency list, clones LeFusion) and registers the app on Modal, but does **not** yet download data or weights — that happens the first time `setup_data_and_weights` actually runs (step 1 of the workflow below), triggered automatically by `trigger_lefusion.py`.

## Workflow (order matters)

1. **Deploy**: `modal deploy modal_lefusion.py` — run this once, and again after any code change to the file. (Re-running this is cheap and safe if nothing changed.)
2. **Trigger**: `python trigger_lefusion.py` — this first calls `setup_data_and_weights` and *waits* for it to finish (downloads LeFusion's preprocessed EMIDEC data + pretrained weights into the volume if not already cached — slow on first run, fast after), then spawns `generate_synthetic_emidec` as a detached background job and prints a call ID.
3. **Check**: `python check_lefusion_result.py <call_id>` — poll until it reports finished. Safe to close your laptop and check back later from anywhere; the job runs on Modal's side independent of your local connection.
4. **Render/analyze**: once a case exists in the output folder, run any of the inspection/verification scripts above against it, e.g. `python render_and_download.py Case_P001.nii.gz`.

### Why not just `modal run modal_lefusion.py`?

Don't. `modal run` spins up a *temporary* app that gets torn down the instant the local script exits — including any job you `.spawn()`'d from inside it. We confirmed this with a real run: the job showed `Status=Cancelled`, zero containers ever went live, no logs at all, immediately after "local entrypoint completed" printed. That's why this file has no `@app.local_entrypoint()` — the only supported path is `modal deploy` + `trigger_lefusion.py`, which spawns the job against the already-deployed, persistent app instead.

## Gotchas already solved (read this before re-debugging any of these)

**pip / pytorch-lightning version conflict.** LeFusion's own README requires `pip==22.3.1`, not whatever ships in a modern base image. Modern pip (24.1+) rejects `pytorch-lightning==1.6.4`'s malformed version metadata (`torch (>=1.8.*)` isn't valid PEP 440), which older pip silently tolerated. The image spec downgrades pip explicitly before installing anything else — if you rebuild this from scratch on different infra, keep that ordering.

**Two packages missing from LeFusion's own `requirements.txt`.** A hand-transcribed subset of LeFusion's requirements was missing two packages needed at runtime, discovered only by hitting real `ModuleNotFoundError`s. The fix was to stop hand-picking and install LeFusion's *complete* original pinned list instead (that's the `LEFUSION_REQUIREMENTS` list in `modal_lefusion.py`) — don't trim that list down again without testing a full generation run afterward.

**`typing_extensions` conflict with Modal's own SDK.** LeFusion's requirements pin `typing_extensions==4.2.0`. Modal's SDK runs *inside* the same container to execute the function, and needs a modern `typing_extensions` — installing LeFusion's old pin broke Modal's own import machinery (`AttributeError: module 'typing_extensions' has no attribute 'TypeVar'`). Fixed by explicitly reinstalling `typing_extensions>=4.12` in a separate `run_commands` step *after* LeFusion's requirements are installed, so it doesn't get silently re-pinned back down.

**Missing system graphics library.** `opencv-python` (a LeFusion dependency) needs `libgl1` and `libglib2.0-0` at the OS level, which a minimal `debian_slim` base image doesn't include. The image spec `apt_install`s both explicitly before the pip install step.

**Nested-path issue in the preprocessed data archive.** The HuggingFace `.tar` archives for LeFusion's preprocessed EMIDEC data preserve the original author's absolute directory structure (confirmed from a real extraction: files landed under `home/liuyuhe/LeFusion/LeFusion_LIDC/data/EMIDEC/...`), not a clean top-level folder the way LeFusion's README implies. `setup_data_and_weights` handles this by searching (`glob`) for the `Pathological`/`Normal` folders inside the extracted tree and symlinking whatever it finds to the expected path, rather than hardcoding the exact nested path — if you re-extract these archives on different infra and get an unexpected structure, check that the glob is still finding the folders rather than assuming the path is fixed.

**`modal run` vs `modal deploy` trap.** Covered above under Workflow — `modal run` silently cancels any spawned background job the moment the local script exits. Always use `modal deploy` + `trigger_lefusion.py`.

**Mask axis-order bug.** The synthetic image array and mask array can have the *same* shape but in a *different* axis order (observed: image `72x72x10` = HxWxD, mask `10x72x72` = DxHxW) — not a resolution mismatch, just a transposed array. Naively slicing axis 2 on both (as if they shared a convention) sliced through the wrong axis on the mask, producing a thin vertical-strip artifact in the first real render. Every function in `modal_lefusion.py` that touches both arrays (`render_synthetic_case`, `analyze_synthetic_geometry`, `scan_all_slices_geometry`, `check_synthetic_consistency`) detects this by comparing sorted shapes and transposes the mask to match before doing anything else. If you add a new function that reads both image and mask, copy that same shape-check-and-transpose block rather than assuming the arrays already line up.

One more worth knowing even though it isn't in the list above: image and mask can also have **different depths** (observed: mask 34 slices, image 10 slices) — likely because the mask preserves the real conditioning case's native resolution while the synthetic image is fixed at the diffusion model's generation size. `render_synthetic_case` maps the chosen slice *proportionally* between the two rather than assuming the indices line up directly.

## Adapting this to your own setup

This is currently built against **my own Modal account and volume**, hardcoded in a few places:
- `modal.Volume.from_name("cardiac-data", ...)` in `modal_lefusion.py` and `inspect_lefusion_output.py`
- The app name `"lefusion-emidec-poc"`, referenced in `trigger_lefusion.py`, `check_synthetic_geometry.py`, `check_all_slices.py`, `check_real_vs_synthetic_identity.py`, `render_and_download.py`, `check_synthetic_consistency.py`, `plot_consistency_comparison.py`

**If you're using your own Modal account:** run `modal setup` to authenticate as yourself, then either create a volume named `cardiac-data` under your own account (simplest — no code changes needed, since Modal resources are scoped per-account/workspace already), or if you want a different volume/app name, change the two constants above (`VOLUME_PATH`'s name and `app = modal.App("...")`) in `modal_lefusion.py` and update the matching `modal.Function.from_name("...", ...)` calls in every script listed above to match. First deploy under your account will re-download LeFusion's data and weights into your own volume (same first-run cost described in Setup).

**If you're moving off Modal entirely to different compute:** everything under "Gotchas already solved" above — the pinned dependency versions, the pip downgrade, the `typing_extensions` fix, the `libgl1`/`libglib2.0-0` system packages, the nested-tar-path handling, the axis-order fix — still applies, since those are properties of LeFusion itself and its data archives, not of Modal. What would need rewriting is the *orchestration*: the `@app.function` decorators, the volume-mount pattern, the deploy/spawn/poll workflow, and the `modal.Function.from_name` calls in every trigger/check/render script — all of that is Modal-specific plumbing around the same underlying `python LeFusion/inference/inference.py data_type=emidec ...` command (see the `cmd = [...]` list inside `generate_synthetic_emidec` in `modal_lefusion.py` for the exact args), which would need to be run and orchestrated some other way (e.g. a local script with a real GPU, a different cloud GPU provider, a Slurm job).

## Known open issue with the output itself

Independent of the pipeline mechanics above: whether LeFusion's masks/images produced here represent genuinely new synthetic pathology geometry, or largely reproduce the real conditioning case's geometry with new texture, is an open, only partially investigated question — see `check_real_vs_synthetic_identity.py` / `compare_two_real_cases.py` above and open question #2 in the top-level [README.md](../README.md). Don't treat a generated case as validated just because the pipeline ran without errors.
