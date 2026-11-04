"""
modal_nnunet.py

nnU-Net v2 (2D) on Modal for the segmentation comparison. Data lives in the
cardiac-data volume under nnunet/ (nnUNet_raw, nnUNet_preprocessed,
nnUNet_results).

Two steps, so GPU time is only billed for training:
    modal run experiments/nnunet/modal_nnunet.py::prep             (CPU: verify dataset, plan, preprocess)
    modal run experiments/nnunet/modal_nnunet.py::run_train        (GPU: ONE reduced-epoch run on arm 1, fold 'all')

Upload first:
    modal volume put cardiac-data ~/nnunet_work/nnUNet_raw/Dataset501_EMIDECArm1 nnunet/nnUNet_raw/Dataset501_EMIDECArm1

Reduced-epoch trainers (nnUNetTrainer_20epochs etc.) are for pipeline and
cost checks only, never for reported results. Training uses fold 'all' (all 80
train patients, no validation split), so nnU-Net's own validation is computed on
the training cases and only shows the model fits; no test case is used here.
"""

import os
import re
import sys
import json
import shutil
import time
import subprocess

import modal

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _REPO_ROOT)
import config

app = modal.App("nnunet-emidec")
volume = modal.Volume.from_name("cardiac-data", create_if_missing=True)
VOL = "/cardiac-data"
GPU = "L4"
N_CPU = 8

_HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.11").pip_install("nnunetv2==2.8.1")
         .add_local_file(os.path.join(_REPO_ROOT, "config.py"), "/root/config.py")
         .add_local_file(os.path.join(_HERE, "nnUNetTrainerEmidec.py"), "/root/nnUNetTrainerEmidec.py"))

# arm -> nnU-Net dataset, trainer class, share of insert volumes per batch (see SEGMENTATION_COMPARISON_PLAN.md)
ARMS = {1: (501, "nnUNetTrainerEmidec", 0.0),
        2: (502, "nnUNetTrainerEmidec", 1 / 3),
        3: (501, "nnUNetTrainerEmidecStrongAug", 0.0),
        4: (504, "nnUNetTrainerEmidec", 1 / 3),
        5: (505, "nnUNetTrainerEmidec", 1 / 3)}


def _env(results=None, extra=None):
    e = dict(os.environ)
    e.update({"nnUNet_raw": f"{VOL}/nnunet/nnUNet_raw",
              "nnUNet_preprocessed": f"{VOL}/nnunet/nnUNet_preprocessed",
              "nnUNet_results": results or f"{VOL}/nnunet/nnUNet_results",
              "nnUNet_n_proc_DA": str(N_CPU)})
    e.update(extra or {})
    return e


def _run(cmd, env=None):
    t = time.time()
    p = subprocess.run(cmd, env=env or _env(), capture_output=True, text=True)
    return {"returncode": p.returncode, "seconds": round(time.time() - t, 1),
            "tail": (p.stdout + p.stderr)[-2500:]}


@app.function(image=image, cpu=N_CPU, memory=16384, volumes={VOL: volume}, timeout=1800)
def prepare(trainer="nnUNetTrainer_20epochs"):
    from importlib.metadata import version
    import nnunetv2
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    name = config.NNUNET_DATASET_NAME
    d = config.NNUNET_DATASET_ID
    cls = recursive_find_python_class(os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                      trainer, "nnunetv2.training.nnUNetTrainer")
    out = {"nnunetv2": version("nnunetv2"), "torch": version("torch"), "trainer_found": cls is not None}
    if cls is None:
        return out
    volume.reload()
    res = _run(["nnUNetv2_plan_and_preprocess", "-d", str(d), "-c", "2d", "--verify_dataset_integrity", "-np", str(N_CPU)])
    out["plan_and_preprocess"] = res
    if res["returncode"] != 0:
        return out
    pre = f"{VOL}/nnunet/nnUNet_preprocessed/{name}"
    out["preprocessed"] = sorted(os.listdir(pre))
    volume.commit()
    return out


@app.function(image=image, gpu=GPU, cpu=N_CPU, memory=16384, volumes={VOL: volume}, timeout=3600)
def train(trainer="nnUNetTrainer_20epochs", fold="all"):
    name = config.NNUNET_DATASET_NAME
    volume.reload()
    res = _run(["nnUNetv2_train", str(config.NNUNET_DATASET_ID), "2d", str(fold), "-tr", trainer, "-device", "cuda"])  # no --val: fold all
    out = {"gpu": GPU, "trainer": trainer, "fold": fold, "train": res}
    folder = f"{VOL}/nnunet/nnUNet_results/{name}/{trainer}__nnUNetPlans__2d/fold_{fold}"
    if os.path.isdir(folder):
        logs = sorted(f for f in os.listdir(folder) if f.startswith("training_log"))
        if logs:
            text = open(os.path.join(folder, logs[-1])).read()
            out["epoch_seconds"] = [float(x) for x in re.findall(r"Epoch time: ([0-9.]+) s", text)]
            out["pseudo_dice_last"] = re.findall(r"Pseudo dice (\[.*?\])", text)[-1:]
        summ = os.path.join(folder, "validation", "summary.json")
        if os.path.exists(summ):
            s = json.load(open(summ))
            out["val_foreground_mean"] = s["foreground_mean"]
            out["val_mean_per_label"] = {k: v["Dice"] for k, v in s["mean"].items()}
    volume.commit()
    return out


@app.local_entrypoint()
def prep(trainer="nnUNetTrainer_20epochs"):
    print(json.dumps(prepare.remote(trainer), indent=1))


@app.local_entrypoint()
def run_train(trainer="nnUNetTrainer_20epochs", fold="all"):
    print(json.dumps(train.remote(trainer, str(fold)), indent=1))


@app.function(image=image, gpu=GPU, cpu=N_CPU, memory=16384, volumes={VOL: volume}, timeout=6 * 3600, retries=1)
def train_arm(arm, seed, epochs_override=None):
    """One full run (1000 epochs, final checkpoint) of one arm and seed, fold 'all'. Each (arm, seed) has its own
    results folder, nnunet/results/arm{arm}_seed{seed}, so seeds cannot overwrite each other.
    Preprocessed Dataset50x folders must already be on the volume (same plans as Dataset501)."""
    import nnunetv2
    shutil.copy("/root/nnUNetTrainerEmidec.py", os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer", "nnUNetTrainerEmidec.py"))
    import threading
    did, trainer, frac = ARMS[arm]
    results = f"{VOL}/nnunet/results/arm{arm}_seed{seed}"
    volume.reload()
    stop = threading.Event()

    def _commit_loop():  # makes the training log and checkpoints visible from outside while the job runs
        while not stop.wait(120):
            try:
                volume.commit()
            except Exception:
                pass
    threading.Thread(target=_commit_loop, daemon=True).start()
    # --c: continue from checkpoint_latest.pth (saved every 50 epochs) if one exists, else start from scratch
    res = _run(["nnUNetv2_train", str(did), "2d", "all", "-tr", trainer, "-device", "cuda", "--c"],
               _env(results, {"EMIDEC_SEED": str(seed), "EMIDEC_SYN_FRACTION": str(frac)}))
    stop.set()
    out = {"arm": arm, "seed": seed, "gpu": GPU, "dataset": did, "trainer": trainer, "syn_fraction": frac, "train": res}
    for root, _, files in os.walk(results):
        if "checkpoint_final.pth" in files:
            out["checkpoint_final"] = os.path.join(root, "checkpoint_final.pth")
            out["epoch_seconds"] = []
            for f in sorted(f for f in files if f.startswith("training_log")):
                out["epoch_seconds"] += [float(x) for x in re.findall(r"Epoch time: ([0-9.]+) s", open(os.path.join(root, f)).read())]
    volume.commit()
    if res["returncode"] != 0 or "checkpoint_final" not in out:
        raise RuntimeError(f"arm {arm} seed {seed} failed: {res['tail']}")  # raising lets Modal retry (resumes via --c)
    return out


@app.local_entrypoint()
def run_arms(arms="1,2,3", seeds="0,1,2"):
    """Launch every (arm, seed) as its own parallel GPU job. Only run after an explicit go."""
    jobs = [(int(a), int(s)) for a in arms.split(",") for s in seeds.split(",")]
    print("jobs:", jobs)
    for arm, seed in jobs:  # spawn: jobs keep running if this client disconnects (use `modal run --detach`)
        call = train_arm.spawn(arm, seed)
        print(f"arm {arm} seed {seed} function_call_id {call.object_id}")
