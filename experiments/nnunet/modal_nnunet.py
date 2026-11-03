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

image = (modal.Image.debian_slim(python_version="3.11").pip_install("nnunetv2")
         .add_local_file(os.path.join(_REPO_ROOT, "config.py"), "/root/config.py"))


def _env():
    e = dict(os.environ)
    e.update({"nnUNet_raw": f"{VOL}/nnunet/nnUNet_raw",
              "nnUNet_preprocessed": f"{VOL}/nnunet/nnUNet_preprocessed",
              "nnUNet_results": f"{VOL}/nnunet/nnUNet_results",
              "nnUNet_n_proc_DA": str(N_CPU)})
    return e


def _run(cmd):
    t = time.time()
    p = subprocess.run(cmd, env=_env(), capture_output=True, text=True)
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
