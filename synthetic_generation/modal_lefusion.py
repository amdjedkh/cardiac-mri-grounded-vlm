"""
modal_lefusion.py

Runs LeFusion's pretrained EMIDEC model on Modal, using your existing
cardiac-data volume. Everything in this script (repo structure, exact
inference command, pinned dependency versions) was verified directly
against the real HINTLab/LeFusion repo before writing this, not guessed
from memory -- specifically:

  - emidec_inference.sh confirmed to call:
    python LeFusion/inference/inference.py data_type=emidec ...
    (a Hydra-config-style script, args passed as key=value)
  - requirements.txt confirmed pinned: torch==2.1.2, monai==0.9.0,
    nibabel==4.0.1, numpy==1.23.0, hydra-core==1.2.0, python 3.10
  - Data path: data/EMIDEC/Pathological/ (from the HuggingFace preprocessed
    EMIDEC set, 57 pathological + 43 healthy cases per LeFusion's own README)
  - Weights path: LeFusion/LeFusion_Model/EMIDEC/emidec.pt

FLAG: this pinned stack (numpy 1.23, torch 2.1.2, CUDA 12.1) is old relative
to what Modal's newer base images assume. This deploy may need debugging on
first real run -- treat this as a first attempt, not a guaranteed-working
script, the same way every other new integration in this project needed at
least one real fix after first contact with real infrastructure.

FLAG 2: LeFusion's own benchmark used a 40GB A100. This starts on A10G
(24GB, matches your existing MedGemma setup) to reuse what you already have
running -- if you hit an out-of-memory error, that's the first thing to
change, swap gpu="A10G" for gpu="A100" below.

Setup (one-time):
    modal volume create cardiac-data   (skip if it already exists)
    modal deploy modal_lefusion.py     (first deploy also downloads data +
                                         weights into the volume, will be
                                         slower than later runs)

Usage:
    modal run modal_lefusion.py
"""

import modal

app = modal.App("lefusion-emidec-poc")

volume = modal.Volume.from_name("cardiac-data", create_if_missing=True)
VOLUME_PATH = "/cardiac-data"

LEFUSION_REQUIREMENTS = [
    "absl-py==1.1.0", "accelerate==0.11.0", "aiohttp==3.8.1", "aiosignal==1.2.0",
    "antlr4-python3-runtime==4.9.3", "async-timeout==4.0.2", "attrs==21.4.0",
    "autopep8==1.6.0", "blessed==1.20.0", "blobfile==2.1.1", "cachetools==5.2.0",
    "certifi==2022.6.15", "charset-normalizer==2.0.12", "click==8.1.3",
    "contextlib2==21.6.0", "cycler==0.11.0", "Deprecated==1.2.13",
    "docker-pycreds==0.4.0", "einops==0.4.1", "einops-exts==0.0.3",
    "elasticdeform==0.5.1", "ema-pytorch==0.0.8", "filelock==3.15.4",
    "fonttools==4.34.4", "frozenlist==1.3.0", "fsspec==2022.5.0", "ftfy==6.1.1",
    "future==0.18.2", "gitdb==4.0.9", "GitPython==3.1.27", "glob2==0.7",
    "google-auth==2.9.0", "google-auth-oauthlib==0.4.6", "gpustat==1.1.1",
    "grpcio==1.47.0", "h5py==3.7.0", "huggingface-hub==0.17.3", "humanize==4.2.2",
    "hydra-core==1.2.0", "idna==3.3", "imageio==2.19.3", "imageio-ffmpeg==0.4.7",
    "importlib-metadata==4.12.0", "importlib-resources==5.9.0", "Jinja2==3.1.4",
    "joblib==1.1.0", "kiwisolver==1.4.3", "lxml==4.9.1", "Markdown==3.3.7",
    "MarkupSafe==2.1.5", "matplotlib==3.5.2", "ml_collections==0.1.1",
    "monai==0.9.0", "mpmath==1.3.0", "multidict==6.0.2", "networkx==2.8.5",
    "nibabel==4.0.1", "nilearn==0.9.1", "numpy==1.23.0",
    "nvidia-cublas-cu12==12.1.3.1", "nvidia-cuda-cupti-cu12==12.1.105",
    "nvidia-cuda-nvrtc-cu12==12.1.105", "nvidia-cuda-runtime-cu12==12.1.105",
    "nvidia-cudnn-cu12==8.9.2.26", "nvidia-cufft-cu12==11.0.2.54",
    "nvidia-curand-cu12==10.3.2.106", "nvidia-cusolver-cu12==11.4.5.107",
    "nvidia-cusparse-cu12==12.1.0.106", "nvidia-ml-py==12.560.30",
    "nvidia-nccl-cu12==2.18.1", "nvidia-nvjitlink-cu12==12.6.20",
    "nvidia-nvtx-cu12==12.1.105", "oauthlib==3.2.0", "omegaconf==2.2.3",
    "opencv-python==4.10.0.84", "packaging==24.1", "pandas==1.4.3",
    "pathtools==0.1.2", "Pillow==9.1.1", "promise==2.3", "protobuf==3.20.1",
    "psutil==6.0.0", "pyasn1==0.4.8", "pyasn1-modules==0.2.8",
    "pycodestyle==2.8.0", "pycryptodomex==3.21.0", "pyDeprecate==0.3.1",
    "pydicom==2.3.0", "pyparsing==3.1.2", "python-dateutil==2.9.0.post0",
    "pytorch-lightning==1.6.4", "pytz==2022.1", "PyWavelets==1.3.0",
    "PyYAML==6.0", "pyzmq==19.0.2", "regex==2022.6.2", "requests==2.28.0",
    "requests-oauthlib==1.3.1", "rotary-embedding-torch==0.1.5", "rsa==4.8",
    "safetensors==0.4.4", "scikit-image==0.19.3", "scikit-learn==1.1.2",
    "scikit-video==1.1.11", "scipy==1.8.1", "seaborn==0.11.2",
    "sentry-sdk==1.7.2", "setproctitle==1.2.3", "shortuuid==1.0.9",
    "SimpleITK==2.1.1.2", "six==1.16.0", "sk-video==1.1.10", "smmap==5.0.0",
    "sympy==1.13.1", "tensorboard==2.11.2", "tensorboard-data-server==0.6.1",
    "tensorboard-plugin-wit==1.8.1", "tensorboardX==2.4.1",
    "threadpoolctl==3.1.0", "tifffile==2022.8.3", "timm==1.0.8", "toml==0.10.2",
    "torch==2.1.2", "torch-tb-profiler==0.4.0", "torchio==0.18.80",
    "torchmetrics==0.9.1", "torchstat==0.0.7", "torchvision==0.16.2",
    "tqdm==4.64.0", "triton==2.1.0",
    "urllib3==1.26.9", "wandb==0.12.21", "wcwidth==0.2.13", "Werkzeug==2.1.2",
    "wrapt==1.14.1", "yarl==1.7.2", "zipp==3.8.0",
]

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libgl1", "libglib2.0-0")
    .run_commands(
        # LeFusion's own README explicitly requires pip==22.3.1 -- modern pip
        # (24.1+) rejects pytorch-lightning==1.6.4's malformed version metadata
        # ("torch (>=1.8.*)"), which older pip silently tolerated. Missed this
        # on the first attempt; confirmed via the real build error.
        "python -m pip install pip==22.3.1",
        "python -m pip install " + " ".join(LEFUSION_REQUIREMENTS),
        # Modal's own SDK runs INSIDE this container to execute the function,
        # and needs a modern typing_extensions -- LeFusion's requirements.txt
        # pins typing_extensions==4.2.0, which broke Modal's own import
        # machinery when installed (confirmed via a real
        # "AttributeError: module 'typing_extensions' has no attribute
        # 'TypeVar'" failure). Explicitly restoring a modern version here,
        # after LeFusion's own packages are already installed, so this
        # doesn't get silently re-pinned back down by anything above.
        "python -m pip install 'typing_extensions>=4.12'",
    )
    .run_commands(
        "git clone https://github.com/HINTLab/LeFusion.git /root/LeFusion",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
def setup_data_and_weights():
    import os
    import glob
    import subprocess

    data_dir = f"{VOLUME_PATH}/LeFusion_data/EMIDEC"
    model_dir = f"{VOLUME_PATH}/LeFusion_model/EMIDEC"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    def find_and_link(folder_name, dest_link):
        # the .tar files preserve the original author's absolute directory
        # structure (confirmed from a real extraction: files landed under
        # home/liuyuhe/LeFusion/LeFusion_LIDC/data/EMIDEC/...), not a clean
        # top-level folder as the README implies. Search for it instead of
        # hardcoding that exact path, so this doesn't break if it varies.
        matches = glob.glob(f"{data_dir}/**/{folder_name}", recursive=True)
        matches = [m for m in matches if os.path.isdir(m)]
        if not matches:
            return None
        real_path = os.path.abspath(matches[0])
        if os.path.islink(dest_link) or os.path.exists(dest_link):
            os.remove(dest_link) if os.path.islink(dest_link) else None
        if not os.path.exists(dest_link):
            os.symlink(real_path, dest_link)
        return real_path

    pathological_link = f"{data_dir}/Pathological"
    normal_link = f"{data_dir}/Normal"

    if not os.path.exists(pathological_link):
        subprocess.run(
            ["wget", "-q",
             "https://huggingface.co/datasets/YuheLiuu/LeFusion_Preprocessed_Data/resolve/main/EMIDEC/Pathological.tar",
             "-O", f"{data_dir}/Pathological.tar"], check=True)
        subprocess.run(["tar", "-xf", f"{data_dir}/Pathological.tar", "-C", data_dir], check=True)
        found = find_and_link("Pathological", pathological_link)
        print("Pathological extracted to:", found)

    if not os.path.exists(normal_link):
        subprocess.run(
            ["wget", "-q",
             "https://huggingface.co/datasets/YuheLiuu/LeFusion_Preprocessed_Data/resolve/main/EMIDEC/Normal.tar",
             "-O", f"{data_dir}/Normal.tar"], check=True)
        subprocess.run(["tar", "-xf", f"{data_dir}/Normal.tar", "-C", data_dir], check=True)
        found = find_and_link("Normal", normal_link)
        print("Normal extracted to:", found)

    if not os.path.exists(f"{model_dir}/emidec.pt"):
        subprocess.run(
            ["wget", "-q",
             "https://huggingface.co/YuheLiuu/LeFusion_Pretrained_model/resolve/main/emidec.pt",
             "-O", f"{model_dir}/emidec.pt"], check=True)

    volume.commit()
    print("Data and weights ready in volume.")
    print("Pathological link resolves to:", os.path.realpath(pathological_link)
          if os.path.exists(pathological_link) else "MISSING")
    print("Pathological contents:", os.listdir(pathological_link)
          if os.path.exists(pathological_link) else "N/A")
    print("Weight file exists:", os.path.exists(f"{model_dir}/emidec.pt"))


@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    timeout=10800,  # 3 hours -- generous ceiling for a full-directory run;
                    # a small max_cases run finishes in a couple minutes and
                    # won't come close to this
)
def generate_synthetic_emidec(batch_size: int = 1, max_cases: int = 3):
    """Runs LeFusion's actual EMIDEC inference command (verified against the
    real emidec_inference.sh), pointed at the volume's data/weights.

    max_cases limits how many real cases get used as conditioning input for
    generation, by building a small subset folder (symlinks, not copies) with
    just the first N cases instead of pointing at the full ~67-case
    Pathological/ folder. Set to None to run the full directory. Confirmed
    from a real run: each case takes ~42-43 seconds (896-step diffusion loop
    at ~21 it/s on A10G), so the full directory is roughly 45-50 minutes --
    fine for a real production run, overkill for a first proof-of-concept
    where a handful of examples is enough to show the pipeline works.
    """
    import subprocess
    import os

    data_dir = f"{VOLUME_PATH}/LeFusion_data/EMIDEC"
    model_path = f"{VOLUME_PATH}/LeFusion_model/EMIDEC/emidec.pt"
    out_img = f"{VOLUME_PATH}/LeFusion_output/Image/"
    out_mask = f"{VOLUME_PATH}/LeFusion_output/Mask/"
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

    full_pathological = f"{data_dir}/Pathological"
    if max_cases is not None:
        subset_dir = f"{VOLUME_PATH}/LeFusion_subset_Pathological"
        subset_images = f"{subset_dir}/images"
        subset_labels = f"{subset_dir}/labels"
        os.makedirs(subset_images, exist_ok=True)
        os.makedirs(subset_labels, exist_ok=True)

        real_images_dir = f"{full_pathological}/images"
        all_case_files = sorted(os.listdir(real_images_dir))
        already_done = set(os.listdir(out_img)) if os.path.exists(out_img) else set()
        new_case_files = [f for f in all_case_files if f not in already_done]
        case_files = new_case_files[:max_cases]
        print(f"{len(already_done)} cases already generated, skipping those. "
              f"Adding {len(case_files)} new cases: {case_files}")
        for fname in case_files:
            for sub, dest_root in [("images", subset_images), ("labels", subset_labels)]:
                src = os.path.abspath(f"{full_pathological}/{sub}/{fname}")
                dst = f"{dest_root}/{fname}"
                if not os.path.exists(dst) and os.path.exists(src):
                    os.symlink(src, dst)
        dataset_root = subset_dir
        print(f"Using {len(case_files)} cases (max_cases={max_cases}): {case_files}")
    else:
        dataset_root = full_pathological
        print("Using the FULL Pathological directory (max_cases=None) -- "
              "this will take roughly 45-50 minutes based on the confirmed "
              "~43 sec/case rate.")

    cmd = [
        "python", "/root/LeFusion/LeFusion/inference/inference.py",
        "data_type=emidec",
        "types=1",
        "diffusion_img_size=72",
        "diffusion_depth_size=10",
        "diffusion_num_channels=2",
        f"dataset_root_dir={dataset_root}/",
        f"target_img_path={out_img}",
        f"target_label_path={out_mask}",
        "schedule_jump_params.jump_length=2",
        "schedule_jump_params.jump_n_sample=2",
        f"model_path={model_path}",
        "cond_dim=32",
        f"batch_size={batch_size}",
    ]
    # snapshot before running, so we can report only what THIS run actually
    # produced -- the previous version listed the whole output folder, which
    # included leftovers from earlier runs and made it look like 47 files got
    # generated when only 3 new ones actually had (confirmed by checking
    # timestamps directly: the other 44 were from an earlier killed run and
    # an old test).
    files_before = set(os.listdir(out_img)) if os.path.exists(out_img) else set()

    print("Running:", " ".join(cmd))

    # stream output live instead of capturing silently until the process
    # ends -- the previous version showed nothing at all while running,
    # which is why there was no visible progress. This prints each line
    # as it's produced, so any progress bar or status print from the
    # underlying script shows up in real time in the Modal log view.
    all_lines = []
    process = subprocess.Popen(
        cmd, cwd="/root/LeFusion", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
        all_lines.append(line)
    process.wait()
    combined_output = "".join(all_lines)

    volume.commit()
    files_after = set(os.listdir(out_img)) if os.path.exists(out_img) else set()
    newly_generated = sorted(files_after - files_before)
    return {"returncode": process.returncode,
            "newly_generated_files": newly_generated,
            "total_files_in_output_folder": len(files_after),
            "output_tail": combined_output[-2000:]}


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=300,
)
def analyze_synthetic_geometry(case_id: str, slice_idx: int = None):
    """Runs the SAME angular coverage + contiguity math already tested and
    used on real EMIDEC cases (verify_slice_claims.py), applied to a
    synthetic case instead, so the numbers are directly comparable to the
    real-case baselines (P019: infarct contiguity 0.823, MVO 0.817; P004:
    infarct contiguity 0.861)."""
    import os
    import numpy as np
    import nibabel as nib

    img_path = f"{VOLUME_PATH}/LeFusion_output/Image/{case_id}"
    mask_path = f"{VOLUME_PATH}/LeFusion_output/Mask/{case_id}"
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return {"error": f"Case not found: {case_id}"}

    img_data = np.asarray(nib.load(img_path).get_fdata())
    mask_data = np.asarray(nib.load(mask_path).get_fdata()).round().astype(int)

    # same axis-order fix confirmed necessary in render_synthetic_case
    if img_data.shape != mask_data.shape and sorted(img_data.shape) == sorted(mask_data.shape):
        remaining = list(range(mask_data.ndim))
        perm = []
        for target_size in img_data.shape:
            for ax in remaining:
                if mask_data.shape[ax] == target_size:
                    perm.append(ax)
                    remaining.remove(ax)
                    break
        mask_data = np.transpose(mask_data, perm)

    LABEL_MAP = {"background": 0, "lv_cavity": 1, "myocardium_normal": 2, "infarct": 3, "mvo": 4}

    def angular_coverage(mask_2d, label, center):
        ys, xs = np.where(mask_2d == label)
        if len(ys) == 0:
            return {"present": False, "coverage_deg": 0.0, "angles_deg": []}
        cy, cx = center
        angles = np.degrees(np.arctan2(ys - cy, xs - cx)) % 360
        angles_sorted = np.sort(np.unique(np.round(angles).astype(int)) % 360)
        if len(angles_sorted) == 1:
            return {"present": True, "coverage_deg": 1.0, "angles_deg": angles_sorted.tolist(),
                     "arc_start_deg": float(angles_sorted[0]), "arc_end_deg": float(angles_sorted[0])}
        gaps = np.diff(angles_sorted)
        wrap_gap = 360 - (angles_sorted[-1] - angles_sorted[0])
        all_gaps = np.append(gaps, wrap_gap)
        max_gap_idx = np.argmax(all_gaps)
        if max_gap_idx == len(gaps):
            arc_start, arc_end = angles_sorted[0], angles_sorted[-1]
        else:
            arc_start = angles_sorted[max_gap_idx + 1]
            arc_end = angles_sorted[max_gap_idx]
        coverage = 360 - all_gaps[max_gap_idx]
        return {"present": True, "coverage_deg": float(coverage),
                "coverage_pct_of_circle": float(coverage / 360 * 100),
                "arc_start_deg": float(arc_start), "arc_end_deg": float(arc_end),
                "angles_deg": angles_sorted.tolist()}

    def contiguity_score(mask_2d, label, center):
        cov = angular_coverage(mask_2d, label, center)
        if not cov["present"] or cov["coverage_deg"] <= 1:
            return {"present": cov["present"], "contiguity": None}
        angles_present = set(cov["angles_deg"])
        arc_start = cov["arc_start_deg"]
        span = int(round(cov["coverage_deg"]))
        covered_count = 0
        for i in range(span + 1):
            deg = int(round(arc_start + i)) % 360
            if deg in angles_present:
                covered_count += 1
        return {"present": True, "contiguity": round(covered_count / (span + 1), 3),
                "coverage_deg": cov["coverage_deg"],
                "degrees_with_label": covered_count, "degrees_in_span": span + 1}

    if slice_idx is None:
        # target pathology specifically (infarct + MVO), not just any
        # foreground -- picking by total foreground let cavity/myocardium
        # dominate the choice, which could skip past a slice that actually
        # has infarct/MVO if another slice happened to have more total
        # anatomy pixels. Confirmed necessary after Case_P003 came back
        # "not present" on the auto-picked slice.
        pathology_per_slice = [
            ((mask_data[:, :, s] == LABEL_MAP["infarct"]) | (mask_data[:, :, s] == LABEL_MAP["mvo"])).sum()
            for s in range(mask_data.shape[2])
        ]
        if max(pathology_per_slice) == 0:
            return {"error": f"No infarct or MVO found on ANY slice for {case_id} "
                              f"(genuinely no pathology in this synthetic case, not a slice-picking issue)"}
        slice_idx = int(np.argmax(pathology_per_slice))

    mask_2d = mask_data[:, :, slice_idx]
    cavity_ys, cavity_xs = np.where(mask_2d == LABEL_MAP["lv_cavity"])
    if len(cavity_ys) == 0:
        return {"error": f"No LV cavity found on slice {slice_idx}"}
    center = (cavity_ys.mean(), cavity_xs.mean())

    result = {"case_id": case_id, "slice_used": slice_idx}
    for region_name, label_key in [("infarct", "infarct"), ("mvo", "mvo")]:
        cov = angular_coverage(mask_2d, LABEL_MAP[label_key], center)
        if not cov["present"]:
            result[region_name] = {"present": False}
            continue
        cscore = contiguity_score(mask_2d, LABEL_MAP[label_key], center)
        result[region_name] = {
            "present": True,
            "coverage_pct_of_circle": round(cov["coverage_pct_of_circle"], 1),
            "contiguity": cscore["contiguity"],
        }
    return result


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=300,
)
def scan_all_slices_geometry(case_id: str):
    """Runs the same contiguity math across EVERY slice that has infarct or
    MVO, not just one auto-picked slice -- so we can tell whether a low
    contiguity score is a consistent pattern across the whole case, or just
    one unlucky slice, before treating it as a real outlier."""
    import os
    import numpy as np
    import nibabel as nib

    mask_path = f"{VOLUME_PATH}/LeFusion_output/Mask/{case_id}"
    img_path = f"{VOLUME_PATH}/LeFusion_output/Image/{case_id}"
    if not os.path.exists(mask_path):
        return {"error": f"Case not found: {case_id}"}

    mask_data = np.asarray(nib.load(mask_path).get_fdata()).round().astype(int)
    img_data = np.asarray(nib.load(img_path).get_fdata())
    if img_data.shape != mask_data.shape and sorted(img_data.shape) == sorted(mask_data.shape):
        remaining = list(range(mask_data.ndim))
        perm = []
        for target_size in img_data.shape:
            for ax in remaining:
                if mask_data.shape[ax] == target_size:
                    perm.append(ax)
                    remaining.remove(ax)
                    break
        mask_data = np.transpose(mask_data, perm)

    LABEL_MAP = {"background": 0, "lv_cavity": 1, "myocardium_normal": 2, "infarct": 3, "mvo": 4}

    def angular_coverage(mask_2d, label, center):
        ys, xs = np.where(mask_2d == label)
        if len(ys) == 0:
            return {"present": False}
        cy, cx = center
        angles = np.degrees(np.arctan2(ys - cy, xs - cx)) % 360
        angles_sorted = np.sort(np.unique(np.round(angles).astype(int)) % 360)
        if len(angles_sorted) == 1:
            return {"present": True, "coverage_deg": 1.0, "angles_deg": angles_sorted.tolist(),
                     "arc_start_deg": float(angles_sorted[0])}
        gaps = np.diff(angles_sorted)
        wrap_gap = 360 - (angles_sorted[-1] - angles_sorted[0])
        all_gaps = np.append(gaps, wrap_gap)
        max_gap_idx = np.argmax(all_gaps)
        if max_gap_idx == len(gaps):
            arc_start, arc_end = angles_sorted[0], angles_sorted[-1]
        else:
            arc_start = angles_sorted[max_gap_idx + 1]
            arc_end = angles_sorted[max_gap_idx]
        coverage = 360 - all_gaps[max_gap_idx]
        return {"present": True, "coverage_deg": float(coverage), "arc_start_deg": float(arc_start),
                "angles_deg": angles_sorted.tolist()}

    def contiguity_score(mask_2d, label, center):
        cov = angular_coverage(mask_2d, label, center)
        if not cov["present"] or cov["coverage_deg"] <= 1:
            return None
        angles_present = set(cov["angles_deg"])
        span = int(round(cov["coverage_deg"]))
        covered_count = sum(1 for i in range(span + 1)
                             if int(round(cov["arc_start_deg"] + i)) % 360 in angles_present)
        return round(covered_count / (span + 1), 3)

    per_slice_results = []
    for s in range(mask_data.shape[2]):
        mask_2d = mask_data[:, :, s]
        cavity_ys, cavity_xs = np.where(mask_2d == LABEL_MAP["lv_cavity"])
        if len(cavity_ys) == 0:
            continue
        center = (cavity_ys.mean(), cavity_xs.mean())
        infarct_present = (mask_2d == LABEL_MAP["infarct"]).any()
        if not infarct_present:
            continue
        cscore = contiguity_score(mask_2d, LABEL_MAP["infarct"], center)
        per_slice_results.append({"slice": s, "infarct_contiguity": cscore})

    return {"case_id": case_id, "total_slices": mask_data.shape[2], "slices_with_infarct": per_slice_results}


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=300,
)
def compare_synthetic_to_real_mask(case_id: str):
    """Directly checks whether the 'synthetic' mask is essentially the real
    conditioning mask, resampled to LeFusion's fixed generation resolution,
    rather than genuinely new synthetic geometry. Fixes two issues found in
    the first version of this check: (1) the synthetic mask's axis order
    needed correcting first (same [D,H,W] vs [H,W,D] issue found earlier),
    and (2) the real mask is at its own native resolution (confirmed 62x65
    for P004) while the synthetic output is fixed at 72x72 -- a raw pixel
    comparison can never match without resampling one to the other first."""
    import os
    import numpy as np
    import nibabel as nib
    from scipy.ndimage import zoom

    synth_mask_path = f"{VOLUME_PATH}/LeFusion_output/Mask/{case_id}"
    real_mask_path = f"{VOLUME_PATH}/LeFusion_data/EMIDEC/Pathological/labels/{case_id}"

    if not os.path.exists(synth_mask_path):
        return {"error": f"Synthetic mask not found: {synth_mask_path}"}
    if not os.path.exists(real_mask_path):
        return {"error": f"Real mask not found: {real_mask_path}"}

    synth_data = np.asarray(nib.load(synth_mask_path).get_fdata()).round().astype(int)
    real_data = np.asarray(nib.load(real_mask_path).get_fdata()).round().astype(int)

    # fix axis order on the synthetic mask first, same issue confirmed earlier
    if sorted(synth_data.shape) == sorted((10, 72, 72)) and synth_data.shape != (72, 72, 10):
        depth_axis = synth_data.shape.index(min(synth_data.shape))
        order = [ax for ax in range(3) if ax != depth_axis] + [depth_axis]
        synth_data = np.transpose(synth_data, order)

    result = {
        "case_id": case_id,
        "synthetic_shape_after_axis_fix": list(synth_data.shape),
        "real_shape_native": list(real_data.shape),
    }

    # resample the real mask (nearest-neighbor -- correct for label/categorical
    # data, unlike linear interpolation which would invent fractional label
    # values that don't correspond to any real class) to match the
    # synthetic's spatial resolution, so the comparison is apples-to-apples
    zoom_factors = [s / r for s, r in zip(synth_data.shape, real_data.shape)]
    real_resampled = zoom(real_data, zoom_factors, order=0)
    result["real_shape_after_resampling"] = list(real_resampled.shape)

    if real_resampled.shape == synth_data.shape:
        matching_fraction = float((real_resampled == synth_data).mean())
        result["fraction_matching_pixels_after_resampling"] = round(matching_fraction, 4)
        result["essentially_identical"] = matching_fraction > 0.90
        result["real_label_distribution"] = {int(l): int((real_resampled == l).sum()) for l in np.unique(real_resampled)}
        result["synthetic_label_distribution"] = {int(l): int((synth_data == l).sum()) for l in np.unique(synth_data)}
    else:
        result["error_after_resampling"] = f"shapes still don't match: {real_resampled.shape} vs {synth_data.shape}"

    return result


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=300,
)
def render_synthetic_case(case_id: str, slice_idx: int = None):
    """Loads one synthetic case's image + mask from the volume, inspects the
    REAL label values present (rather than assuming EMIDEC's 4-class scheme
    applies to LeFusion's output, since LeFusion is lesion-focused and might
    only produce infarct/MVO labels, not full cavity+myocardium), and renders
    an overlay PNG. Returns the PNG bytes directly so the caller can write
    them to local disk without needing a separate volume download step."""
    import os
    import io
    import numpy as np
    import nibabel as nib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    img_path = f"{VOLUME_PATH}/LeFusion_output/Image/{case_id}"
    mask_path = f"{VOLUME_PATH}/LeFusion_output/Mask/{case_id}"

    if not os.path.exists(img_path):
        return {"error": f"Image not found: {img_path}"}
    if not os.path.exists(mask_path):
        return {"error": f"Mask not found: {mask_path}"}

    img_data = np.asarray(nib.load(img_path).get_fdata())
    mask_data = np.asarray(nib.load(mask_path).get_fdata()).round().astype(int)

    # confirmed via a real run: mask and image can have the SAME dimensions
    # but in a DIFFERENT axis order (image was HxWxD = 72x72x10, mask was
    # DxHxW = 10x72x72) -- not actually a resolution mismatch at all, just a
    # transposed array. Slicing axis 2 on both, as if they shared the same
    # convention, sliced through the wrong axis on the mask (depth x height
    # at a fixed width position), producing the thin vertical-strip artifact
    # seen in the first real render. Detect and correct the transpose here
    # rather than assuming both arrays use the same axis order.
    if img_data.shape != mask_data.shape and sorted(img_data.shape) == sorted(mask_data.shape):
        # find the permutation of mask_data's axes that matches img_data's shape
        remaining = list(range(mask_data.ndim))
        perm = []
        for target_size in img_data.shape:
            for ax in remaining:
                if mask_data.shape[ax] == target_size:
                    perm.append(ax)
                    remaining.remove(ax)
                    break
        mask_data = np.transpose(mask_data, perm)
        print(f"NOTE: mask array was transposed (axis order {perm}) to align with "
              f"the image's axis order -- shapes matched after transpose: "
              f"{mask_data.shape} == {img_data.shape}")

    unique_labels = sorted(np.unique(mask_data).tolist())
    print(f"Image shape: {img_data.shape}, dtype-range: [{img_data.min():.2f}, {img_data.max():.2f}]")
    print(f"Mask shape: {mask_data.shape}, unique label values found: {unique_labels}")

    if slice_idx is None:
        # pick the slice with the most non-background mask pixels, so the
        # rendered example actually shows something, not an empty slice
        non_bg_per_slice = [(mask_data[:, :, s] != 0).sum() for s in range(mask_data.shape[2])]
        mask_slice_idx = int(np.argmax(non_bg_per_slice))
        print(f"Auto-picked mask slice {mask_slice_idx} (most non-background label pixels: {max(non_bg_per_slice)})")
    else:
        mask_slice_idx = slice_idx

    # confirmed via a real run: image and mask can have DIFFERENT depths
    # (mask had 34 slices, image had 10 -- likely the mask preserves the
    # real conditioning case's original resolution while the synthetic
    # image is fixed at the diffusion model's generation size). Map the
    # chosen slice proportionally instead of assuming they match, so the
    # two panels still show roughly the same anatomical position.
    if img_data.shape[2] != mask_data.shape[2]:
        frac = mask_slice_idx / max(mask_data.shape[2] - 1, 1)
        img_slice_idx = int(round(frac * (img_data.shape[2] - 1)))
        print(f"NOTE: image depth ({img_data.shape[2]}) != mask depth ({mask_data.shape[2]}) -- "
              f"mapping mask slice {mask_slice_idx} proportionally to image slice {img_slice_idx}")
    else:
        img_slice_idx = mask_slice_idx

    img_slice = img_data[:, :, img_slice_idx]
    mask_slice = mask_data[:, :, mask_slice_idx]

    # generic color mapping for whatever labels are actually present --
    # doesn't assume EMIDEC's specific 4-class scheme, since we don't yet
    # know if LeFusion's synthetic masks follow it
    palette = ["#E24A4A", "#E2C34A", "#4A90D9", "#7ED957", "#B366D9", "#D98A4A"]
    nonzero_labels = [l for l in unique_labels if l != 0]
    label_colors = {l: palette[i % len(palette)] for i, l in enumerate(nonzero_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), facecolor="white")
    axes[0].imshow(img_slice.T, cmap="gray", origin="lower")
    axes[0].set_title(f"{case_id} \u2014 synthetic image\nslice {img_slice_idx}", fontsize=10.5)
    axes[0].axis("off")

    axes[1].imshow(img_slice.T, cmap="gray", origin="lower")
    overlay = np.zeros((*mask_slice.T.shape, 4))
    for label, color in label_colors.items():
        rgb = tuple(int(color[i:i+2], 16) / 255 for i in (1, 3, 5))
        overlay[(mask_slice.T == label)] = (*rgb, 0.55)
    axes[1].imshow(overlay, origin="lower")
    axes[1].set_title(f"{case_id} \u2014 synthetic mask overlay\nlabels found: {unique_labels}", fontsize=10.5)
    axes[1].axis("off")

    legend_elements = [mpatches.Patch(facecolor=color, label=f"label {label}")
                        for label, color in label_colors.items()]
    if legend_elements:
        axes[1].legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("LeFusion synthetic EMIDEC-style case (pretrained weights, first generation)",
                 fontsize=11.5, fontweight="bold", y=1.02)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()
    buf.seek(0)

    return {
        "png_bytes": buf.read(),
        "image_shape": list(img_data.shape),
        "unique_mask_labels": unique_labels,
        "mask_slice_used": mask_slice_idx,
        "image_slice_used": img_slice_idx,
        "mask_shape": list(mask_data.shape),
    }


# NOTE: there is deliberately no @app.local_entrypoint() / "modal run" workflow
# here anymore. `modal run` creates a temporary app that gets torn down the
# moment the local script finishes -- which cancels any spawned job too, even
# though .spawn() is meant to detach from the local connection specifically.
# Confirmed via a real run: the job showed Status=Cancelled, 0 containers ever
# went live, no logs at all, right after "local entrypoint completed" printed.
#
# Correct workflow instead:
#   modal deploy modal_lefusion.py
#   python trigger_lefusion.py
# (trigger_lefusion.py connects to the DEPLOYED, persistent app and spawns
# the job there, so it survives regardless of what the local script does.)
