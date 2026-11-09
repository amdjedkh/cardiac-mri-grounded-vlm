"""
medgemma_100.py

MedGemma-4B zero-shot on the 100 EMIDEC cases with GEMINI'S EXACT PROMPT TEXT and the same images.
The system text, user text and image pair (plain + overlay of the middle slice, same files as the Gemini run)
come from report_generation/prompts_region_grounded.gemini_region_grounded_report, unchanged. Only the model
differs (so MedGemma's own chat template is applied by the processor).

Settings (stored in every output record): google/medgemma-4b-it at a recorded revision hash, bf16, greedy
decoding (do_sample False), max_new_tokens from config.MEDGEMMA_MAX_NEW_TOKENS (1024), A10G, no fallback model.
Output: EMIDEC_ROOT/run_medgemma_100/outputs_region_grounded/<case>.json (a new folder; no other folder is touched).
Cost guard: the remote loop stops once MAX_SECONDS of generation have passed (A10G rate in config below is an
unverified assumption); hard cap 3 USD.

    modal run experiments/medgemma_100.py --cases Case_N006,Case_P001,Case_P002     # the 3-case check
    modal run experiments/medgemma_100.py --revision <sha>                          # the other cases, same revision
"""

import os
import sys
import json
import glob
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path[:0] = [os.path.join(REPO, "report_generation"), REPO]
import config

A10G_USD_PER_HOUR = 1.10   # ASSUMPTION from memory, not verified against Modal billing
HARD_CAP_USD = 3.0
MAX_SECONDS = int(HARD_CAP_USD / A10G_USD_PER_HOUR * 3600 * 0.85)   # stop with margin for the cold start

app = modal.App("medgemma-emidec-100")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers>=4.50.0", "accelerate", "pillow", "huggingface_hub")
         .add_local_file(os.path.join(REPO, "config.py"), "/root/config.py"))


@app.function(image=image, gpu="A10G", timeout=3 * 3600, secrets=[modal.Secret.from_name("huggingface-secret")])
def run(items, revision, max_seconds):
    import io
    import torch
    import transformers
    from huggingface_hub import HfApi
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    mid = config.MEDGEMMA_MODEL_ID
    sha = HfApi().model_info(mid, revision=revision or None).sha
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(mid, revision=sha)
    model = AutoModelForImageTextToText.from_pretrained(mid, revision=sha, dtype=torch.bfloat16, device_map="cuda")
    info = {"model_id": mid, "revision": sha, "dtype": "bfloat16", "do_sample": config.MEDGEMMA_DO_SAMPLE,
            "max_new_tokens": config.MEDGEMMA_MAX_NEW_TOKENS, "gpu": "A10G",
            "torch": str(torch.__version__), "transformers": str(transformers.__version__), "load_seconds": round(time.time() - t0, 1)}
    out, t_loop = [], time.time()
    for it in items:
        if time.time() - t_loop > max_seconds:
            out.append({"case": it["case"], "skipped": "time cap reached"})
            continue
        imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in it["images"]]
        content = [{"type": "text", "text": it["user_text"]}] + [{"type": "image", "image": im} for im in imgs]
        messages = [{"role": "system", "content": [{"type": "text", "text": it["system"]}]},
                    {"role": "user", "content": content}]
        inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
        n_in = inputs["input_ids"].shape[-1]
        t = time.time()
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=config.MEDGEMMA_MAX_NEW_TOKENS, do_sample=config.MEDGEMMA_DO_SAMPLE)
        new = gen[0][n_in:]
        out.append({"case": it["case"], "output": proc.decode(new, skip_special_tokens=True),
                    "seconds": round(time.time() - t, 2), "input_tokens": int(n_in), "new_tokens": int(len(new)),
                    "hit_max_new_tokens": bool(len(new) >= config.MEDGEMMA_MAX_NEW_TOKENS)})
    info["container_seconds"] = round(time.time() - t0, 1)
    return {"info": info, "results": out}


@app.local_entrypoint()
def main(cases: str = "", revision: str = ""):
    from prompts_region_grounded import gemini_region_grounded_report, check_region_grounding
    root = config.EMIDEC_ROOT
    all_ids = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(root, "patients", "*.json")))
    assert len(all_ids) == 100, len(all_ids)
    out_dir = os.path.join(root, "run_medgemma_100", "outputs_region_grounded")
    os.makedirs(out_dir, exist_ok=True)
    ids = cases.split(",") if cases else [c for c in all_ids if not os.path.exists(os.path.join(out_dir, f"{c}.json"))]
    items = []
    for c in ids:
        record = json.load(open(os.path.join(root, "patients", f"{c}.json")))
        d = os.path.join(root, "overlays", c)
        plains, overlays = sorted(glob.glob(os.path.join(d, "*_plain.png"))), sorted(glob.glob(os.path.join(d, "*_overlay.png")))
        m = len(plains) // 2   # same slice rule as run_region_grounded_poc.find_one_overlay_pair
        p = gemini_region_grounded_report(record, [plains[m]], [overlays[m]])
        items.append({"case": c, "system": p["system"], "user_text": p["user_text"], "slice_used": os.path.relpath(plains[m], root),
                      "images": [open(x, "rb").read() for x in p["image_paths"]]})
    print(f"{len(items)} cases, revision {revision or '(latest, will be recorded)'}, time cap {MAX_SECONDS} s")
    res = run.remote(items, revision, MAX_SECONDS)
    info = res["info"]
    by = {it["case"]: it for it in items}
    for r in res["results"]:
        if "output" not in r:
            print(r["case"], r["skipped"])
            continue
        it = by[r["case"]]
        rec = {"patient_id": r["case"], "model": info["model_id"], "slice_used": it["slice_used"], "output": r["output"],
               "grounding_check": check_region_grounding(r["output"]),
               "generation_settings": {**info, "prompt_source": "report_generation/prompts_region_grounded.gemini_region_grounded_report (unchanged)"},
               "seconds": r["seconds"], "input_tokens": r["input_tokens"], "new_tokens": r["new_tokens"],
               "hit_max_new_tokens": r["hit_max_new_tokens"]}
        json.dump(rec, open(os.path.join(out_dir, f"{r['case']}.json"), "w"), indent=2)
        print(f"{r['case']}: {r['seconds']} s, {r['new_tokens']} new tokens, hit max {r['hit_max_new_tokens']}")
    print("settings:", json.dumps(info))
    print(f"container seconds {info['container_seconds']} -> estimated cost ${info['container_seconds'] / 3600 * A10G_USD_PER_HOUR:.2f} "
          f"at an UNVERIFIED ${A10G_USD_PER_HOUR}/h (check the Modal billing page)")
