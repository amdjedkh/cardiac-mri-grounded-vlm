"""
modal_medgemma.py

Deploys MedGemma (google/medgemma-4b-it, the multimodal instruction-tuned
variant) as a Modal function on your existing A10G setup, for the two
MedGemma baseline conditions.

Setup (one-time):
    1. Request access to MedGemma on Hugging Face (it's gated):
       https://huggingface.co/google/medgemma-4b-it
    2. Create a Modal secret with your HF token:
       modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
    3. Deploy:
       modal deploy modal_medgemma.py

Usage from run_medgemma.py (local script):
    the Modal function is called remotely per patient -- see run_medgemma.py
"""

import modal

app = modal.App("medgemma-emidec-poc")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.50.0",
        "accelerate",
        "pillow",
    )
)

MODEL_ID = "google/medgemma-4b-it"


@app.cls(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,  # keep warm 5 min after last call, avoids reloading between patients
)
class MedGemma:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
        )

    @modal.method()
    def generate(self, system: str, user_text: str, image_bytes_list: list) -> str:
        import io
        import torch
        from PIL import Image

        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]

        content = [{"type": "text", "text": user_text}]
        for img in images:
            content.append({"type": "image", "image": img})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content},
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to("cuda", dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated = output[0][input_len:]
        text = self.processor.decode(generated, skip_special_tokens=True)
        return text


@app.local_entrypoint()
def test():
    """Quick smoke test: modal run modal_medgemma.py"""
    m = MedGemma()
    result = m.generate.remote(
        system="You are a helpful assistant.",
        user_text="Say hello in one sentence.",
        image_bytes_list=[],
    )
    print("Smoke test output:", result)
