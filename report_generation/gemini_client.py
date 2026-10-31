"""
gemini_client.py

The one Gemini call used by every report runner. Model, temperature and seed
come from config.py. The model is pinned: there is no fallback list. A
rate-limit or temporary-unavailable error (not 402/403 billing errors) is retried on the SAME model; any
other error, or running out of retries, raises.
"""

import os
import sys
import time

from google import genai
from google.genai import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config


def make_client():
    """Client for the backend chosen in config.py. The key is never printed."""
    if config.GEMINI_BACKEND == "vertex":
        return genai.Client(vertexai=True, project=config.GCP_PROJECT, location=config.GCP_LOCATION)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (config.GEMINI_BACKEND is 'api_key').")
    return genai.Client(api_key=api_key)


def generation_settings() -> dict:
    """Stored in every report JSON so a result can be traced to its settings."""
    return {
        "model": config.GEMINI_MODEL,
        "temperature": config.GEMINI_TEMPERATURE,
        "seed": config.GEMINI_SEED,
        "backend": config.GEMINI_BACKEND,
    }


def call_gemini(client, system: str, user_text: str, image_paths: list, max_retries: int = None):
    """Returns (text, model_name). Raises if the pinned model keeps failing."""
    max_retries = max_retries or config.GEMINI_MAX_RETRIES
    parts = [types.Part.from_text(text=user_text)]
    for p in image_paths:
        with open(p, "rb") as f:
            parts.append(types.Part.from_bytes(data=f.read(), mime_type="image/png"))

    gen_config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=config.GEMINI_TEMPERATURE,
        seed=config.GEMINI_SEED,
    )
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[types.Content(role="user", parts=parts)],
                config=gen_config,
            )
            return response.text, config.GEMINI_MODEL
        except Exception as e:
            msg = str(e)
            # 402/403 are billing or permission errors: retrying cannot help
            if any(code in msg[:12] for code in ("402", "403")):
                raise
            if any(code in msg for code in ("RESOURCE_EXHAUSTED", "429", "UNAVAILABLE", "503")):
                wait = min(30 * (attempt + 1), 120)
                print(f"    [{config.GEMINI_MODEL}] rate limited, retrying in {wait}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"{config.GEMINI_MODEL} still failing after {max_retries} retries "
                       "(no fallback model is configured, by design)")
