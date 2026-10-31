"""
config.py

Single place for seeds and model settings. Scripts import from here instead
of hardcoding values. To use from a script in a subfolder:

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import config

Modal apps ship this file into their containers (see modal_lefusion.py,
modal_medgemma.py).
"""

# ---------------------------------------------------------------- seeds
SPLIT_SEED = 42            # patient-level train/test split (splits/make_split.py)
TEST_FRACTION = 0.20       # share of each stratum held out as the test set
GEMINI_SEED = 42           # sent with every Gemini request
MASK_DIVERSITY_SEED = 42   # lesion augmentation (synthetic_generation/mask_diversity.py)

# LeFusion's inference.py has no seed option (no "seed" anywhere in the
# HINTLab/LeFusion code, checked 2026-10-29), so synthetic generation cannot be
# made repeatable from config. Kept here as None so this is visible, not silent.
LEFUSION_SEED = None

# ------------------------------------------------- Gemini (report generation)
# Pinned. There is NO fallback to other models: if this model is unavailable
# the run fails. 95 of the first 100 reports were made with this model; 5
# (Case_N006, P001, P002, P004, P007) came from gemini-3.5-flash through the
# old fallback list and should be regenerated for a consistent set.
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_TEMPERATURE = 0.0
GEMINI_MAX_RETRIES = 6     # retries on the SAME model for 429/503 errors only
# "vertex" bills the Google Cloud project (login: gcloud auth application-default login);
# "api_key" uses the Gemini Developer API with GEMINI_API_KEY from the environment.
GEMINI_BACKEND = "vertex"
GCP_PROJECT = "alpine-myth-513108-k7"
GCP_LOCATION = "global"

# ------------------------------------------------------ MedGemma (baseline)
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
MEDGEMMA_MAX_NEW_TOKENS = 1024
MEDGEMMA_DO_SAMPLE = False  # greedy decoding, so no seed is needed

# ----------------------------------------------------- LeFusion (synthetic)
LEFUSION_GPU = "A100"
LEFUSION_BATCH_SIZE = 1
LEFUSION_DIFFUSION_IMG_SIZE = 72
LEFUSION_DIFFUSION_DEPTH_SIZE = 10
LEFUSION_DIFFUSION_NUM_CHANNELS = 2
LEFUSION_COND_DIM = 32
LEFUSION_TYPES = 1
LEFUSION_JUMP_LENGTH = 2
LEFUSION_JUMP_N_SAMPLE = 2

# ------------------------------------- mask diversity (lesion augmentation)
MASK_ROTATION_MAX_DEG = 60            # lesion rotates around the wall by up to +-this
MASK_ANGULAR_SCALE_RANGE = (0.80, 1.25)   # stretch/shrink of the lesion's angular extent
MASK_DEPTH_SCALE_RANGE = (0.85, 1.15)     # stretch/shrink of the lesion's wall depth
MASK_SLICE_SHIFT_MAX = 1              # lesion moves up to +-this many slices
MASK_AREA_RATIO_RANGE = (0.5, 1.5)    # allowed new/old area for infarct, MVO and total lesion
MASK_LEAVE_TOLERANCE = 0.02           # max fraction of lesion pixels allowed to leave the wall
MASK_NEIGHBOUR_TOL_PX = 1.5           # max distance (px) between a target pixel and its source pixel
