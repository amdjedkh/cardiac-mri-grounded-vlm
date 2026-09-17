"""
modal_physician_tool.py

Deploys physician_tool.py as a live, password-protected web app on Modal.

This file did not previously exist anywhere persistent -- it was run
ad-hoc from a terminal and never saved, which is why "redeploying" did
nothing (there was no file to redeploy). Reconstructed here from scratch
and committed to the repo so this can't happen again.

Depends on the following pre-existing account-level Modal resources
(confirmed against the real deployed objects, not assumed):
  - A volume named "physician-tool-data", containing (at its root):
      real_cases/                (Case_* folders, each with Images/ + Contours/;
                                   confirmed populated, 100 cases)
      synthetic_cases/           (optional -- confirmed MISSING as of last
                                   check; app handles this gracefully, Module 1
                                   just shows zero synthetic cases)
      outputs_region_grounded/   (report JSONs; confirmed populated, 100 files)
      overlays/                  (rendered slice PNGs, Case_*/slice_NN_*.png;
                                   confirmed populated, 100 cases)
  - A secret named "physician-tool-auth", containing two keys:
      BASIC_AUTH_USER and BASIC_AUTH_PASSWORD
    (NOT "USERNAME"/"PASSWORD" -- an earlier reconstruction of this file
    guessed those names, which deployed "successfully" but then crashed
    every request with KeyError('USERNAME'), which looked like the app
    hanging/infinite-loading in a browser rather than an obvious error.
    Confirmed the real names via a read-only key-listing check against
    the deployed secret, not by guessing again.)

If the volume or secret is missing entirely, this deploy will fail
clearly. If the volume exists but folders are missing/misnamed, the app
will silently start with zero cases -- run the "inspect volume contents"
check below BEFORE assuming the code is wrong.

Usage:
    modal deploy modal_physician_tool.py
"""

import base64
import os

import modal

app = modal.App("physician-tool")

VOLUME_PATH = "/data"
volume = modal.Volume.from_name("physician-tool-data", create_if_missing=False)
auth_secret = modal.Secret.from_name("physician-tool-auth")

PHYSICIAN_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("flask", "nibabel", "numpy", "matplotlib", "scipy")
    .add_local_file(
        os.path.join(PHYSICIAN_TOOL_DIR, "physician_tool.py"),
        "/root/physician_tool.py",
    )
    .add_local_dir(
        os.path.join(PHYSICIAN_TOOL_DIR, "templates"),
        "/root/templates",
    )
)


def make_basic_auth_wsgi(flask_wsgi_app, username, password):
    """Wraps a Flask app's WSGI callable with HTTP Basic Auth. Kept
    self-contained here rather than editing physician_tool.py itself,
    since that file is under active, separate development."""
    def wrapped(environ, start_response):
        auth_header = environ.get("HTTP_AUTHORIZATION", "")
        valid = False
        if auth_header.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
                user, _, pw = decoded.partition(":")
                valid = (user == username and pw == password)
            except Exception:
                valid = False
        if not valid:
            start_response("401 Unauthorized", [("WWW-Authenticate", 'Basic realm="Physician Tool"')])
            return [b"Authentication required"]
        return flask_wsgi_app(environ, start_response)
    return wrapped


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=[auth_secret],
    timeout=600,
)
@modal.wsgi_app()
def flask_app():
    import sys
    sys.path.insert(0, "/root")

    # point the app at the mounted volume instead of a local working
    # directory -- these env vars take priority in physician_tool.py's
    # own path resolution (checked before falling back to cwd)
    os.environ["REAL_CASES_DIR"] = f"{VOLUME_PATH}/real_cases"
    os.environ["SYNTHETIC_CASES_DIR"] = f"{VOLUME_PATH}/synthetic_cases"
    os.environ["REPORTS_DIR"] = f"{VOLUME_PATH}/outputs_region_grounded"
    os.environ["REPORT_OVERLAYS_DIR"] = f"{VOLUME_PATH}/overlays"

    # physician_tool.py's templates need to be found relative to itself
    os.chdir("/root")

    import physician_tool as pt

    # confirmed via a read-only key-name check against the deployed secret
    # (KeyError on first deploy showed the reconstructed guess was wrong):
    # the actual keys are BASIC_AUTH_USER / BASIC_AUTH_PASSWORD
    username = os.environ["BASIC_AUTH_USER"]
    password = os.environ["BASIC_AUTH_PASSWORD"]
    return make_basic_auth_wsgi(pt.app.wsgi_app, username, password)


@app.function(image=image, volumes={VOLUME_PATH: volume})
def inspect_volume_contents():
    """Run this FIRST if the deployed app shows zero cases -- confirms
    what's actually on the volume before assuming the code is wrong."""
    for sub in ["real_cases", "synthetic_cases", "outputs_region_grounded", "overlays"]:
        path = f"{VOLUME_PATH}/{sub}"
        if os.path.isdir(path):
            contents = os.listdir(path)
            print(f"{sub}/  ({len(contents)} items): {contents[:5]}{'...' if len(contents) > 5 else ''}")
        else:
            print(f"{sub}/  MISSING")


@app.local_entrypoint()
def check():
    inspect_volume_contents.remote()
