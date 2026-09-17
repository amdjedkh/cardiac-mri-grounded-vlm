"""
modal_physician_tool.py

Deploys physician_tool.py as a live, password-protected web app on Modal.

This file did not previously exist anywhere persistent -- it was run
ad-hoc from a terminal and never saved, which is why "redeploying" did
nothing (there was no file to redeploy). Reconstructed here from scratch
and committed to the repo so this can't happen again.

ASSUMES the following already exist on your Modal account, from the
original deployment (these are account-level resources, independent of
this file, so they should still exist even though the file was lost):
  - A volume named "physician-tool-data", containing (at its root):
      real_cases/         (Case_* folders, each with Images/ + Contours/)
      outputs_region_grounded/   (report JSONs)
      overlays/           (rendered slice PNGs, Case_*/slice_NN_*.png)
  - A secret named "physician-tool-auth", containing two keys:
      USERNAME and PASSWORD

If either doesn't exist, this deploy will fail clearly (missing secret)
or the app will start with zero cases found (missing/misnamed volume
folders) -- in the latter case, run the "inspect volume contents" check
below BEFORE assuming the code is wrong.

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

    username = os.environ["USERNAME"]
    password = os.environ["PASSWORD"]
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
