"""
modal_physician_tool.py

Deploys physician_tool.py (Flask app, Module 1 image/mask review + Module 2
report review with the callout visualization) as a persistent Modal web
endpoint, reachable from any laptop, backed by a Modal Volume instead of
local disk -- so the case data, generated reports, overlays, and physician
decisions all live in one place instead of only on one Mac.

Setup (one-time):
    1. Create the volume (separate from the existing 'cardiac-data' volume
       used by the LeFusion/synthetic-generation pipeline -- this one is
       dedicated to physician_tool's own data, kept independent on purpose):
           modal volume create physician-tool-data

    2. Upload your existing local data into it. Run each of these from
       wherever that folder currently lives on your Mac (skip any you don't
       have -- e.g. skip synthetic_cases if you only review real cases):
           modal volume put physician-tool-data /path/to/Case_* real_cases/
           modal volume put physician-tool-data /path/to/synthetic_cases synthetic_cases/
           modal volume put physician-tool-data /path/to/outputs_region_grounded outputs_region_grounded/
           modal volume put physician-tool-data /path/to/overlays overlays/
       (real_cases/ should end up containing Case_XXXX/Images/... and
       Case_XXXX/Contours/... directly, matching what REAL_CASES_DIR expects
       -- i.e. upload the PARENT folder that directly contains the Case_*
       directories, not a folder one level higher.)

    3. Set the shared login used to gate access (pick your own values):
           modal secret create physician-tool-auth \\
               BASIC_AUTH_USER=physician \\
               BASIC_AUTH_PASSWORD=<a real password, not this placeholder>

    4. Deploy:
           modal deploy modal_physician_tool.py
       Prints a persistent https://....modal.run URL on success. Anyone with
       that URL AND the credentials from step 3 can use the tool -- do not
       post the URL anywhere public, this serves real EMIDEC patient data.

To add more cases/reports later, repeat step 2's relevant `modal volume put`
command -- no redeploy needed, each new container reads the volume's latest
committed state on startup.
"""

import modal

app = modal.App("physician-tool")

volume = modal.Volume.from_name("physician-tool-data", create_if_missing=True)
VOLUME_PATH = "/physician-tool-data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "flask",
        "numpy",
        "nibabel",
        "matplotlib",
    )
    .add_local_file("physician_tool.py", "/root/physician_tool.py")
    .add_local_dir("templates", "/root/templates")
)


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("physician-tool-auth")],
    # keep the container warm for a bit after the last request -- a physician
    # clicking through several cases in a row shouldn't eat a cold-start
    # (mostly matplotlib/nibabel import time) on every single click
    scaledown_window=600,
    timeout=120,
)
@modal.concurrent(max_inputs=20)
@modal.wsgi_app()
def flask_app():
    import os
    import sys

    # physician_tool.py resolves all its data paths from os.getcwd() (see
    # WORK_DIR in that file) -- running from inside the mounted volume makes
    # every one of those paths land on the volume with zero changes to the
    # reviewed/tested core file, instead of re-deriving each path here.
    os.makedirs(VOLUME_PATH, exist_ok=True)
    os.chdir(VOLUME_PATH)
    sys.path.insert(0, "/root")

    # REAL_CASES_DIR defaults to WORK_DIR in physician_tool.py, which would
    # make it VOLUME_PATH itself -- but real cases live one level down, in
    # real_cases/, per the upload layout in this file's docstring.
    os.environ.setdefault("REAL_CASES_DIR", os.path.join(VOLUME_PATH, "real_cases"))
    os.environ.setdefault("SYNTHETIC_CASES_DIR", os.path.join(VOLUME_PATH, "synthetic_cases"))
    os.environ.setdefault("REPORTS_DIR", os.path.join(VOLUME_PATH, "outputs_region_grounded"))
    os.environ.setdefault("REPORT_OVERLAYS_DIR", os.path.join(VOLUME_PATH, "overlays"))

    import physician_tool as pt
    from flask import request, Response

    auth_user = os.environ["BASIC_AUTH_USER"]
    auth_password = os.environ["BASIC_AUTH_PASSWORD"]

    def check_auth(username, password):
        return username == auth_user and password == auth_password

    @pt.app.before_request
    def require_basic_auth():
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response(
                "Login required.", 401,
                {"WWW-Authenticate": 'Basic realm="Physician Tool"'},
            )

    # physician_tool.py writes decisions with plain open(path, "a") (see
    # api_validate / api_validate_report), which lands on the volume's local
    # FUSE mount fine, but Modal Volumes need an explicit commit() to make a
    # write visible to OTHER containers (e.g. the next request landing on a
    # fresh container once this one scales down). Without this, a saved
    # decision could look successful but not durably persist or be visible
    # elsewhere -- committing after every request is cheap next to the cost
    # of a physician's review silently not sticking.
    @pt.app.after_request
    def commit_volume(response):
        if request.method == "POST":
            volume.commit()
        return response

    return pt.app
