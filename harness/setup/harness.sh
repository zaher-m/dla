#!/usr/bin/env bash
# The harness itself: ingestion, the PDF Structural Reference, metrics, report
# generation, the pipeline orchestrator and the web application.
#
# The container image already carries these dependencies system-wide, so the UI
# and the pipeline work on a fresh clone before this script has ever run. Build
# this environment when you want the harness pinned independently of the image
# — for example to upgrade PyMuPDF without rebuilding, or to run outside the
# container. `core.paths.harness_python()` prefers it when it exists.
source "$(dirname "$0")/_common.sh"
PY=$(mkenv harness)
pipi "$PY" \
     "pymupdf" "pillow" "numpy<2.3" "opencv-python-headless" "shapely" \
     "matplotlib" "pandas" "jinja2" "pyyaml" "scipy" \
     "fastapi" "uvicorn[standard]" "python-multipart"
"$PY" -c "import fitz, fastapi, uvicorn; print('harness ok: pymupdf', fitz.__doc__.split()[1])"
record_env "$PY" harness
