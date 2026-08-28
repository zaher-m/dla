#!/usr/bin/env bash
# Eynollah: pixelwise segmentation via ONNX Runtime.
#
# Two deliberate departures from the other environments:
#   * Python 3.11, not the container's 3.12 — eynollah declares support for
#     3.8-3.11 only, and on 3.12 its multiprocessing Predictor dies with
#     "SemLock._rebuild: FileNotFoundError" before any model runs.  uv fetches a
#     standalone CPython 3.11 for this venv alone.
#   * no --system-site-packages — eynollah needs no torch, so nothing is
#     inherited and the 3.11/3.12 split costs nothing.
# onnxruntime-gpu has no linux-aarch64 wheel on PyPI, so this runs on CPU.
source "$(dirname "$0")/_common.sh"
n=eynollah
if [ ! -x "$ENVS/$n/bin/python" ]; then
  uv venv --seed --python 3.11 "$ENVS/$n"
fi
PY="$ENVS/$n/bin/python"
pipi "$PY" "onnxruntime" "numpy<2" "opencv-python-headless" "scikit-learn" "scikit-image" \
           "shapely" "pillow" "biopython" "loky" "tqdm" "pyyaml" "lxml" "click" "ocrd" "tabulate" || true
pipi "$PY" --no-deps -e "$REPOS/eynollah"
"$PY" -c "import eynollah, onnxruntime, sys; print('eynollah on', sys.version.split()[0], 'ort', onnxruntime.__version__)"
record_env "$PY" "$n"
