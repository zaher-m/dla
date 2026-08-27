#!/usr/bin/env bash
# VGT (Vision Grid Transformer): image plus a word grid from the PDF text layer.
#
# VGT is a two-stream detector: a DiT-base visual stream plus a Grid Transformer
# that consumes a token-id raster built from word boxes.  Upstream builds that
# raster with pdfplumber over a machine-readable PDF, which is exactly what this
# corpus is -- every page is born-digital with an intact text layer -- so the
# text stream costs no OCR and carries no OCR error.  pdfplumber is installed so
# the grid is produced by the *upstream* extractor, not a substitute.
#
# Reuses the detectron2 source build (same recipe as detectron2.sh/swindocseg.sh).
source "$(dirname "$0")/_common.sh"
PY=$(mkenv vgt)
pipi "$PY" "pillow" "opencv-python-headless" "pyyaml" "pycocotools" "omegaconf" "hydra-core" \
           "fvcore" "iopath" "cloudpickle" "tabulate" "termcolor" "matplotlib" "tqdm" \
           "huggingface_hub" "timm" "transformers" "shapely" "scipy" "numpy<2.3" \
           "pdfplumber" "einops"
export MAX_JOBS="${MAX_JOBS:-16}"
pipi "$PY" --no-build-isolation --no-deps -e "$REPOS/detectron2"
assert_torch "$PY" vgt
record_env "$PY" vgt
