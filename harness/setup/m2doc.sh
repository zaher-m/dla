#!/usr/bin/env bash
# M2Doc (AAAI'24) -- the second OCR-assisted arm, and the interesting one.
#
# VGT's text stream runs on an English WordPiece vocabulary that shreds Arabic
# into single characters.  M2Doc fuses visual features with
# `bert-base-multilingual-cased`, which actually covers Arabic, so it tests
# whether the text layer is worth more when the encoder can read it.
#
# The repo vendors its own fork of mmdetection 3.3.0, which wants
# mmcv >= 2.0.0rc4, < 2.2.0 and mmengine >= 0.7.1.  mmcv 2.x has no aarch64 /
# wheel for current torch, so it is built from source the same way mmcv 1.x
# was for RoDLA.
source "$(dirname "$0")/_common.sh"
PY=$(mkenv m2doc)
export MAX_JOBS="${MAX_JOBS:-16}"
export FORCE_CUDA=1
export MMCV_WITH_OPS=1

pipi "$PY" "numpy<2" "pillow" "opencv-python-headless" "pyyaml" "addict" "yapf<0.41" \
           "termcolor" "packaging" "pycocotools" "shapely" "scipy" "matplotlib" \
           "tqdm" "timm" "regex" "six" "terminaltables" "transformers" "rich" \
           "pdfplumber"
pipi "$PY" "mmengine==0.10.7"

echo "[m2doc] building mmcv 2.1.0 from source"
pipi "$PY" --no-build-isolation --no-deps "mmcv==2.1.0" \
  && echo "[m2doc] mmcv built OK" || { echo "[m2doc] mmcv BUILD FAILED"; exit 21; }

pipi "$PY" --no-build-isolation --no-deps \
     -e "$REPOS/M2Doc/mmdetection" \
  || { echo "[m2doc] vendored mmdetection install FAILED"; exit 22; }

assert_torch "$PY" m2doc
record_env "$PY" m2doc
