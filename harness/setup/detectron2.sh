#!/usr/bin/env bash
# Detectron2 built from source against the inherited CUDA torch.
# Hosts: Layout-Parser (PubLayNet / PRImA), PDF-Extract-Kit LayoutLMv3,
# UniLM LayoutLMv3-PubLayNet.
source "$(dirname "$0")/_common.sh"
PY=$(mkenv detectron2)
pipi "$PY" "pillow" "opencv-python-headless" "pyyaml" "pycocotools" "omegaconf" "hydra-core" \
           "fvcore" "iopath" "cloudpickle" "tabulate" "termcolor" "matplotlib" "tqdm" \
           "huggingface_hub" "timm" "transformers" "shapely" "scipy" "numpy<2.3" "black"
export MAX_JOBS="${MAX_JOBS:-16}"
pipi "$PY" --no-build-isolation --no-deps -e "$REPOS/detectron2"
pipi "$PY" --no-deps "layoutparser"
pipi "$PY" "effdet" --no-deps || true
assert_torch "$PY" detectron2
record_env "$PY" detectron2
