#!/usr/bin/env bash
# SwinDocSegmenter (Swin-L + MaskDINO) — true instance segmentation on DocLayNet.
#
# Reuses the detectron2 environment's build recipe.  The vendored MaskDINO pixel
# decoder wants a compiled MultiScaleDeformableAttention CUDA kernel; the repo
# ships only an x86_64/py38 .so, useless here.  We attempt the source build for
# this arch and fall back to MaskDINO's own pure-PyTorch path
# (`ms_deform_attn_core_pytorch`), which the module already selects on ImportError.
source "$(dirname "$0")/_common.sh"
PY=$(mkenv swindocseg)
pipi "$PY" "pillow" "opencv-python-headless" "pyyaml" "pycocotools" "omegaconf" "hydra-core" \
           "fvcore" "iopath" "cloudpickle" "tabulate" "termcolor" "matplotlib" "tqdm" \
           "huggingface_hub" "timm" "shapely" "scipy" "numpy<2.3" "cython" "h5py" \
           "scikit-image" "submitit"
export MAX_JOBS="${MAX_JOBS:-16}"
pipi "$PY" --no-build-isolation --no-deps -e "$REPOS/detectron2"

OPS="$REPOS/SwinDocSegmenter/maskdino/modeling/pixel_decoder/ops"
if [ -d "$OPS" ]; then
  echo "[swindocseg] building the MSDeformAttn CUDA kernel"
  ( cd "$OPS" && FORCE_CUDA=1 "$PY" setup.py build install ) \
     && echo "[swindocseg] MSDeformAttn kernel built" \
     || echo "[swindocseg] kernel build FAILED - falling back to ms_deform_attn_core_pytorch"
fi
assert_torch "$PY" swindocseg
record_env "$PY" swindocseg
