#!/usr/bin/env bash
# DocSAM (CVPR 2025, xhli-git/DocSAM) — unified instance + semantic document
# segmentation.
#
# Two things make this system different from everything else in the benchmark:
#   * it is a *segmentation* model (Mask2Former decoder) that emits real
#     per-instance masks, not boxes dressed up as polygons; and
#   * its class set is a **prompt**.  Class names are embedded with a sentence
#     encoder (all-MiniLM-L6-v2) and used as semantic queries, so the taxonomy
#     is chosen at inference time instead of being frozen into a linear head.
#
# Upstream pins torch 2.5.1 / numpy 2.2.4 / transformers 4.49.0.  torch is
# inherited from the container's CUDA build; only transformers is
# pinned, because the vendored `models/mask2former` is a fork of the 4.49
# implementation and imports internals that later releases moved
# (`transformers.pytorch_utils.is_torch_greater_or_equal_than_2_1`,
# `transformers.file_utils`).
#
# `torch_xla` (in upstream requirements.txt) is a TPU runtime and is never
# imported by the inference path; it is not installed.
#
# `jpeg4py` needs libturbojpeg, which is not present in the container and
# cannot be installed without root.  It is imported only by
# `datasets/dataset.py`, whose COCO ground-truth loader this benchmark does not
# use — the adapter feeds pages directly and stubs the module.  See
# harness/adapters/docsam_layout.py.
source "$(dirname "$0")/_common.sh"
PY=$(mkenv docsam)

pipi "$PY" "transformers==4.49.0" "einops" "torch_dct" "prefetch_generator" \
           "pycocotools" "opencv-python-headless" "scipy" "accelerate" \
           "huggingface_hub" "safetensors" "tqdm" "pillow" "pyyaml"

# The repository hardcodes *relative* paths for its two pretrained components
# ("./pretrained_model/mask2former/...", "./pretrained_model/sentence/...").
# They are staged under assets/models/docsam and the adapter chdirs there,
# so the upstream tree stays unmodified.
STAGE="$MODELS/docsam/pretrained_model"
mkdir -p "$STAGE/mask2former" "$STAGE/sentence"
"$PY" - "$STAGE" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
stage = sys.argv[1]
want = {
    "mask2former/facebook-mask2former-swin-base-coco-panoptic":
        "facebook/mask2former-swin-base-coco-panoptic",
    "mask2former/facebook-mask2former-swin-large-coco-panoptic":
        "facebook/mask2former-swin-large-coco-panoptic",
    "sentence/all-MiniLM-L6-v2":
        "sentence-transformers/all-MiniLM-L6-v2",
}
for rel, repo in want.items():
    dst = os.path.join(stage, rel)
    if os.path.exists(os.path.join(dst, "config.json")):
        print("have", rel); continue
    src = snapshot_download(repo, allow_patterns=[
        "*.json", "*.txt", "*.bin", "*.safetensors", "*.model"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)
    print("linked", rel, "->", src)
PY

record_env "$PY" docsam
assert_torch "$PY" docsam
