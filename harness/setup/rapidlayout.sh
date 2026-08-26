#!/usr/bin/env bash
# RapidLayout (RapidAI) — ONNX Runtime layout detection.
#
# RapidLayout is an aggregator: it republishes several *third-party* layout
# checkpoints as ONNX behind one API.  Three of its eleven model types are
# already in this benchmark under their own repositories (DocLayout-YOLO
# docstructbench/d4la/docsynth via PDF-Extract-Kit, PP-DocLayoutV2/V3 via
# PaddleOCR) and are deliberately NOT registered again here.
#
# What is new are two families we have no other access to:
#
#   * PP-layout PicoDet checkpoints trained on **CDLA** (Chinese academic
#     literature, 10 classes) and PubLayNet — PicoDet is an architecture no
#     other system here uses.
#   * **360LayoutAnalysis** YOLOv8n checkpoints (360AILAB-NLP): `paper`,
#     `report` and `general6`. The `report` model is trained on Chinese
#     research reports, a training distribution nothing else here covers.
#
# Weights are fetched from the upstream ModelScope release by URL and verified
# against the SHA256 digests that RapidLayout itself publishes in
# rapid_layout/configs/default_models.yaml, so the provenance chain is the
# upstream release, not a re-export by us.
source "$(dirname "$0")/_common.sh"
PY=$(mkenv rapidlayout)

pipi "$PY" "rapid-layout==1.2.1" "onnxruntime" "opencv-python-headless" "pillow"

DEST="$MODELS/rapidlayout"
mkdir -p "$DEST"
CFG="$REPOS/RapidLayout/rapid_layout/configs/default_models.yaml"
"$PY" - "$CFG" "$DEST" <<'PY'
import hashlib, os, sys, urllib.request, yaml
cfg, dest = sys.argv[1], sys.argv[2]
spec = yaml.safe_load(open(cfg))
want = ["pp_layout_cdla", "pp_layout_publaynet",
        "yolov8n_layout_paper", "yolov8n_layout_report", "yolov8n_layout_general6"]
for name in want:
    url, want_sha = spec[name]["model_dir_or_path"], spec[name]["SHA256"]
    out = os.path.join(dest, name + ".onnx")
    if not os.path.exists(out):
        print("downloading", name, url, flush=True)
        urllib.request.urlretrieve(url, out)
    got = hashlib.sha256(open(out, "rb").read()).hexdigest()
    if got != want_sha:
        raise SystemExit(f"SHA256 mismatch for {name}: {got} != {want_sha}")
    print(f"ok {name} {os.path.getsize(out)/2**20:.1f} MB sha256={got[:12]}")
PY

# Record each model's own label list, straight out of the ONNX metadata.
"$PY" - "$DEST" <<'PY'
import json, os, sys
import onnxruntime as ort
dest = sys.argv[1]
labels = {}
for f in sorted(os.listdir(dest)):
    if not f.endswith(".onnx"):
        continue
    m = ort.InferenceSession(os.path.join(dest, f),
                             providers=["CPUExecutionProvider"]).get_modelmeta()
    labels[f[:-5]] = m.custom_metadata_map
print(json.dumps(labels, ensure_ascii=False, indent=1))
json.dump(labels, open(os.path.join(dest, "_labels.json"), "w"),
          ensure_ascii=False, indent=1)
PY

record_env "$PY" rapidlayout
