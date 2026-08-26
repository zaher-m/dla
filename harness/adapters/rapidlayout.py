#!/usr/bin/env python3
"""RapidLayout adapter: ONNX Runtime, third-party layout checkpoints.

RapidLayout republishes several independently trained detectors behind one API.
Only the checkpoints not otherwise reachable are registered here: the PP-layout
PicoDet models (CDLA, PubLayNet) and the 360LayoutAnalysis YOLOv8n models.
DocLayout-YOLO and PP-DocLayoutV2/V3 come through their own repositories.

Label sets are read from each ONNX file's metadata, not hardcoded. CPU only:
onnxruntime publishes no CUDA execution provider for this platform.
"""
import os, sys, hashlib, json
import numpy as np
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions
from core import paths

MODELS = paths.model_dir("rapidlayout")


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg = run.cfg
    name = cfg["model_type"]
    thr = float(cfg.get("threshold", 0.5))
    iou = float(cfg.get("iou_threshold", 0.5))
    wpath = os.path.join(MODELS, name + ".onnx")

    t = Timer()
    with t.phase("model_load"):
        from rapid_layout import RapidLayout
        from importlib.metadata import version as _pkgver
        engine = RapidLayout(model_type=name, model_dir_or_path=wpath,
                             conf_thresh=thr, iou_thresh=iou)
        labels = list(engine.session.characters)
        engine(job["pages"][0]["input_path"])          # warm-up
    run.model_load_s = t.pop()["total_s"]

    with open(wpath, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    run.set_model_info(
        model_type=name, local_path=wpath, sha256=sha, labels=labels,
        framework=f"rapid_layout {_pkgver('rapid_layout')} on onnxruntime",
        device="cpu (onnxruntime publishes no aarch64 CUDA execution provider)",
        threshold=thr, iou_threshold=iou,
        upstream_source=cfg.get("upstream_source", ""),
        architecture=cfg.get("architecture", ""),
        training_set=cfg.get("training_set", ""),
        provenance=("ONNX weights from the RapidAI ModelScope release "
                    "(RapidAI/RapidLayout v1.2.0), verified at setup time against the "
                    "SHA256 digests published in the repository's own "
                    "rapid_layout/configs/default_models.yaml"))

    for page in job["pages"]:
        try:
            with t.phase("inference"):
                out = engine(page["input_path"])
            with t.phase("postprocess"):
                # The YOLOv8 handler returns numpy arrays and the PP handler
                # returns lists, so `or []` is not safe here: truth-testing a
                # numpy array raises.  Normalise explicitly.
                def _seq(v):
                    return [] if v is None else list(v)
                boxes, names, scores = _seq(out.boxes), _seq(out.class_names), _seq(out.scores)
                items = []
                for bb, cname, score in zip(boxes, names, scores):
                    items.append({"source_class": str(cname),
                                  "bbox": [float(v) for v in bb],
                                  "confidence": float(score)})
                regions = build_regions(run.taxonomy, items)
                raw = {"boxes": [[float(v) for v in b] for b in boxes],
                       "class_names": [str(c) for c in names],
                       "scores": [float(s) for s in scores],
                       "labels": labels}
            run.emit(page, regions, t.pop(), raw=raw,
                     meta={"model_type": name, "device": "cpu",
                           "runtime": "onnxruntime", "threshold": thr})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
