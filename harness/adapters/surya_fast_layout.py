#!/usr/bin/env python3
"""Surya 2 fast-layout adapter — vendored RF-DETR detector + learned reading order.

Surya 2 ships two layout paths.  `LayoutPredictor` is the VLM (see
`surya_layout.py`); `FastLayoutPredictor` is a lightweight RF-DETR object
detector with an autoregressive reading-order head, documented in the repo as a
"drop-in alternative … same LayoutResult/LayoutBox output".  It is the closer
analogue to the other detectors in this benchmark.

Upstream `FastLayoutPredictor` is a thin HTTP client of a shared server process
(`surya.fast_layout.server`) so that N marker workers share one model.  Here it
is driven in-process through the exact same call chain the server uses —
`load_detector` → `RfDetrTorch.detect` → `build_layout_result` — so the
detections, relabeling, containment merge and reading order are bit-identical
to the served path, without HTTP latency polluting the per-phase timings.
"""
import os, sys
from PIL import Image
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg = run.cfg
    thr = float(cfg.get("threshold", 0.4))
    use_order = bool(cfg.get("use_order", True))
    device = cfg.get("device", "cuda")

    t = Timer()
    with t.phase("model_load"):
        from surya.common.rfdetr_torch import load_detector, resolve_model_dir
        from surya.common.order.predictor import load_order_predictor
        from surya.fast_layout import build_layout_result
        from surya.settings import settings

        if cfg.get("containment_threshold") is not None:
            settings.FAST_LAYOUT_CONTAINMENT_THRESHOLD = float(cfg["containment_threshold"])

        ckpt = cfg.get("checkpoint", settings.FAST_LAYOUT_MODEL_CHECKPOINT)
        model_dir = resolve_model_dir(ckpt)
        det = load_detector(model_dir, num_threads=settings.FAST_LAYOUT_NUM_THREADS,
                            device=device)
        order = load_order_predictor(device=device) if use_order else None
        # Warm-up: first call builds CUDA kernels / autotunes; excluded from timings.
        det.detect([Image.new("RGB", (1024, 1024), "white")], threshold=thr,
                   return_features=use_order)
    run.model_load_s = t.pop()["total_s"]

    run.set_model_info(
        checkpoint=ckpt, model_dir=model_dir,
        framework="surya-ocr fast-layout (vendored RF-DETR, torch)",
        detector="RF-DETR (DINOv2-windowed backbone, LW-DETR head)",
        device=str(det.device), threshold=thr,
        labels=sorted(det.id2label.values()),
        reading_order=bool(order is not None),
        reading_order_source=("learned AR head (order_ar.pt)" if order is not None
                              else "raster sort (order model unavailable)"),
        containment_merge=settings.FAST_LAYOUT_CONTAINMENT_THRESHOLD,
        served_path_bypassed="in-process; identical call chain to surya.fast_layout.server",
    )

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                img = Image.open(page["input_path"]).convert("RGB")
            with t.phase("inference"):
                dets = det.detect([img], threshold=thr, batch_size=1,
                                  return_features=use_order)[0]
                cuda_sync()
            with t.phase("postprocess"):
                res = build_layout_result(img, dets, order)
                items = []
                for b in res.bboxes:
                    poly = [[float(x), float(y)] for x, y in b.polygon] if b.polygon else None
                    items.append({"source_class": b.label,
                                  "bbox": [float(v) for v in b.bbox],
                                  "confidence": (float(b.confidence)
                                                 if b.confidence is not None else None),
                                  "polygon": poly,
                                  "reading_order": b.position,
                                  "extra": {"raw_label": b.raw_label, "count": b.count}})
                regions = build_regions(run.taxonomy, items)
                raw = {"bboxes": [b.model_dump() for b in res.bboxes],
                       "image_bbox": res.image_bbox, "error": res.error,
                       "detections_pre_merge": [dict(d) for d in dets]}
            run.emit(page, regions, t.pop(), raw=raw,
                     meta={"threshold": thr, "use_order": use_order})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
