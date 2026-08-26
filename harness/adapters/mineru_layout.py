#!/usr/bin/env python3
"""MinerU pipeline-backend layout adapter.

Calls MinerU's own layout component (`PPDocLayoutV2LayoutModel`) directly, so
only layout detection + reading-order prediction are exercised — no OCR,
formula or table stages enter the scored output or the timings.
"""
import os, sys
import numpy as np
from PIL import Image
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg = run.cfg
    t = Timer()
    with t.phase("model_load"):
        import torch
        from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel
        from mineru.utils.enum_class import ModelPath
        from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
        root = auto_download_and_get_model_root_path(ModelPath.pp_doclayout_v2)
        weights = os.path.join(root, ModelPath.pp_doclayout_v2)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        kw = {}
        if cfg.get("conf") is not None:
            kw["conf"] = float(cfg["conf"])
        if cfg.get("use_paddlex_filter_boxes") is not None:
            kw["use_paddlex_filter_boxes"] = bool(cfg["use_paddlex_filter_boxes"])
        model = PPDocLayoutV2LayoutModel(weights, device=dev, **kw)
    run.model_load_s = t.pop()["total_s"]
    run.set_model_info(model="PP-DocLayoutV2 (MinerU PyTorch port)", weights_dir=weights,
                       hf_repo="opendatalab/PDF-Extract-Kit-1.0",
                       framework="pytorch/transformers", device=dev,
                       imgsz=list(model.imgsz), conf=model.conf,
                       use_paddlex_filter_boxes=model.use_paddlex_filter_boxes,
                       labels=getattr(model.config, "id2label", None),
                       reading_order=True)

    model.predict(Image.open(job["pages"][0]["input_path"]).convert("RGB"))
    cuda_sync()

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                img = Image.open(page["input_path"]).convert("RGB")
            with t.phase("inference"):
                res = model.predict(img)
                cuda_sync()
            with t.phase("postprocess"):
                items = [{"source_class": d["label"], "bbox": d["bbox"],
                          "confidence": d["score"], "reading_order": d.get("index")}
                         for d in res]
                regions = build_regions(run.taxonomy, items)
            run.emit(page, regions, t.pop(), raw={"layout": res},
                     meta={"backend": "pipeline", "device": dev})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
