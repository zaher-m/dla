#!/usr/bin/env python3
"""MinerU 2.5 VLM-backend layout adapter.

MinerU's VLM backend performs layout detection as its first decoding stage and
emits typed blocks with bounding boxes in reading order.  Only that stage's
output is scored; the per-block content decoding that follows is not part of
the layout comparison.
"""
import os, sys, json
import numpy as np
from PIL import Image
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset


def main():
    job = parse_job()
    run = AdapterRun(job)
    t = Timer()
    with t.phase("model_load"):
        import torch
        from mineru.backend.vlm.vlm_analyze import ModelSingleton
        from mineru.utils.enum_class import ModelPath
        singleton = ModelSingleton()
        predictor = singleton.get_model(backend="transformers", model_path=None,
                                        server_url=None)
    run.model_load_s = t.pop()["total_s"]
    run.set_model_info(model=ModelPath.vlm_root_hf, backend="transformers",
                       framework="MinerU 2.5 VLM", reading_order=True,
                       device="cuda" if torch.cuda.is_available() else "cpu")

    def layout_of(img):
        # two_step_extract's first stage is layout; call it directly
        from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make  # noqa: F401
        out = predictor.batch_two_step_extract(images=[img])
        return out[0]

    warm = Image.open(job["pages"][0]["input_path"]).convert("RGB")
    try:
        layout_of(warm)
    except Exception:
        pass
    cuda_sync()

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                img = Image.open(page["input_path"]).convert("RGB")
            with t.phase("inference"):
                blocks = layout_of(img)
                cuda_sync()
            with t.phase("postprocess"):
                items = []
                for i, b in enumerate(blocks):
                    bb = b.get("bbox") or b.get("poly")
                    if bb is None:
                        continue
                    if len(bb) == 8:
                        xs, ys = bb[0::2], bb[1::2]
                        bb = [min(xs), min(ys), max(xs), max(ys)]
                    items.append({"source_class": b.get("type") or b.get("label"),
                                  "bbox": bb, "confidence": b.get("score"),
                                  "reading_order": b.get("index", i)})
                regions = build_regions(run.taxonomy, items)
            run.emit(page, regions, t.pop(), raw={"blocks": blocks},
                     meta={"backend": "vlm-transformers"})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
