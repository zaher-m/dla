#!/usr/bin/env python3
"""PDF-Extract-Kit LayoutLMv3 adapter (Cascade R-CNN + LayoutLMv3/ViT backbone).

Uses PDF-Extract-Kit's own `Layoutlmv3_Predictor` and its published
DocStructBench checkpoint from `opendatalab/PDF-Extract-Kit-1.0`.  The model is
run in image-only mode (`MODEL.IMAGE_ONLY=True` in the repo's inference config),
so no OCR text stream is required or used.
"""
import os, sys
import numpy as np
from PIL import Image
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset

ID_TO_NAMES = {0: 'title', 1: 'plain text', 2: 'abandon', 3: 'figure', 4: 'figure_caption',
               5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula',
               9: 'formula_caption'}


def main():
    job = parse_job()
    run = AdapterRun(job)
    pek = os.path.join(job["bench"], "repositories", "PDF-Extract-Kit")
    sys.path.insert(0, pek)
    cwd = os.getcwd()
    os.chdir(pek)                      # the repo's config path is relative
    # PDF-Extract-Kit's `tasks/__init__` eagerly imports every task, dragging in
    # formula/table dependencies (unimernet, struct_eqtable, ...) that have
    # nothing to do with layout.  Register the intermediate packages as plain
    # namespace packages so only the layout module tree is executed; the repo's
    # own layout code is otherwise imported and used unmodified.
    import types
    for name, sub in (("pdf_extract_kit.tasks", "tasks"),
                      ("pdf_extract_kit.tasks.layout_detection", "tasks/layout_detection"),
                      ("pdf_extract_kit.tasks.layout_detection.models",
                       "tasks/layout_detection/models")):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(pek, "pdf_extract_kit", *sub.split("/"))]
            m.__package__ = name
            sys.modules[name] = m

    # PDF-Extract-Kit vendors a LayoutLMv3 implementation written against an
    # older transformers, where several helpers lived in `modeling_utils`.
    # They now live in `pytorch_utils`; re-export them rather than pinning an
    # ancient transformers into this environment.
    import transformers.modeling_utils as _mu
    import transformers.pytorch_utils as _pu
    for _n in ("find_pruneable_heads_and_indices", "prune_linear_layer",
               "apply_chunking_to_forward", "Conv1D"):
        if not hasattr(_mu, _n) and hasattr(_pu, _n):
            setattr(_mu, _n, getattr(_pu, _n))

    t = Timer()
    with t.phase("model_load"):
        import torch
        from huggingface_hub import hf_hub_download
        from pdf_extract_kit.tasks.layout_detection.models.layoutlmv3_util.model_init import (
            Layoutlmv3_Predictor)
        # the ViT backbone builder reads the LayoutLMv3 config that sits beside
        # the checkpoint, so fetch both into the same snapshot directory
        hf_hub_download("opendatalab/PDF-Extract-Kit-1.0",
                        "models/Layout/LayoutLMv3/config.json")
        wt = hf_hub_download("opendatalab/PDF-Extract-Kit-1.0",
                             "models/Layout/LayoutLMv3/model_final.pth")
        predictor = Layoutlmv3_Predictor(wt)
    run.model_load_s = t.pop()["total_s"]
    run.set_model_info(repo_id="opendatalab/PDF-Extract-Kit-1.0",
                       weights="models/Layout/LayoutLMv3/model_final.pth", local_path=wt,
                       architecture="Cascade R-CNN + LayoutLMv3 ViT backbone (image-only)",
                       framework="detectron2", labels=ID_TO_NAMES,
                       device="cuda" if torch.cuda.is_available() else "cpu")

    warm = np.array(Image.open(job["pages"][0]["input_path"]).convert("RGB"))
    predictor(warm, ignore_catids=[]); cuda_sync()

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                im = np.array(Image.open(page["input_path"]).convert("RGB"))
            with t.phase("inference"):
                res = predictor(im, ignore_catids=[])
                cuda_sync()
            with t.phase("postprocess"):
                items = []
                for d in res["layout_dets"]:
                    p = d["poly"]
                    items.append({"source_class": ID_TO_NAMES[int(d["category_id"])],
                                  "bbox": [p[0], p[1], p[4], p[5]],
                                  "confidence": d["score"]})
                regions = build_regions(run.taxonomy, items)
            run.emit(page, regions, t.pop(), raw=res)
        except Exception as e:
            t.pop(); run.fail(page, e)
    os.chdir(cwd)
    run.finish()


if __name__ == "__main__":
    main()
