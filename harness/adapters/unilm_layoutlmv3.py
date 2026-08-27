#!/usr/bin/env python3
"""UniLM LayoutLMv3 adapter — PubLayNet Cascade R-CNN detector.

Runs the checkpoint published with the LayoutLMv3 paper
(`HYPJUDY/layoutlmv3-base-finetuned-publaynet`) through UniLM's own
`cascade_layoutlmv3.yaml` and `ditod` detectron2 code.  The released config
sets MODEL.IMAGE_ONLY, so detection needs no OCR token stream.
"""
import os, sys
import numpy as np
from PIL import Image
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset

PUBLAYNET = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}


def main():
    job = parse_job()
    run = AdapterRun(job)
    od = os.path.join(job["bench"], "repositories", "unilm", "layoutlmv3",
                      "examples", "object_detection")
    sys.path.insert(0, od)
    # `ditod` imports `layoutlmft`, which lives one level up in the repo
    sys.path.insert(0, os.path.join(job["bench"], "repositories", "unilm", "layoutlmv3"))
    cwd = os.getcwd(); os.chdir(od)

    # The vendored layoutlmft registers `layoutlmv3` into the Auto* registries;
    # modern transformers ships that model natively, so make re-registration a
    # no-op instead of a hard error.
    from transformers.models.auto import configuration_auto as _ca
    from transformers.models.auto import auto_factory as _af
    from transformers.models.auto import tokenization_auto as _ta
    def _tolerant(orig):
        def _f(*a, **k):
            try:
                return orig(*a, **k)
            except ValueError as e:
                if "already used" in str(e):
                    return None
                raise
        return _f
    _ca.AutoConfig.register = staticmethod(_tolerant(_ca.AutoConfig.register))
    _af._BaseAutoModelClass.register = classmethod(
        lambda cls, *a, **k: None)
    _ta.AutoTokenizer.register = staticmethod(_tolerant(_ta.AutoTokenizer.register))

    # ditod's table-evaluation helper still does `from collections import Iterable`
    # (removed in Python 3.10); restore the alias rather than editing the repo.
    import collections, collections.abc
    for _n in ("Iterable", "Mapping", "MutableMapping", "Sequence", "Callable"):
        if not hasattr(collections, _n):
            setattr(collections, _n, getattr(collections.abc, _n))

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
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from ditod import add_vit_config
        wt = hf_hub_download("HYPJUDY/layoutlmv3-base-finetuned-publaynet", "model_final.pth")
        cfg = get_cfg()
        add_vit_config(cfg)
        cfg.merge_from_file(os.path.join(od, "cascade_layoutlmv3.yaml"))
        cfg.merge_from_list(["MODEL.WEIGHTS", wt, "MODEL.DEVICE",
                             "cuda" if torch.cuda.is_available() else "cpu"])
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(run.cfg.get("score_thresh", 0.5))
        cfg.freeze()
        predictor = DefaultPredictor(cfg)
    run.model_load_s = t.pop()["total_s"]
    run.set_model_info(repo_id="HYPJUDY/layoutlmv3-base-finetuned-publaynet",
                       local_path=wt, config="unilm/layoutlmv3 cascade_layoutlmv3.yaml",
                       architecture="Cascade R-CNN + LayoutLMv3-base backbone (image-only)",
                       framework="detectron2", labels=PUBLAYNET,
                       score_thresh=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
                       device=cfg.MODEL.DEVICE)

    warm = np.array(Image.open(job["pages"][0]["input_path"]).convert("RGB"))[:, :, ::-1]
    predictor(warm); cuda_sync()

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                im = np.array(Image.open(page["input_path"]).convert("RGB"))[:, :, ::-1]
            with t.phase("inference"):
                out = predictor(im)
                cuda_sync()
            with t.phase("postprocess"):
                inst = out["instances"].to("cpu")
                boxes = inst.pred_boxes.tensor.tolist()
                cls = inst.pred_classes.tolist()
                sc = inst.scores.tolist()
                items = [{"source_class": PUBLAYNET.get(int(c), str(c)), "bbox": b,
                          "confidence": s} for b, c, s in zip(boxes, cls, sc)]
                regions = build_regions(run.taxonomy, items)
                raw = {"boxes": boxes, "classes": cls, "scores": sc, "labels": PUBLAYNET}
            run.emit(page, regions, t.pop(), raw=raw)
        except Exception as e:
            t.pop(); run.fail(page, e)
    os.chdir(cwd)
    run.finish()
if __name__ == "__main__": main()
