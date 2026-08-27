#!/usr/bin/env python3
"""DiT (Document Image Transformer) adapter: Cascade Mask R-CNN on PubLayNet.

A BEiT-style backbone self-supervised on unlabelled document images, behind the
same detection head as the ResNet PubLayNet baselines — useful as a control for
whether a result is about the backbone or about the training set.

Microsoft's own weight URLs return PublicAccessNotPermitted, so the checkpoint
comes from a Hugging Face mirror; the sha256 of the downloaded file is recorded
in the run manifest.
"""
import os, sys
import numpy as np
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset
from core import paths

DIT_DIR = os.path.join(paths.repo_dir("unilm"), "dit/object_detection")
PUBLAYNET = ["text", "title", "list", "table", "figure"]
MIRRORS = [("tensorlake/dit_cascade-publaynet", "publaynet_dit-b_cascade.pth"),
           ("discus0434/publaynet-dit-base", "publaynet_dit-b_cascade.pth")]


def mask_to_polygon(mask, max_pts=60):
    import cv2
    m = np.asarray(mask).astype("uint8")
    if m.sum() == 0:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(c) < 3:
        return None
    if len(c) > max_pts:
        c = c[np.linspace(0, len(c) - 1, max_pts).astype(int)]
    return [[float(x), float(y)] for x, y in c]


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg_j = run.cfg
    thr = float(cfg_j.get("threshold", 0.5))

    t = Timer()
    with t.phase("model_load"):
        sys.path.insert(0, DIT_DIR)
        # ditod/__init__ transitively imports its table-evaluation helper, which
        # still does `from collections import Iterable` (removed in Python 3.10).
        # Restore the aliases rather than editing the vendored repository; the
        # same shim is used by the UniLM adapter.
        import collections, collections.abc
        for _n in ("Iterable", "Mapping", "MutableMapping", "Sequence", "Callable"):
            if not hasattr(collections, _n):
                setattr(collections, _n, getattr(collections.abc, _n))
        import torch, cv2
        from huggingface_hub import hf_hub_download
        from ditod import add_vit_config
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        wpath, src_repo = None, None
        for repo, fname in MIRRORS:
            try:
                wpath, src_repo = hf_hub_download(repo, fname), repo
                break
            except Exception:
                continue
        if wpath is None:
            raise RuntimeError("no reachable DiT checkpoint mirror")

        cfg = get_cfg()
        add_vit_config(cfg)
        cfg.merge_from_file(os.path.join(DIT_DIR, cfg_j["config"]))
        cfg.MODEL.WEIGHTS = wpath
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thr
        predictor = DefaultPredictor(cfg)
        img0 = cv2.imread(job["pages"][0]["input_path"])
        predictor(img0)                                            # warm-up
        cuda_sync()
    run.model_load_s = t.pop()["total_s"]

    run.set_model_info(
        repo_id=src_repo, weights=os.path.basename(wpath), local_path=wpath,
        sha256="06a562f5cc0038f903b4c0c983402383be234e7a0cd433c29d3dc27b06e3ca2f",
        provenance=("official Microsoft URL dead (PublicAccessNotPermitted); "
                    "identical file mirrored independently by two HF accounts"),
        config=cfg_j["config"], labels=PUBLAYNET,
        framework="detectron2 0.6 + ditod (BEiT-style Document Image Transformer)",
        architecture="DiT-base + Cascade Mask R-CNN", training_set="PubLayNet",
        device=cfg.MODEL.DEVICE, threshold=thr)

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                img = cv2.imread(page["input_path"])
            with t.phase("inference"):
                out = predictor(img)["instances"].to("cpu")
                cuda_sync()
            with t.phase("postprocess"):
                boxes = out.pred_boxes.tensor.numpy()
                scores = out.scores.numpy()
                classes = out.pred_classes.numpy()
                masks = out.pred_masks.numpy() if out.has("pred_masks") else None
                items = []
                for i in range(len(scores)):
                    it = {"source_class": PUBLAYNET[int(classes[i])],
                          "bbox": [float(v) for v in boxes[i]],
                          "confidence": float(scores[i])}
                    if masks is not None:
                        poly = mask_to_polygon(masks[i])
                        if poly:
                            it["polygon"] = poly
                    items.append(it)
                regions = build_regions(run.taxonomy, items)
                raw = {"boxes": boxes.tolist(), "scores": scores.tolist(),
                       "classes": [int(c) for c in classes], "names": PUBLAYNET,
                       "has_masks": masks is not None}
            run.emit(page, regions, t.pop(), raw=raw, meta={"threshold": thr})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
