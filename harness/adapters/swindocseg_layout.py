#!/usr/bin/env python3
"""SwinDocSegmenter adapter: Swin-L + MaskDINO, DocLayNet. Emits instance masks.

MaskDINO's MSDeformAttn CUDA op does not compile against torch 2.x as shipped;
it is patched and rebuilt in harness/setup/swindocseg.sh. Its documented
pure-PyTorch fallback is unreachable (the import re-raises), so the kernel is
required.
"""
import os, sys
import numpy as np
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset
from core import paths

REPO = paths.repo_dir("SwinDocSegmenter")
CKPT = os.path.join(paths.model_dir("swindocseg"), "model_final_doclay_swindocseg.pth")
DOCLAYNET = ["Caption", "Footnote", "Formula", "List-item", "Page-footer",
             "Page-header", "Picture", "Section-header", "Table", "Text", "Title"]


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
        sys.path.insert(0, REPO)
        import warnings
        warnings.filterwarnings("ignore")
        import torch, cv2
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.engine import DefaultPredictor
        from maskdino import add_maskformer2_config

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(os.path.join(REPO, cfg_j["config"]))
        cfg.MODEL.WEIGHTS = CKPT
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = thr
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thr
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = thr
        cfg.freeze()
        predictor = DefaultPredictor(cfg)
        img0 = cv2.imread(job["pages"][0]["input_path"])
        predictor(img0)                                            # warm-up
        cuda_sync()
    run.model_load_s = t.pop()["total_s"]

    run.set_model_info(
        checkpoint=CKPT,
        sha256="f54cc1ccb579006cd221141026be8a63be1284065ff332190dd93d35a9c40308",
        provenance="Google Drive (repository Model Zoo); no Hugging Face mirror published",
        config=cfg_j["config"], labels=DOCLAYNET,
        framework="detectron2 + vendored MaskDINO",
        architecture="Swin-L (patch4 window12 384-22k) + MaskDINO decoder",
        training_set="DocLayNet", output="instance masks + boxes",
        device=cfg.MODEL.DEVICE, threshold=thr,
        msdeformattn="CUDA kernel rebuilt from source; validated against "
                     "ms_deform_attn_core_pytorch, max abs err 1.45e-9")

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                img = cv2.imread(page["input_path"])
            with t.phase("inference"):
                out = predictor(img)["instances"].to("cpu")
                cuda_sync()
            with t.phase("postprocess"):
                scores = out.scores.numpy()
                classes = out.pred_classes.numpy()
                boxes = (out.pred_boxes.tensor.numpy() if out.has("pred_boxes") else None)
                masks = out.pred_masks.numpy() if out.has("pred_masks") else None
                items = []
                for i in range(len(scores)):
                    if scores[i] < thr:
                        continue
                    if boxes is not None:
                        bb = [float(v) for v in boxes[i]]
                    else:
                        ys, xs = np.nonzero(masks[i])
                        if len(xs) == 0:
                            continue
                        bb = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                    idx = int(classes[i])
                    it = {"source_class": DOCLAYNET[idx] if idx < len(DOCLAYNET) else str(idx),
                          "bbox": bb, "confidence": float(scores[i])}
                    if masks is not None:
                        poly = mask_to_polygon(masks[i])
                        if poly:
                            it["polygon"] = poly
                    items.append(it)
                regions = build_regions(run.taxonomy, items)
                raw = {"scores": scores.tolist(), "classes": [int(c) for c in classes],
                       "boxes": boxes.tolist() if boxes is not None else None,
                       "names": DOCLAYNET, "has_masks": masks is not None}
            run.emit(page, regions, t.pop(), raw=raw, meta={"threshold": thr})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
