#!/usr/bin/env python3
"""Layout-Parser adapter — Detectron2 backend with the official `lp://` model
catalog (PubLayNet / PRImA / HJDataset / NewspaperNavigator / TableBank).

Mask R-CNN checkpoints produce instance masks; those are preserved as
polygons so the segmentation comparison is not limited to boxes.
"""
import os, sys
import numpy as np
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset

LABEL_MAPS = {
    "PubLayNet": {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    "PrimaLayout": {1: "TextRegion", 2: "ImageRegion", 3: "TableRegion",
                    4: "MathsRegion", 5: "SeparatorRegion", 6: "OtherRegion"},
    "HJDataset": {1: "Page Frame", 2: "Row", 3: "Title Region", 4: "Text Region",
                  5: "Title", 6: "Subtitle", 7: "Other"},
    "NewspaperNavigator": {0: "Photograph", 1: "Illustration", 2: "Map",
                           3: "Comics/Cartoon", 4: "Editorial Cartoon",
                           5: "Headline", 6: "Advertisement"},
    "TableBank": {0: "Table"},
}


def mask_to_polygon(mask):
    import cv2
    m = (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    eps = 0.002 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in approx] if len(approx) >= 3 else None


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg = run.cfg
    ds, arch = cfg["dataset"], cfg["arch"]
    want_masks = bool(cfg.get("masks"))
    t = Timer()
    with t.phase("model_load"):
        import layoutparser as lp
        import torch
        from huggingface_hub import hf_hub_download
        # The `lp://` catalog points at Dropbox URLs that now return an HTML
        # interstitial instead of the file (verified 2026-08-29), so the
        # identical checkpoints are pulled from Layout-Parser's own Hugging
        # Face mirror `layoutparser/detectron2`.
        cfg_path = hf_hub_download("layoutparser/detectron2", f"{ds}/{arch}/config.yml")
        wt_path = hf_hub_download("layoutparser/detectron2", f"{ds}/{arch}/model_final.pth")
        label_map = LABEL_MAPS[ds]
        model = lp.Detectron2LayoutModel(
            cfg_path, model_path=wt_path,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", float(cfg.get("score_thresh", 0.5)),
                          "MODEL.DEVICE", "cuda" if torch.cuda.is_available() else "cpu"],
            label_map=label_map)
    run.model_load_s = t.pop()["total_s"]
    run.set_model_info(catalog=f"lp://{ds}/{arch}", dataset=ds, arch=arch,
                       weights_source="hf:layoutparser/detectron2 (official mirror; "
                                      "the repo's Dropbox catalog URLs are dead)",
                       config_path=cfg_path, weights_path=wt_path,
                       label_map=label_map, framework="detectron2",
                       score_thresh=cfg.get("score_thresh", 0.5),
                       masks=want_masks, layoutparser=lp.__version__)

    import cv2
    warm = cv2.imread(job["pages"][0]["input_path"])[:, :, ::-1]
    model.detect(warm); cuda_sync()

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                img = cv2.imread(page["input_path"])[:, :, ::-1]
            with t.phase("inference"):
                # Layout-Parser's own `gather_output` drops instance masks even for
                # Mask R-CNN checkpoints, so call its underlying detectron2
                # predictor directly to keep them.  Same model, same weights, same
                # config -- only the output marshalling differs.
                outputs = model.model(model.image_loader(img))
                cuda_sync()
            with t.phase("postprocess"):
                inst = outputs["instances"].to("cpu")
                boxes = inst.pred_boxes.tensor.tolist()
                scores = inst.scores.tolist()
                labels = inst.pred_classes.tolist()
                masks = inst.pred_masks.numpy() if inst.has("pred_masks") else None
                items, raw = [], []
                for i, (box, sc, lb) in enumerate(zip(boxes, scores, labels)):
                    x1, y1, x2, y2 = box
                    name = model.label_map.get(lb, lb)
                    poly = None
                    if want_masks and masks is not None:
                        poly = mask_to_polygon(masks[i])
                    items.append({"source_class": name, "bbox": [x1, y1, x2, y2],
                                  "confidence": sc, "polygon": poly})
                    raw.append({"type": name, "score": float(sc),
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "has_mask": poly is not None,
                                "mask_area_px": int(masks[i].sum()) if masks is not None else None})
                regions = build_regions(run.taxonomy, items)
            run.emit(page, regions, t.pop(), raw={"blocks": raw},
                     meta={"catalog": f"lp://{ds}/{arch}"})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
