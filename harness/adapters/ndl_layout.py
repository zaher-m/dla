#!/usr/bin/env python3
"""ndl_layout adapter: Cascade Mask R-CNN with a ConvNeXt-tiny backbone.

The layout module of NDLOCR v2.1, trained on Japanese printed and pre-modern
materials. 17 classes mixing line-level (`line_*`) and block-level regions, with
instance masks. `blocks_only` drops the line classes so the output can be
compared with systems that emit regions only.

mmdetection 2.x, so it reuses the mmcv-full 1.7.2 stack built for RoDLA plus
mmclassification for the backbone. The config's backbone init_cfg points at an
ImageNet checkpoint the detection weights overwrite, so it is disabled at load.
Class names come from the config rather than checkpoint metadata: mmdet 2.x
substitutes the 80 COCO names when a checkpoint carries none.
"""
import os, sys, hashlib
import numpy as np
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset
from core import paths

REPO = paths.repo_dir("ndl_layout")
MODELS = paths.model_dir("ndl_layout")

BLOCK_CLASSES = {"block_fig", "block_table", "block_pillar", "block_folio",
                 "block_rubi", "block_chart", "block_eqn", "block_cfm",
                 "block_eng", "block_ad", "text_block", "text_block_ad"}


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
    thr = float(cfg_j.get("threshold", 0.3))
    blocks_only = bool(cfg_j.get("blocks_only", False))
    wpath = os.path.join(MODELS, cfg_j.get("weights", "ndl_retrainmodel.pth"))
    cpath = os.path.join(REPO, cfg_j["config"])

    t = Timer()
    with t.phase("model_load"):
        import torch, mmcv, mmdet, mmcls          # noqa: F401 — mmcls for the backbone
        from mmcv import Config
        from mmdet.apis import init_detector, inference_detector

        cfg = Config.fromfile(cpath)
        cfg.model.backbone.init_cfg = None
        cfg.model.train_cfg = None
        labels = list(cfg.classes)
        n_out = cfg.model.roi_head.bbox_head[0].num_classes
        if len(labels) != n_out:
            raise RuntimeError(f"class-name/head mismatch: {len(labels)} names for {n_out} outputs")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = init_detector(cfg, wpath, device=device)
        model.CLASSES = tuple(labels)
        inference_detector(model, job["pages"][0]["input_path"])   # warm-up
        cuda_sync()
    run.model_load_s = t.pop()["total_s"]

    with open(wpath, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    run.set_model_info(
        weights=os.path.basename(wpath), local_path=wpath, sha256=sha,
        config=cfg_j["config"], labels=labels,
        framework="mmdet 2.28.1 + mmcv-full 1.7.2 + mmcls 0.25.0 (built from source)",
        architecture="Cascade Mask R-CNN, mmcls ConvNeXt-tiny backbone + FPN, 17 classes, instance masks",
        training_set="NDLOCR v2 Japanese printed and pre-modern materials (vertical script)",
        device=device, threshold=thr, blocks_only=blocks_only,
        n_params=sum(p.numel() for p in model.parameters()),
        provenance=("checkpoint https://lab.ndl.go.jp/dataset/ndlocr_v2/ndl_layout/"
                    "ndl_retrainmodel.pth, config from github.com/ndl-lab/ndl_layout "
                    "(CC BY 4.0); no upstream checksum is published, so the sha256 "
                    "recorded here is of the file as downloaded"),
        deviations=[
            "backbone.init_cfg disabled: it points at an ImageNet ConvNeXt checkpoint "
            "that the detection weights overwrite immediately",
            "class names taken from the config's `classes` tuple rather than checkpoint "
            "metadata, which mmdet 2.x would otherwise fill with the 80 COCO names",
        ])

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                path = page["input_path"]
            with t.phase("inference"):
                res = inference_detector(model, path)
                cuda_sync()
            with t.phase("postprocess"):
                # Cascade Mask R-CNN -> (bbox_results, segm_results), each a list
                # over classes; bbox rows are [x1, y1, x2, y2, score].
                if isinstance(res, tuple):
                    dets, segms = res[0], res[1]
                else:
                    dets, segms = res, None
                items, raw_b, raw_s, raw_c = [], [], [], []
                n_masks = 0
                for ci, arr in enumerate(dets):
                    name = labels[ci]
                    if blocks_only and name not in BLOCK_CLASSES:
                        continue
                    for ri, row in enumerate(np.asarray(arr)):
                        score = float(row[4])
                        if score < thr:
                            continue
                        bb = [float(v) for v in row[:4]]
                        it = {"source_class": name, "bbox": bb, "confidence": score}
                        if segms is not None and ci < len(segms) and ri < len(segms[ci]):
                            m = segms[ci][ri]
                            if not isinstance(m, np.ndarray):      # RLE
                                import pycocotools.mask as mask_utils
                                m = mask_utils.decode(m)
                            poly = mask_to_polygon(m)
                            if poly:
                                it["polygon"] = poly
                                n_masks += 1
                        items.append(it)
                        raw_b.append(bb); raw_s.append(score); raw_c.append(ci)
                regions = build_regions(run.taxonomy, items)
                raw = {"boxes": raw_b, "scores": raw_s, "classes": raw_c,
                       "names": labels, "n_masks": n_masks}
            run.emit(page, regions, t.pop(), raw=raw,
                     meta={"threshold": thr, "blocks_only": blocks_only, "n_masks": n_masks})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
