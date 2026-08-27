#!/usr/bin/env python3
"""RoDLA adapter: InternImage-XL + DINO head, trained on DocLayNet. Boxes only.

Upstream pins Python 3.7 / torch 1.10.2+cu113 / mmcv-full 1.5.0 and ships DCNv3
as a prebuilt cp37 x86_64 wheel. None of that installs here, so the port uses
mmcv-full 1.7.2 (the last 1.x that imports on torch 2.x) + mmdet 2.28.1, with
DCNv3 rebuilt from source. See harness/setup/rodla.sh.
"""
import os, sys, hashlib
import numpy as np
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset
from core import paths

RODLA_DIR = os.path.join(paths.repo_dir("RoDLA"), "model")
MODELS = paths.model_dir("rodla")


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg_j = run.cfg
    thr = float(cfg_j.get("threshold", 0.3))
    wpath = os.path.join(MODELS, cfg_j["weights"])

    t = Timer()
    with t.phase("model_load"):
        sys.path.insert(0, RODLA_DIR)
        import torch, mmcv, mmdet
        from mmcv import Config
        from mmdet.apis import init_detector, inference_detector
        import mmdet_custom, mmcv_custom            # noqa: F401 — registry side effects
        import DCNv3                                 # noqa: F401 — fail loudly if absent

        cfg = Config.fromfile(os.path.join(RODLA_DIR, cfg_j["config"]))
        # The backbone's init_cfg points at a 1.3 GB ImageNet-22k InternImage
        # checkpoint used only to *start* training; the fine-tuned weights
        # overwrite it a moment later.  Dropping it avoids a pointless download.
        cfg.model.backbone.init_cfg = None
        cfg.model.train_cfg = None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = init_detector(cfg, wpath, device=device)
        # mmdet 2.x silently falls back to the 80 COCO class names when a
        # checkpoint carries no CLASSES metadata, which would misname every
        # detection.  Take the names from the dataset class the config names.
        from mmdet.datasets import DATASETS
        want = list(DATASETS.get(cfg.data.test.type).CLASSES)
        labels = list(getattr(model, "CLASSES", None) or want)
        if len(labels) != cfg.model.bbox_head.num_classes:
            labels = want
        if len(labels) != cfg.model.bbox_head.num_classes:
            raise RuntimeError(f"class-name/head mismatch: {len(labels)} names for "
                               f"{cfg.model.bbox_head.num_classes} outputs")
        img0 = job["pages"][0]["input_path"]
        inference_detector(model, img0)              # warm-up
        cuda_sync()
    run.model_load_s = t.pop()["total_s"]

    with open(wpath, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    run.set_model_info(
        repo_id="yufanchen96/RoDLA", weights=cfg_j["weights"], local_path=wpath,
        sha256=sha, config=cfg_j["config"], labels=labels,
        framework="mmdet 2.28.1 + mmcv-full 1.7.2 (both built from source)",
        architecture="InternImage-XL (DCNv3) + DINO head, 339.0M params, boxes only",
        training_set="DocLayNet", device=device, threshold=thr,
        n_params=sum(p.numel() for p in model.parameters()),
        deviations=[
            "upstream's Python 3.7 / torch 1.10.2+cu113 / mmcv-full 1.5.0 stack cannot "
            "install on this platform; ported to mmcv-full 1.7.2 + mmdet 2.28.1, both built "
            "from source",
            "DCNv3 kernel rebuilt from source (the shipped wheel is cp37/x86_64) "
            "after patching torch-2.x-removed APIs; validated against the pure-PyTorch "
            "reference at 1.75e-09 max abs err",
            "timm unpinned from 0.6.11, which cannot be imported on Python 3.12",
        ],
        provenance=("Google Drive link from the repository README "
                    "(id 18U6agKsUwU4I3__g8OXUwMghK41esS8h); no checksum is published "
                    "upstream, so the sha256 recorded here is of the file as downloaded"))

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                path = page["input_path"]
            with t.phase("inference"):
                res = inference_detector(model, path)
                cuda_sync()
            with t.phase("postprocess"):
                # mmdet 2.x: list over classes of (N, 5) [x1,y1,x2,y2,score]
                dets = res[0] if isinstance(res, tuple) else res
                items, raw_b, raw_s, raw_c = [], [], [], []
                for ci, arr in enumerate(dets):
                    arr = np.asarray(arr)
                    for row in arr:
                        if float(row[4]) < thr:
                            continue
                        bb = [float(v) for v in row[:4]]
                        items.append({"source_class": labels[ci], "bbox": bb,
                                      "confidence": float(row[4])})
                        raw_b.append(bb); raw_s.append(float(row[4])); raw_c.append(ci)
                regions = build_regions(run.taxonomy, items)
                raw = {"boxes": raw_b, "scores": raw_s, "classes": raw_c, "names": labels}
            run.emit(page, regions, t.pop(), raw=raw, meta={"threshold": thr})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
