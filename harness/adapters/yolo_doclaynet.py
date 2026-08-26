#!/usr/bin/env python3
"""Ultralytics-format DocLayNet detector adapter.

`repo_id` selects the Hugging Face repository the weights come from, or
`weights_url` fetches a checkpoint published as a release asset. That covers
several independent DocLayNet YOLO trainings with one adapter:

  hantian/yolo-doclaynet                     v8x / v12l / v26l
  Armaggheddon/yolo{11,26}-document-layout   DocLayNet v1.2
  moured/YOLOv11-Document-Layout-Analysis    an independent v1 training

Class names are read from the checkpoint, never hardcoded.
"""
import os, sys
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset
from core import paths

REPO = "hantian/yolo-doclaynet"


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg = run.cfg
    t = Timer()
    with t.phase("model_load"):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
        import torch
        if cfg.get("weights_url"):
            # Published as a GitHub release asset rather than on the Hub.
            import urllib.request
            dest = os.path.join(paths.model_dir("yolo"), cfg["weights"])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not os.path.exists(dest):
                urllib.request.urlretrieve(cfg["weights_url"], dest)
            wpath = dest
        else:
            wpath = hf_hub_download(cfg.get("repo_id", REPO), cfg["weights"])
        model = YOLO(wpath)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    run.model_load_s = t.pop()["total_s"]
    names = model.names
    import hashlib
    with open(wpath, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    run.set_model_info(repo_id=cfg.get("repo_id", REPO), weights=cfg["weights"],
                       local_path=wpath, sha256=sha,
                       weights_url=cfg.get("weights_url", ""),
                       labels=names, framework="ultralytics",
                       architecture=cfg.get("architecture", ""),
                       training_set=cfg.get("training_set", "DocLayNet"),
                       n_params=sum(p.numel() for p in model.model.parameters()))

    warm = job["pages"][0]["input_path"]
    model.predict(warm, imgsz=cfg["imgsz"], conf=cfg["conf"], iou=cfg["iou"], verbose=False)
    cuda_sync()

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("inference"):
                r = model.predict(page["input_path"], imgsz=cfg["imgsz"], conf=cfg["conf"],
                                  iou=cfg["iou"], verbose=False, device=model.device)[0]
                cuda_sync()
            with t.phase("postprocess"):
                b = r.boxes
                xyxy = b.xyxy.cpu().tolist(); cls = b.cls.cpu().tolist(); conf = b.conf.cpu().tolist()
                raw = {"boxes": xyxy, "classes": cls, "scores": conf, "names": names,
                       "orig_shape": list(r.orig_shape), "speed_ms": r.speed}
                items = [{"source_class": names[int(c)], "bbox": bb, "confidence": s}
                         for bb, c, s in zip(xyxy, cls, conf)]
                regions = build_regions(run.taxonomy, items)
            run.emit(page, regions, t.pop(), raw=raw, meta={"weights": cfg["weights"]})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
