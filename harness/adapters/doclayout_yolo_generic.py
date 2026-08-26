#!/usr/bin/env python3
"""DocLayout-YOLO adapter — any published checkpoint of the opendatalab model.

DocLayout-YOLO is a YOLOv10 fork with document-specific changes (DocSynth-300K
synthetic pretraining, global-to-local adaptive perception).  The authors publish
several checkpoints that differ only in *training distribution*, which is exactly
the variable this benchmark can measure:

  DocStructBench   diverse real documents  (10 classes)
  DocLayNet        the DocLayNet 11 classes, with and without DocSynth-300K pretraining
  D4LA             27 business/letter-oriented classes

The class map is read from the checkpoint itself (`model.names`) rather than
hard-coded, so a new checkpoint needs only a registry entry plus a taxonomy
mapping if its label set is new.  Inference parameters come from the config, with
each checkpoint's own training `imgsz` as the default.
"""
import os, sys
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg = run.cfg
    repo, fname = cfg["repo_id"], cfg["weights"]
    imgsz = int(cfg.get("img_size", 1024))
    conf = float(cfg.get("conf_thres", 0.25))
    iou = float(cfg.get("iou_thres", 0.45))

    t = Timer()
    with t.phase("model_load"):
        from huggingface_hub import hf_hub_download
        import torch
        wpath = hf_hub_download(repo, fname)
        try:
            from doclayout_yolo import YOLOv10 as M
            backend = "doclayout_yolo.YOLOv10"
        except Exception:
            from ultralytics import YOLO as M
            backend = "ultralytics.YOLO"
        model = M(wpath)
        dev = 0 if torch.cuda.is_available() else "cpu"
        # Warm-up: excluded from the per-page timings.
        model.predict(job["pages"][0]["input_path"], imgsz=imgsz, conf=conf,
                      iou=iou, verbose=False, device=dev)
        cuda_sync()
    run.model_load_s = t.pop()["total_s"]

    run.set_model_info(repo_id=repo, weights=fname, local_path=wpath,
                       labels=model.names, framework=backend,
                       architecture="YOLOv10 (DocLayout-YOLO fork)",
                       training_set=cfg.get("training_set"),
                       docsynth_pretrained=cfg.get("docsynth_pretrained"),
                       img_size=imgsz, conf_thres=conf, iou_thres=iou)

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("inference"):
                r = model.predict(page["input_path"], imgsz=imgsz, conf=conf,
                                  iou=iou, verbose=False, device=dev)[0]
                cuda_sync()
            with t.phase("postprocess"):
                b = r.boxes
                items = []
                for xyxy, c, cid in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist()):
                    items.append({"source_class": model.names[int(cid)],
                                  "bbox": [float(v) for v in xyxy],
                                  "confidence": float(c)})
                regions = build_regions(run.taxonomy, items)
                raw = {"boxes": b.xyxy.tolist(), "conf": b.conf.tolist(),
                       "cls": [int(v) for v in b.cls.tolist()], "names": model.names}
            run.emit(page, regions, t.pop(), raw=raw,
                     meta={"img_size": imgsz, "conf": conf, "iou": iou})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
