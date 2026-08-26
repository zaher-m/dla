#!/usr/bin/env python3
"""PaddleOCR layout adapter: PP-DocLayout family under ONNX Runtime.

PaddlePaddle's CPU build segfaults on these RT-DETR graphs on this platform and
there is no CUDA build for it, so the official weights are run through the
official paddle2onnx export instead. Preprocessing and the label list are taken
from each model's own inference.yml, so only the executor differs.
"""
import os, sys, json, yaml
import numpy as np
from PIL import Image
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions
from core import paths

MODELS_DIR = paths.model_dir("paddle")


def load_spec(model_name, onnx_repo=None):
    """Resolve weights + preprocessing spec.

    PaddlePaddle now publishes official ONNX exports for some models
    (`PP-DocLayoutV3_onnx`), which carry `inference.onnx` and `inference.yml`
    together — preferred, because it removes our paddle2onnx conversion step from
    the provenance chain entirely.  Everything else still uses the local export
    produced by the official converter.
    """
    from huggingface_hub import snapshot_download
    if onnx_repo:
        src = snapshot_download(onnx_repo)
        spec = yaml.safe_load(open(os.path.join(src, "inference.yml")))
        return src, spec, os.path.join(src, "inference.onnx")
    src = snapshot_download(f"PaddlePaddle/{model_name}")
    spec = yaml.safe_load(open(os.path.join(src, "inference.yml")))
    return src, spec, os.path.join(MODELS_DIR, f"{model_name}.onnx")


def preprocess(img, spec):
    """Replicate the Preprocess chain declared in the model's inference.yml."""
    arr = np.asarray(img.convert("RGB")).astype("float32")
    h0, w0 = arr.shape[:2]
    target = None
    for op in spec["Preprocess"]:
        if op["type"] == "Resize":
            target = op["target_size"]                      # [w, h]
            keep = op.get("keep_ratio", False)
            interp = op.get("interp", 2)
            resample = {0: Image.NEAREST, 1: Image.LANCZOS, 2: Image.BILINEAR,
                        3: Image.BICUBIC, 4: Image.BOX}.get(interp, Image.BILINEAR)
            if keep:
                s = min(target[0] / w0, target[1] / h0)
                nw, nh = int(round(w0 * s)), int(round(h0 * s))
            else:
                nw, nh = target[0], target[1]
            arr = np.asarray(img.convert("RGB").resize((nw, nh), resample)).astype("float32")
        elif op["type"] == "NormalizeImage":
            mean = np.array(op["mean"], dtype="float32")
            std = np.array(op["std"], dtype="float32")
            # PaddleDetection's NormalizeImage defaults is_scale=True, so the
            # /255 happens even when norm_type is "none" (mean=0, std=1).
            if op.get("is_scale", True):
                arr = arr / 255.0
            arr = (arr - mean) / std
        elif op["type"] == "Permute":
            arr = arr.transpose(2, 0, 1)
    chw = arr if arr.ndim == 3 and arr.shape[0] == 3 else arr.transpose(2, 0, 1)
    nh, nw = chw.shape[1], chw.shape[2]
    scale_factor = np.array([[nh / h0, nw / w0]], dtype="float32")
    im_shape = np.array([[nh, nw]], dtype="float32")
    return chw[None].astype("float32"), im_shape, scale_factor, (w0, h0)



def mask_polygon(mask, w0, h0, max_pts=60):
    """Largest contour of one binary instance mask, in page pixels.

    PP-DocLayoutV3 returns a full-page mask grid per instance (the 800x800 input
    downsampled 4x to 200x200), so the mapping to page coordinates is a plain
    scale by (w0/W, h0/H) -- verified against the model's own boxes, which the
    contour extents reproduce to within a few pixels.  Returns None when the mask
    is empty or degenerates to its own bounding rectangle, so a box dressed up as
    a polygon is never reported as segmentation.
    """
    import cv2
    m = (mask > 0).astype("uint8")
    if m.sum() == 0:
        return None
    H, W = m.shape
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    eps = 0.002 * cv2.arcLength(c, True)
    c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(c) < 3:
        return None
    if len(c) > max_pts:
        idx = np.linspace(0, len(c) - 1, max_pts).astype(int)
        c = c[idx]
    sx, sy = w0 / W, h0 / H
    return [[float(x * sx), float(y * sy)] for x, y in c]


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg = run.cfg
    name = cfg["model"]
    thr = float(cfg.get("threshold", 0.5))

    t = Timer()
    with t.phase("model_load"):
        import onnxruntime as ort
        src, spec, onnx_path = load_spec(name, cfg.get("onnx_repo"))
        labels = spec["label_list"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(cfg.get("cpu_threads", 8))
        sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        innames = [i.name for i in sess.get_inputs()]
    run.model_load_s = t.pop()["total_s"]
    run.set_model_info(model_name=name, hf_repo=f"PaddlePaddle/{name}",
                       weights_dir=src, onnx=onnx_path,
                       framework=("onnxruntime (official PaddlePaddle ONNX release)"
                                  if cfg.get("onnx_repo") else
                                  "onnxruntime (official paddle2onnx export of official Paddle weights)"),
                       onnx_source=cfg.get("onnx_repo") or "local paddle2onnx export",
                       device="cpu (no aarch64 CUDA PaddlePaddle build; native aarch64 CPU Paddle segfaults)",
                       labels=labels, preprocess=spec["Preprocess"],
                       arch=spec.get("arch"), onnx_inputs=innames,
                       config_applied={"threshold": thr})

    def infer(path):
        img = Image.open(path)
        x, im_shape, sf, (w0, h0) = preprocess(img, spec)
        feed = {}
        for n in innames:
            feed[n] = {"image": x, "im_shape": im_shape, "scale_factor": sf}.get(n, x)
        return sess.run(None, feed), (w0, h0), (x.shape[3], x.shape[2])

    infer(job["pages"][0]["input_path"])                     # warm-up

    for page in job["pages"]:
        try:
            with t.phase("preprocess"):
                img = Image.open(page["input_path"])
                x, im_shape, sf, (w0, h0) = preprocess(img, spec)
                feed = {n: {"image": x, "im_shape": im_shape, "scale_factor": sf}.get(n, x)
                        for n in innames}
            with t.phase("inference"):
                outs = sess.run(None, feed)
            with t.phase("postprocess"):
                # PP-DocLayout* emit (N,6); PP-DocLayoutV2 emits (N,8) where the
                # trailing two columns are decoder query indices, NOT reading
                # order -- the reading-order head is not part of the exported
                # graph, so no ordering is claimed here.
                # PP-DocLayout*      -> (N,6)
                # PP-DocLayoutV2     -> (N,8); cols 6/7 are decoder query
                #                       indices, NOT reading order
                # PP-DocLayoutV3     -> (N,7) + an (N,H,W) binary mask stack:
                #                       a real instance-segmentation head
                dets, masks, n_valid = None, None, None
                for o in outs:
                    if not isinstance(o, np.ndarray):
                        continue
                    if o.ndim == 2 and o.shape[-1] in (6, 7, 8) and dets is None:
                        dets = o
                    elif o.ndim == 3 and o.shape[0] == (dets.shape[0] if dets is not None else -1):
                        masks = o
                    elif o.ndim == 1 and o.size == 1 and np.issubdtype(o.dtype, np.integer):
                        n_valid = int(o[0])
                if dets is not None and n_valid is not None:
                    dets = dets[:n_valid]
                    if masks is not None:
                        masks = masks[:n_valid]
                items = []
                if dets is not None:
                    for i, row in enumerate(dets):
                        cls_id, score = row[0], row[1]
                        x1, y1, x2, y2 = row[2], row[3], row[4], row[5]
                        if score < thr or int(cls_id) < 0:
                            continue
                        it = {"source_class": labels[int(cls_id)],
                              "bbox": [float(x1), float(y1), float(x2), float(y2)],
                              "confidence": float(score)}
                        if row.shape[0] == 8:
                            it["extra"] = {"query_index": [float(row[6]), float(row[7])]}
                        elif row.shape[0] == 7:
                            it["extra"] = {"query_index": float(row[6])}
                        if masks is not None and cfg.get("masks", True):
                            poly = mask_polygon(masks[i], w0, h0)
                            if poly:
                                it["polygon"] = poly
                        items.append(it)
                regions = build_regions(run.taxonomy, items)
                raw = {"dets": dets.tolist() if dets is not None else None,
                       "labels": labels, "input_hw": [x.shape[2], x.shape[3]],
                       "orig_wh": [w0, h0],
                       "n_masks": int(masks.shape[0]) if masks is not None else 0,
                       "mask_grid": list(masks.shape[1:]) if masks is not None else None}
            run.emit(page, regions, t.pop(), raw=raw,
                     meta={"model": name, "device": "cpu", "runtime": "onnxruntime"})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
