#!/usr/bin/env python3
"""DocSAM adapter: Mask2Former decoder with class names as semantic queries.

Class names are embedded with a sentence encoder and used as queries, so the
class set is an inference-time argument rather than a fixed head. The registry
runs the same weights under different prompts to measure whether that matters.

datasets/dataset.py is not used; its inference-stage preprocessing is
reproduced here (BGR 0-255, short side resized to the mean of `short_range`,
long side capped at twice that, area interpolation, dataset name prefixed onto
the background token). Everything after that is upstream's `predict_slide_window`.
jpeg4py is stubbed: it needs libturbojpeg and is only imported by the unused
COCO loader.

Results come back in the resized frame and are scaled to page pixels here.
"""
import os, sys, hashlib, types
import numpy as np
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset
from core import paths

REPO = paths.repo_dir("DocSAM")
STAGE = paths.model_dir("docsam")      # holds pretrained_model/ + checkpoints


def mask_to_polygon(mask, sx, sy, max_pts=60):
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
    return [[float(x * sx), float(y * sy)] for x, y in c]


def load_page(path, short_range, keep_size=False):
    """Upstream `_load_image` + `_data_resize`, test-time branch."""
    import cv2, torch
    import torch.nn.functional as F
    img = cv2.imread(path, cv2.IMREAD_COLOR)          # BGR, as upstream
    if img is None:
        raise RuntimeError(f"cv2 could not read {path}")
    h0, w0 = img.shape[:2]
    image = torch.from_numpy(img.transpose((2, 0, 1))).float()
    mask = torch.ones((1, h0, w0)).float()
    low, high = short_range
    if keep_size:
        hei, wid = h0, w0
        if min(hei, wid) < low:
            scale = low / min(hei, wid)
        elif min(hei, wid) > high:
            scale = high / min(hei, wid)
        else:
            scale = 1.0
        hei, wid = round(hei * scale), round(wid * scale)
        hei, wid = min(hei, min(hei, wid) * 2), min(wid, min(hei, wid) * 2)
    else:
        short_side = (low + high) // 2
        scale = short_side / min(h0, w0)
        hei = min(round(h0 * scale), short_side * 2)
        wid = min(round(w0 * scale), short_side * 2)
    image = F.interpolate(image[None], size=(hei, wid), mode="area")[0]
    mask = F.interpolate(mask[None], size=(hei, wid), mode="nearest")[0]
    return image, mask, (w0, h0), (wid, hei)


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg = run.cfg
    thr = float(cfg.get("threshold", 0.5))
    model_size = cfg.get("model_size", "large")
    short_range = tuple(cfg.get("short_range", [704, 896]))
    patch_size = tuple(cfg.get("patch_size", [640, 640]))
    keep_size = bool(cfg.get("keep_size", False))
    dataset_name = cfg.get("dataset_name", "Document")
    class_names = list(cfg["class_names"]) + ["_background_"]
    wpath = os.path.join(STAGE, cfg["weights"])

    t = Timer()
    with t.phase("model_load"):
        import torch
        # `datasets/dataset.py` imports jpeg4py at module scope purely for its
        # COCO ground-truth loader, which this adapter never calls.  libturbojpeg
        # is absent from the container and unobtainable without root, so the
        # module is stubbed; if anything ever *did* call it the AttributeError
        # would be immediate and loud.
        sys.modules.setdefault("jpeg4py", types.ModuleType("jpeg4py"))
        sys.path.insert(0, REPO)
        # Upstream hardcodes relative paths for the Mask2Former and sentence
        # encoders ("./pretrained_model/..."), so run from the staging dir.
        os.chdir(STAGE)
        from models.DocSAM import DocSAM
        import test as docsam_test

        model = DocSAM(model_size=model_size)
        model = docsam_test.load_para_weights(model, wpath)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
    run.model_load_s = t.pop()["total_s"]

    with open(wpath, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    run.set_model_info(
        weights=cfg["weights"], local_path=wpath, sha256=sha,
        architecture=(f"DocSAM ({model_size}): Swin-{model_size} + Mask2Former decoder with "
                      "900 queries, sentence-embedded class names as semantic queries"),
        text_encoder="sentence-transformers/all-MiniLM-L6-v2",
        visual_init=f"facebook/mask2former-swin-{model_size}-coco-panoptic",
        prompt_classes=class_names, dataset_name_token=dataset_name,
        short_range=list(short_range), patch_size=list(patch_size), keep_size=keep_size,
        threshold=thr, device=device,
        n_params=sum(p.numel() for p in model.parameters()),
        output="per-instance binary masks + boxes + a semantic mask stack",
        framework="transformers 4.49.0 (vendored Mask2Former fork)",
        provenance=("checkpoint from the Google Drive links in MODELZOO.md of "
                    "github.com/xhli-git/DocSAM (MIT); no upstream checksum is "
                    "published, so the sha256 recorded here is of the file as downloaded"),
        deviations=[
            "datasets/dataset.py is bypassed; its inference-stage preprocessing is "
            "reproduced in the adapter so pages can be fed directly (see module docstring)",
            "jpeg4py stubbed — needs libturbojpeg, absent from the container; it is only "
            "imported by the unused COCO loader",
            "torch_xla (upstream requirements.txt) not installed: it is a TPU runtime and "
            "is never imported on the inference path",
        ])

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                image, pmask, (w0, h0), (wr, hr) = load_page(
                    page["input_path"], short_range, keep_size)
                batch = {
                    "pixel_values": [image.to(device)],
                    "pixel_mask": [pmask.to(device)],
                    "class_names": [list(class_names[:-1]) + [f"{dataset_name} _background_"]],
                    "image_bboxes": [torch.tensor([0, 0, wr, hr]).long()],
                    "image_names": [page["page_id"]],
                    "dataset_names": [dataset_name],
                }
            with t.phase("inference"):
                with torch.no_grad():
                    results = docsam_test.predict_slide_window(
                        model, batch, patch_size=patch_size)[0]
                cuda_sync()
            with t.phase("postprocess"):
                sx, sy = w0 / wr, h0 / hr
                labels = results["instance_labels"].cpu().numpy()
                scores = results["instance_scores"].cpu().numpy()
                bboxes = results["instance_bboxes"].cpu().numpy()
                masks = results["instance_maskes"].cpu().numpy()
                items, raw = [], []
                n_masks = 0
                for j in range(len(scores)):
                    s = float(scores[j])
                    if s < thr:
                        continue
                    name = class_names[int(labels[j]) - 1]
                    x, y, w, h = [float(v) for v in bboxes[j]]
                    bb = [x * sx, y * sy, (x + w) * sx, (y + h) * sy]
                    it = {"source_class": name, "bbox": bb, "confidence": s}
                    poly = mask_to_polygon(masks[j], sx, sy)
                    if poly:
                        it["polygon"] = poly
                        n_masks += 1
                    items.append(it)
                    raw.append({"class": name, "bbox_xywh": [x, y, w, h], "score": s})
                sem = results["semantic_maskes"].cpu().numpy()
                sem_cov = [float(sem[c].mean()) for c in range(sem.shape[0])]
                regions = build_regions(run.taxonomy, items)
            run.emit(page, regions, t.pop(),
                     raw={"instances": raw, "prompt_classes": class_names,
                          "resized_wh": [wr, hr], "orig_wh": [w0, h0],
                          "semantic_coverage": dict(zip(class_names[:-1], sem_cov))},
                     meta={"threshold": thr, "n_masks": n_masks,
                           "prompt_classes": class_names})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
