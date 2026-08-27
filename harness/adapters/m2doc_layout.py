#!/usr/bin/env python3
"""M2Doc adapter: DINO-4scale + ResNet-50 with multilingual text-line fusion.

The model takes line boxes and their text alongside the image. Lines come from
the PDF text layer via pdfplumber (upstream's own extractor) or PyMuPDF
(column- and cell-aware, rotation corrected), selected by `line_source`.
`blank_text` keeps the boxes and empties the strings, which isolates line
geometry from line content.

mmengine 0.10.7 calls torch.load without weights_only, which torch >= 2.6
refuses for this checkpoint; the call is wrapped for that one load only.
"""
import os, sys, hashlib
import numpy as np
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset
from core import paths

M2DOC_DIR = os.path.join(paths.repo_dir("M2Doc"), "mmdetection")
MODELS = paths.model_dir("m2doc")


def pdf_lines(pdf_path, img_w, img_h, source="pdfplumber"):
    """Text lines from the PDF's own content stream, in image pixel coords.

    Two extractors, because they disagree on what a "line" is and M2Doc is fed
    one box + one string per line:

    pdfplumber  what upstream's own tooling uses.  `extract_text_lines()` groups
                by baseline across the whole page width, so on a two-column page
                or a ruled table it merges both columns, or a whole table row, into
                a single line box: 39 lines where the reference finds 490 glyph
                line fragments on page_027.
    pymupdf     the content stream's own line segmentation, which respects columns
                and cells.  Closer to what an OCR line detector would emit, i.e.
                closer to the input M2Doc was trained on.
    """
    if source == "pymupdf":
        import pymupdf
        doc = pymupdf.open(pdf_path)
        page = doc[0]
        pw, ph = float(page.rect.width), float(page.rect.height)
        M = page.rotation_matrix
        raw = []
        for blk in page.get_text("rawdict")["blocks"]:
            if blk["type"] != 0:
                continue
            for ln in blk["lines"]:
                txt = "".join(ch["c"] for sp in ln["spans"] for ch in sp["chars"])
                r = pymupdf.Rect(ln["bbox"]) * M
                r.normalize()
                if r.width > 1 and r.height > 1 and txt.strip():
                    raw.append(([r.x0, r.y0, r.x1, r.y1], txt))
    else:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            pw, ph = float(page.width), float(page.height)
            raw = [([ln["x0"], ln["top"], ln["x1"], ln["bottom"]], ln["text"])
                   for ln in page.extract_text_lines()]
    sx, sy = img_w / pw, img_h / ph
    boxes, texts = [], []
    for b, t in raw:
        if not all(abs(v) < 1e5 for v in b):      # broken content-stream coordinates
            continue
        boxes.append([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy])
        texts.append(t)
    return (np.array(boxes, dtype=np.float32).reshape(-1, 4), texts,
            dict(page_pts=[pw, ph], scale=[sx, sy], n_lines=len(texts),
                 line_source=source))


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg_j = run.cfg
    thr = float(cfg_j.get("threshold", 0.3))
    blank = bool(cfg_j.get("blank_text", False))
    line_source = cfg_j.get("line_source", "pdfplumber")
    wpath = os.path.join(MODELS, cfg_j["weights"])

    t = Timer()
    with t.phase("model_load"):
        sys.path.insert(0, M2DOC_DIR)
        import torch
        from mmengine import Config
        from mmengine.dataset import Compose, pseudo_collate
        from mmdet.apis import init_detector

        cfg = Config.fromfile(os.path.join(M2DOC_DIR, cfg_j["config"]))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # mmengine 0.10.7 calls torch.load without `weights_only`, and torch >= 2.6
        # defaults it to True, which refuses this checkpoint: its training state
        # carries an mmengine HistoryBuffer and numpy arrays.  The file's sha256 is
        # recorded below and it is loaded from local disk only, so the restriction
        # is lifted for exactly this one call and restored immediately.
        _load = torch.load

        def _load_full(*a, **kw):
            kw["weights_only"] = False
            return _load(*a, **kw)

        torch.load = _load_full
        try:
            model = init_detector(cfg, wpath, device=device)
        finally:
            torch.load = _load
        labels = list(model.dataset_meta["classes"])
        pipeline = Compose(cfg.test_pipeline)
        cuda_sync()
    run.model_load_s = t.pop()["total_s"]

    with open(wpath, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    dev = ["text lines read from the PDF content stream with %s instead of an OCR "
           "engine; the corpus is born-digital, so this is the same information with "
           "no OCR error" % line_source,
           "line boxes scaled from PDF points into 300-dpi image pixels",
           "per-line `labels` (a training-time layout class) supplied as zeros"]
    if blank:
        dev.append("blank_text=True — line boxes kept, every string replaced by '', "
                   "ablating the text content while preserving text geometry")

    run.set_model_info(
        repo_id="johnning2333/M2Doc", weights=cfg_j["weights"], local_path=wpath,
        sha256=sha, config=cfg_j["config"], labels=labels,
        framework="vendored mmdetection 3.3.0 + mmcv 2.1.0 (built from source)",
        architecture=("DINO-4scale + M2Doc fusion (ResNet-50 early fusion + decoder late "
                      "fusion) over bert-base-multilingual-cased line embeddings"),
        training_set="DocLayNet", device=device, threshold=thr, deviations=dev,
        provenance=("Google Drive link from the repository README; no upstream checksum "
                    "is published, so the sha256 recorded here is of the file as downloaded"))

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                import cv2
                h, w = cv2.imread(page["input_path"]).shape[:2]
                tb, txts, ginfo = pdf_lines(page["page_pdf"], w, h, line_source)
                if blank:
                    txts = [""] * len(txts)
                data = pipeline({"img_path": page["input_path"], "img_id": 0,
                                 "instances": [], "text_bboxes": tb, "texts": txts,
                                 "text_labels": [[0]] * len(txts)})
                batch = pseudo_collate([data])
            with t.phase("inference"):
                with torch.no_grad():
                    out = model.test_step(batch)[0]
                cuda_sync()
            with t.phase("postprocess"):
                pi = out.pred_instances
                boxes = pi.bboxes.cpu().numpy()
                scores = pi.scores.cpu().numpy()
                classes = pi.labels.cpu().numpy()
                keep = scores >= thr
                boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
                items = [{"source_class": labels[int(classes[i])],
                          "bbox": [float(v) for v in boxes[i]],
                          "confidence": float(scores[i])} for i in range(len(scores))]
                regions = build_regions(run.taxonomy, items)
                raw = {"boxes": boxes.tolist(), "scores": scores.tolist(),
                       "classes": [int(c) for c in classes], "names": labels,
                       "lines": ginfo}
            run.emit(page, regions, t.pop(), raw=raw,
                     meta={"threshold": thr, "lines": ginfo, "blank_text": blank})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
