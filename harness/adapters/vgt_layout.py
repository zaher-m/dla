#!/usr/bin/env python3
"""VGT (Vision Grid Transformer) adapter: text-grid + image, Cascade R-CNN head.

VGT needs a word grid alongside the page image. For born-digital PDFs that grid
is free: the content stream already gives each word and its box, so pdfplumber
supplies it with no OCR pass and no OCR error — the same extractor upstream's
create_grid_input.py uses.

Two config knobs isolate what the text is worth, on identical weights:

  use_unk_text=False   real token ids in the grid
  use_unk_text=True    every token replaced by [UNK], boxes unchanged

The grid embedding is a frozen layoutlm-base-uncased table (English WordPiece),
so non-Latin scripts decompose into single characters rather than producing
[UNK]; the adapter records unk_frac, tokens_per_word and
nonascii_char_token_frac so that is visible in the output rather than assumed.

Loading note: the released configs declare WORDGRID.VOCAB_SIZE 30552 while every
published checkpoint stores 30522. detectron2 logs a skipped tensor and leaves
the embedding at its random init, which silently yields zero detections. The
vocab size is taken from the checkpoint and the loaded table is asserted equal.
"""
import os, sys, pickle, hashlib
import numpy as np
# The harness package sits one level up; resolve it from this file so the
# adapter works whether it is launched by the runner or by hand.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adapter_base import AdapterRun, Timer, parse_job, build_regions, cuda_sync, cuda_reset
from core import paths

VGT_DIR = os.path.join(paths.repo_dir("ALM"), "DocumentUnderstanding/VGT/object_detection")
MODELS = paths.model_dir("vgt")
GRID_CACHE = os.path.join(os.environ.get("DLA_WORKSPACE", paths.WORKSPACE),
                          "working", "vgt_grid")

LABELS = {
    "publaynet": ["text", "title", "list", "table", "figure"],
    "doclaynet": ["Caption", "Footnote", "Formula", "List-item", "Page-footer",
                  "Page-header", "Picture", "Section-header", "Table", "Text", "Title"],
    # Names from the D4LA dataset definition (arXiv:2308.14978 App. B) and
    # DocLayout-YOLO's d4la.yaml, which agree.  VGT's own demo script prints
    # "Regionlist" at 12 and "Footnote" at 21; the dataset says "RegionList"
    # and "Footer" (which the paper defines as the document footnote).
    "d4la": ["DocTitle", "ParaTitle", "ParaText", "ListText", "RegionTitle", "Date",
             "LetterHead", "LetterDear", "LetterSign", "Question", "OtherText",
             "RegionKV", "RegionList", "Abstract", "Author", "TableName", "Table",
             "Figure", "FigureName", "Equation", "Reference", "Footer", "PageHeader",
             "PageFooter", "Number", "Catalog", "PageNumber"],
}


def build_grid(pdf_path, img_w, img_h, tokenizer, cgi):
    """Upstream grid dict, with word boxes scaled into image pixel space."""
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        pw, ph = float(page.width), float(page.height)
        words = page.extract_words()
    if not words:                                          # upstream np.concatenate([]) would raise
        grid = {"input_ids": np.zeros((0,), dtype=np.int64),
                "bbox_subword_list": np.zeros((0, 4)), "texts": [],
                "bbox_texts_list": np.zeros((0, 4))}
    else:
        grid = cgi.create_grid_dict(tokenizer, words)      # upstream, verbatim
    sx, sy = img_w / pw, img_h / ph
    for key in ("bbox_subword_list", "bbox_texts_list"):
        b = np.asarray(grid[key], dtype=np.float64)
        if b.size:
            b = b * np.array([sx, sy, sx, sy])             # XYWH_ABS
        grid[key] = b
    return grid, dict(page_pts=[pw, ph], scale=[sx, sy], n_words=len(words))


def main():
    job = parse_job()
    run = AdapterRun(job)
    cfg_j = run.cfg
    thr = float(cfg_j.get("threshold", 0.5))
    dataset = cfg_j["dataset"]
    labels = LABELS[dataset]
    use_unk = bool(cfg_j.get("use_unk_text", False))
    chan = cfg_j.get("channel_order", "RGB").upper()
    tok_name = cfg_j.get("tokenizer", "bert-base-uncased")
    wpath = os.path.join(MODELS, cfg_j["weights"])

    t = Timer()
    with t.phase("model_load"):
        sys.path.insert(0, VGT_DIR)
        import torch, cv2
        from transformers import AutoTokenizer
        from ditod import add_vit_config
        from ditod.VGTTrainer import DefaultPredictor
        from detectron2.config import get_cfg
        import create_grid_input as cgi

        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        unk_id = tokenizer.unk_token_id
        # ids of every single-character non-ASCII piece in the vocabulary --
        # the tokens an Arabic word actually decomposes into.
        global NONASCII_CHAR_IDS
        NONASCII_CHAR_IDS = np.array(sorted(
            i for tk, i in tokenizer.get_vocab().items()
            if len(tk.lstrip("#")) == 1 and not tk.lstrip("#").isascii()), dtype=np.int64)

        cfg = get_cfg()
        add_vit_config(cfg)
        cfg.merge_from_file(os.path.join(VGT_DIR, cfg_j["config"]))
        cfg.MODEL.WEIGHTS = wpath
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thr
        cfg.MODEL.WORDGRID.USE_UNK_TEXT = use_unk
        # The fine-tuned checkpoint carries its own trained grid-embedding table,
        # so the layoutlm bootstrap file is neither needed nor wanted here; it is
        # only used to *initialise* training.  Loading it would be overwritten a
        # moment later by DetectionCheckpointer anyway.
        cfg.MODEL.WORDGRID.USE_PRETRAIN_WEIGHT = False
        # Upstream's configs declare WORDGRID.VOCAB_SIZE 30552 but every released
        # checkpoint stores a (30522, 768) table.  detectron2 does not fail on the
        # mismatch -- it logs "Skip loading parameter ... incompatible shapes" and
        # leaves the grid embedding at its RANDOM init, which silently turns the
        # text stream into noise and drives detections to zero.  Take the size
        # from the checkpoint itself.
        ckpt = torch.load(wpath, map_location="cpu", weights_only=False)
        ckpt_emb = (ckpt.get("model", ckpt))["Wordgrid_embedding.embedding.weight"]
        cfg.MODEL.WORDGRID.VOCAB_SIZE = int(ckpt_emb.shape[0])
        cfg.INPUT.FORMAT = "RGB"
        predictor = DefaultPredictor(cfg)
        # Prove the trained table actually reached the model.
        emb = predictor.model.Wordgrid_embedding.embedding.weight.detach().cpu()
        emb_ok = bool(torch.allclose(emb, ckpt_emb.to(emb.dtype)))
        del ckpt, ckpt_emb
        if not emb_ok:
            raise RuntimeError("grid embedding did not load from the checkpoint")
        cuda_sync()
    run.model_load_s = t.pop()["total_s"]

    with open(wpath, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    dev = []
    if chan == "BGR":
        dev.append("channel_order=BGR reproduces upstream inference.py, which feeds "
                   "BGR to a model whose config declares INPUT.FORMAT=RGB")
    else:
        dev.append("images fed as RGB per Base-RCNN-FPN.yaml INPUT.FORMAT; upstream's "
                   "demo inference.py leaves the BGR->RGB swap commented out")
    dev.append("word grid built from the PDF text layer with pdfplumber (upstream's own "
               "create_grid_input.create_grid_dict); no OCR is run")
    dev.append("word boxes scaled from PDF points into 300-dpi image pixels; upstream's "
               "create_grid_input.py emits raw points, correct only at 72 dpi")
    dev.append("MODEL.WORDGRID.VOCAB_SIZE taken from the checkpoint (%d), not from the "
               "config's 30552; at 30552 detectron2 skips the grid-embedding tensor and "
               "runs the text stream on a random table, yielding zero detections"
               % cfg.MODEL.WORDGRID.VOCAB_SIZE)
    if use_unk:
        dev.append("MODEL.WORDGRID.USE_UNK_TEXT=True — word geometry kept, token identity "
                   "replaced by [UNK]; ablation of the text-content channel")

    run.set_model_info(
        repo_id="AlibabaResearch/AdvancedLiterateMachinery", weights=cfg_j["weights"],
        local_path=wpath, sha256=sha, config=cfg_j["config"], labels=labels,
        framework="detectron2 0.6 + ditod (VGT)",
        architecture="DiT-base visual stream + Grid Transformer (text) -> Cascade R-CNN",
        training_set=dataset, device=cfg.MODEL.DEVICE, threshold=thr,
        deviations=dev,
        notes=("grid_embedding_loaded=%s; tokenizer=%s; use_unk_text=%s; channel_order=%s"
               % (emb_ok, tok_name, use_unk, chan)))

    gdir = os.path.join(GRID_CACHE, run.run_id)
    os.makedirs(gdir, exist_ok=True)

    for page in job["pages"]:
        try:
            cuda_reset()
            with t.phase("preprocess"):
                img = cv2.imread(page["input_path"])
                if chan != "BGR":
                    img = img[:, :, ::-1].copy()
                h, w = img.shape[:2]
                gpath = os.path.join(gdir, page["page_id"] + ".pkl")
                grid, ginfo = build_grid(page["page_pdf"], w, h, tokenizer, cgi)
                ids = np.asarray(grid["input_ids"])
                ginfo["n_tokens"] = int(ids.size)
                ginfo["unk_frac"] = (float((ids == unk_id).mean()) if ids.size else 0.0)
                ginfo["tokens_per_word"] = (round(ids.size / ginfo["n_words"], 3)
                                            if ginfo["n_words"] else 0.0)
                ginfo["nonascii_char_token_frac"] = (
                    float(np.isin(ids, NONASCII_CHAR_IDS).mean()) if ids.size else 0.0)
                with open(gpath, "wb") as f:
                    pickle.dump(grid, f)
            with t.phase("inference"):
                out = predictor(img, gpath)["instances"].to("cpu")
                cuda_sync()
            with t.phase("postprocess"):
                boxes = out.pred_boxes.tensor.numpy()
                scores = out.scores.numpy()
                classes = out.pred_classes.numpy()
                items = [{"source_class": labels[int(classes[i])],
                          "bbox": [float(v) for v in boxes[i]],
                          "confidence": float(scores[i])} for i in range(len(scores))]
                regions = build_regions(run.taxonomy, items)
                raw = {"boxes": boxes.tolist(), "scores": scores.tolist(),
                       "classes": [int(c) for c in classes], "names": labels,
                       "grid": ginfo}
            run.emit(page, regions, t.pop(), raw=raw,
                     meta={"threshold": thr, "grid": ginfo,
                           "use_unk_text": use_unk, "channel_order": chan})
        except Exception as e:
            t.pop(); run.fail(page, e)
    run.finish()


if __name__ == "__main__":
    main()
