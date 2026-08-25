#!/usr/bin/env python3
"""Characterise every page of a PDF corpus, before any model sees it.

Geometry, text layers, script/RTL content, raster images, vector drawings and
heuristic layout signals (column count, ruling density, figure area).  Two
things depend on this: page selection, and the report's description of what the
corpus actually contains.  Source PDFs are opened read-only and never modified.

    python -m core.inventory                          # the configured corpus
    python -m core.inventory --corpus data/jobs/x/input --workspace data/jobs/x
"""
import argparse, json, os, re, sys, unicodedata
from collections import Counter
import fitz  # PyMuPDF

# Import the harness package regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths  # noqa: E402

ARABIC = re.compile(r'[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]')
LATIN = re.compile(r'[A-Za-z]')
DIGIT_AR = re.compile(r'[٠-٩۰-۹]')


def script_profile(text):
    ar = len(ARABIC.findall(text)); la = len(LATIN.findall(text))
    tot = ar + la
    return {"arabic_chars": ar, "latin_chars": la,
            "arabic_ratio": round(ar / tot, 4) if tot else 0.0,
            "arabic_indic_digits": len(DIGIT_AR.findall(text))}


def column_signal(blocks, page_w):
    """Heuristic: cluster text-block x-centres to estimate column count."""
    centres = sorted((b[0] + b[2]) / 2 for b in blocks if b[2] - b[0] > page_w * 0.05)
    if len(centres) < 4:
        return 1, 0.0
    # single-linkage clustering with gap = 12% of page width
    gap = page_w * 0.12
    clusters, cur = [], [centres[0]]
    for c in centres[1:]:
        if c - cur[-1] > gap:
            clusters.append(cur); cur = [c]
        else:
            cur.append(c)
    clusters.append(cur)
    clusters = [c for c in clusters if len(c) >= 3]
    if not clusters:
        return 1, 0.0
    widest = max(b[2] - b[0] for b in blocks)
    full_width_frac = sum(1 for b in blocks if (b[2] - b[0]) > page_w * 0.7) / max(len(blocks), 1)
    return len(clusters), round(full_width_frac, 3)


def table_signal(page):
    """Count ruling lines from vector drawings -> table/grid likelihood."""
    h = v = 0
    try:
        for d in page.get_drawings():
            for item in d["items"]:
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    if abs(p1.y - p2.y) < 1.5 and abs(p1.x - p2.x) > 20: h += 1
                    elif abs(p1.x - p2.x) < 1.5 and abs(p1.y - p2.y) > 20: v += 1
                elif item[0] == "re":
                    r = item[1]
                    if r.width > 20 and r.height < 2.5: h += 1
                    if r.height > 20 and r.width < 2.5: v += 1
    except Exception:
        pass
    return h, v


def analyse(corpus):
    docs, pages = [], []
    for fn in sorted(os.listdir(corpus)):
        p = os.path.join(corpus, fn)
        if not fn.lower().endswith(".pdf"):
            continue
        doc = fitz.open(p)
        meta = dict(doc.metadata or {})
        d_text = []
        d_rec = {"file": fn, "path": p, "size_bytes": os.path.getsize(p),
                 "pages": doc.page_count, "metadata": meta,
                 "is_tagged": None, "page_sizes": Counter()}
        for i, page in enumerate(doc):
            r = page.rect
            raw = page.get_text("rawdict")
            text = page.get_text("text")
            d_text.append(text)
            blocks = [b[:4] for b in page.get_text("blocks") if b[6] == 0]
            ncols, fullw = column_signal(blocks, r.width)
            hl, vl = table_signal(page)
            imgs = page.get_images(full=True)
            img_area = 0.0
            for im in imgs:
                try:
                    for rr in page.get_image_rects(im[0]):
                        img_area += rr.width * rr.height
                except Exception:
                    pass
            spans = [s for b in raw["blocks"] if b["type"] == 0
                     for l in b["lines"] for s in l["spans"]]
            sizes = Counter(round(s["size"], 1) for s in spans)
            fonts = Counter(s["font"] for s in spans)
            body = sizes.most_common(1)[0][0] if sizes else 0
            big = sum(n for sz, n in sizes.items() if sz > body * 1.25)
            sp = script_profile(text)
            # RTL dominant direction from span ordering
            pages.append({
                "doc": fn, "page": i + 1,
                "width_pt": round(r.width, 2), "height_pt": round(r.height, 2),
                "rotation": page.rotation,
                "has_text_layer": len(text.strip()) > 20,
                "char_count": len(text.strip()),
                "text_blocks": len(blocks),
                "est_columns": ncols,
                "full_width_block_frac": fullw,
                "hlines": hl, "vlines": vl,
                "table_likely": bool(hl >= 3 and vl >= 2),
                "n_images": len(imgs),
                "image_area_frac": round(img_area / (r.width * r.height), 4) if imgs else 0.0,
                "scanned_page": bool(len(text.strip()) < 50 and img_area > 0.5 * r.width * r.height),
                "n_drawings": len(page.get_drawings()),
                "body_font_pt": body,
                "large_text_spans": big,
                "n_fonts": len(fonts),
                "top_fonts": fonts.most_common(3),
                **sp,
            })
            d_rec["page_sizes"][f"{round(r.width)}x{round(r.height)}"] += 1
        full = "\n".join(d_text)
        d_rec["script"] = script_profile(full)
        d_rec["page_sizes"] = dict(d_rec["page_sizes"])
        d_rec["total_chars"] = len(full)
        docs.append(d_rec)
        doc.close()
    return docs, pages


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=None, help="directory of PDFs (default: paths.corpus)")
    ap.add_argument("--workspace", default=None, help="output workspace (default: paths.workspace)")
    a = ap.parse_args()
    corpus = paths.resolve(a.corpus) if a.corpus else paths.CORPUS
    ws = paths.ensure_workspace(paths.resolve(a.workspace) if a.workspace else None)
    out = os.path.join(ws, "inventory")

    if not os.path.isdir(corpus):
        raise SystemExit(f"corpus directory does not exist: {corpus}")
    docs, pages = analyse(corpus)
    if not docs:
        raise SystemExit(f"no PDF files found in {corpus}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "corpus_inventory.json"), "w", encoding="utf8") as f:
        json.dump({"corpus": corpus, "documents": docs, "pages": pages}, f,
                  ensure_ascii=False, indent=2)
    print(f"documents={len(docs)} pages={len(pages)}")
    for d in docs:
        print(f"  {d['file']}: {d['pages']}p arabic_ratio={d['script']['arabic_ratio']} "
              f"chars={d['total_chars']} sizes={d['page_sizes']}")


if __name__ == "__main__":
    main()
