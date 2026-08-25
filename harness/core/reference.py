#!/usr/bin/env python3
"""PDF Structural Reference (PSR) — a deterministic, reproducible geometric
reference derived from the source PDFs' own content streams.

THIS IS NOT HUMAN GROUND TRUTH.  Every document in the corpus is born-digital
with an intact text layer, so the PDF itself states exactly where glyphs,
raster images and vector drawings sit on the page.  That yields an objective
geometric reference for *localisation* questions that needs no annotator:

  text_area     union of glyph bounding boxes (line granularity)
  image_rects   placement rectangles of embedded raster images
  vector_areas  clustered vector-drawing extents (charts, rules, table grids)
  ruling_lines  long horizontal/vertical strokes -> table grid evidence
  columns       inter-column gutters found from the glyph x-profile
  gutters       vertical whitespace corridors wider than a threshold

It cannot say whether a block is a "caption" or a "heading" — class quality is
assessed separately (consensus + expert review).  What it *can* measure
objectively, per system:

  text_recall     how much real text area a system's text-bearing regions cover
  text_precision  how much of those regions is actually on text
  spill           predicted text area lying on no glyph at all
  column_bleed    predicted text regions crossing a real inter-column gutter
  figure_iou      overlap of predicted figures with real image/vector areas
  fragmentation   predicted regions per real text line
"""
import json, os, sys
from collections import defaultdict
import numpy as np
import fitz

# Import the harness package regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths  # noqa: E402
ROOT = paths.ROOT
BENCH = paths.WORKSPACE
OUT_NAME = os.path.join("inventory", "pdf_structural_reference.json")


def cluster_boxes(boxes, gap=6.0):
    """Merge overlapping/near-touching boxes (single-linkage on inflation)."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            b = boxes.pop()
            merged = True
            while merged:
                merged = False
                for i, o in enumerate(boxes):
                    if (b[0] - gap < o[2] and o[0] - gap < b[2] and
                            b[1] - gap < o[3] and o[1] - gap < b[3]):
                        b = [min(b[0], o[0]), min(b[1], o[1]), max(b[2], o[2]), max(b[3], o[3])]
                        boxes.pop(i); merged = True; changed = True
                        break
            out.append(b)
        boxes = out
    return boxes


def find_gutters(line_boxes, W, H, min_w_frac=0.028, max_cover=0.06):
    """Vertical whitespace corridors that separate text columns.

    Full-width lines (running headers, page-spanning titles, footers) legitimately
    cross a gutter, so they are excluded from the profile; a corridor only has to
    be clear of *column* text.  A corridor counts when its per-x text coverage is
    under `max_cover` of the busiest column and it is wider than `min_w_frac`.
    """
    if len(line_boxes) < 8:
        return []
    col_lines = [b for b in line_boxes if (b[2] - b[0]) < W * 0.62]
    if len(col_lines) < 8:
        return []
    ys = [b[1] for b in col_lines] + [b[3] for b in col_lines]
    top, bot = min(ys), max(ys)
    band = max(bot - top, 1.0)
    if band < H * 0.25:
        return []
    cover = np.zeros(int(W) + 2, dtype=np.float32)
    for x1, y1, x2, y2 in col_lines:
        cover[int(max(0, x1)):int(min(W, x2)) + 1] += (y2 - y1)
    cover /= band
    peak = float(cover.max()) or 1.0
    clear = cover <= peak * max_cover
    gutters, run = [], None
    for x in range(len(clear)):
        if clear[x]:
            run = x if run is None else run
        else:
            if run is not None and x - run >= W * min_w_frac:
                gutters.append([float(run), float(top), float(x), float(bot)])
            run = None
    if run is not None and len(clear) - run >= W * min_w_frac:
        gutters.append([float(run), float(top), float(len(clear) - 1), float(bot)])
    return [g for g in gutters if g[0] > W * 0.08 and g[2] < W * 0.92]


def column_bands(line_boxes, W, H):
    """Text column bands from the horizontal ink profile.

    Robust where gutter-hunting is not: running headers/footers and page-wide
    titles are excluded, the profile is smoothed by the median line height, and
    any resulting band narrower than 12% of the page is discarded as noise.
    """
    col_lines = [b for b in line_boxes if (b[2] - b[0]) < W * 0.62]
    if len(col_lines) < 10:
        return []
    heights = sorted(b[3] - b[1] for b in col_lines)
    lh = heights[len(heights) // 2] or 10.0
    prof = np.zeros(int(W) + 2, dtype=np.float32)
    for x1, y1, x2, y2 in col_lines:
        prof[int(max(0, x1)):int(min(W, x2)) + 1] += 1.0
    thresh = max(2.0, prof.max() * 0.05)
    ink = prof > thresh
    # bridge gaps narrower than one line height (inter-word / kerning noise)
    bridge = int(max(4, lh * 0.9))
    idx = np.flatnonzero(ink)
    if idx.size == 0:
        return []
    bands, start, prev = [], idx[0], idx[0]
    for x in idx[1:]:
        if x - prev > bridge:
            bands.append([float(start), float(prev)]); start = x
        prev = x
    bands.append([float(start), float(prev)])
    bands = [b for b in bands if (b[1] - b[0]) > W * 0.12]
    return bands


def build(ws=None, corpus=None):
    ws = ws or BENCH
    corpus = corpus or paths.CORPUS
    out = os.path.join(ws, OUT_NAME)
    with open(os.path.join(ws, "inventory", "selected_pages.json"), encoding="utf8") as f:
        pages = json.load(f)
    docs = {}
    ref = {}
    for p in pages:
        d = docs.setdefault(p["doc"], fitz.open(os.path.join(corpus, p["doc"])))
        page = d[p["page"] - 1]
        rect = page.rect
        sx = p["px_width"] / rect.width
        sy = p["px_height"] / rect.height

        # PyMuPDF renders `page.get_pixmap()` in *display* space (the /Rotate is
        # applied), but `get_text`, `get_image_rects` and `get_drawings` all report
        # geometry in the page's *unrotated* space.  On the four landscape pages in
        # this corpus (/Rotate 90) that put the whole reference 90 degrees out of
        # register with the very images every system is scored against -- PSR text
        # lines ran to y=3135 on a page only 2480 px tall.  `page.rotation_matrix`
        # is the identity when /Rotate is 0, so this changes nothing elsewhere.
        M = page.rotation_matrix

        def R(b):                       # unrotated box -> display-space box
            r = fitz.Rect(b[0], b[1], b[2], b[3]) * M
            r.normalize()
            return [r.x0, r.y0, r.x1, r.y1]

        PW, PH = float(p["px_width"]), float(p["px_height"])

        def SC(b):                      # display-space box -> pixel space, clipped
            # Content streams routinely draw past the media box (a chart whose
            # plotting area runs off the left edge, a rule that overshoots).  Nothing
            # off the page is visible to a detector, so leaving it in the reference
            # only inflates reference areas and caps IoU below 1 for a perfect
            # prediction -- page_002 had a graphic area starting at x = -1120 px.
            return [min(max(b[0] * sx, 0.0), PW), min(max(b[1] * sy, 0.0), PH),
                    min(max(b[2] * sx, 0.0), PW), min(max(b[3] * sy, 0.0), PH)]

        def S(b):
            return SC(R(b))

        raw = page.get_text("rawdict")
        lines, spans = [], []
        for blk in raw["blocks"]:
            if blk["type"] != 0:
                continue
            for ln in blk["lines"]:
                lb = ln["bbox"]
                if lb[2] - lb[0] > 1 and lb[3] - lb[1] > 1:
                    lines.append(S(lb))
                for sp in ln["spans"]:
                    spans.append({"bbox": S(sp["bbox"]), "size": sp["size"],
                                  "font": sp["font"]})
        blocks = [S(b[:4]) for b in page.get_text("blocks") if b[6] == 0]

        images = []
        for im in page.get_images(full=True):
            try:
                for rr in page.get_image_rects(im[0]):
                    images.append(S([rr.x0, rr.y0, rr.x1, rr.y1]))
            except Exception:
                pass

        vec, hl, vl = [], [], []
        page_area_pt = rect.width * rect.height
        for dr in page.get_drawings():
            r = dr["rect"]
            if r.width < 2 and r.height < 2:
                continue
            # A drawing counts as *graphic content* only if it is filled or has
            # real internal structure.  A single unfilled rectangle is a frame or
            # a table border, and treating it as a figure would swallow the page.
            is_graphic = (dr.get("fill") is not None) or (len(dr["items"]) >= 4)
            if is_graphic and (r.width * r.height) < page_area_pt * 0.45:
                vec.append(S([r.x0, r.y0, r.x1, r.y1]))
            for it in dr["items"]:
                # classify horizontal vs vertical *after* rotation -- on a /Rotate 90
                # page a stroke that is horizontal in the content stream is drawn
                # vertically, and a table grid built the other way round is nonsense
                if it[0] == "l":
                    a, b = it[1] * M, it[2] * M
                    if abs(a.y - b.y) < 1.5 and abs(a.x - b.x) > 20:
                        hl.append(SC([min(a.x, b.x), a.y - 0.5, max(a.x, b.x), a.y + 0.5]))
                    elif abs(a.x - b.x) < 1.5 and abs(a.y - b.y) > 20:
                        vl.append(SC([a.x - 0.5, min(a.y, b.y), a.x + 0.5, max(a.y, b.y)]))
                elif it[0] == "re":
                    rr = fitz.Rect(it[1]) * M
                    rr.normalize()
                    if rr.width > 20 and rr.height < 2.5:
                        hl.append(SC([rr.x0, rr.y0, rr.x1, rr.y1]))
                    if rr.height > 20 and rr.width < 2.5:
                        vl.append(SC([rr.x0, rr.y0, rr.x1, rr.y1]))

        # graphic areas = raster images + clustered vector drawings big enough to matter
        pa = p["px_width"] * p["px_height"]
        big_vec = [v for v in vec if (v[2]-v[0]) * (v[3]-v[1]) > 0.004 * pa]
        graphics = cluster_boxes(images + big_vec, gap=10)
        graphics = [g for g in graphics
                    if 0.006 * pa < (g[2]-g[0]) * (g[3]-g[1]) < 0.62 * pa]

        # Table grid candidates: cluster the ruling strokes into whole grids.  The
        # linkage gap has to exceed a table row height at 300 dpi (~40-90 px) or a
        # single table shatters into one box per rule.
        grid = cluster_boxes(hl + vl, gap=max(40.0, p["px_height"] * 0.03))
        grid = [g for g in grid if (g[2]-g[0]) > p["px_width"] * 0.25
                and (g[3]-g[1]) > p["px_height"] * 0.04]

        # Attribute every glyph line to the structure that owns it.  Axis labels
        # inside a chart and cells inside a ruled table are *not* body text: a
        # system that wraps them in a `figure`/`table` region is right, not wrong,
        # so scoring them as missed body text would invert the result.
        def inside(box, holders, frac=0.6):
            ba = max((box[2]-box[0]) * (box[3]-box[1]), 1e-6)
            for h in holders:
                ix1, iy1 = max(box[0], h[0]), max(box[1], h[1])
                ix2, iy2 = min(box[2], h[2]), min(box[3], h[3])
                if ix2 > ix1 and iy2 > iy1 and ((ix2-ix1)*(iy2-iy1)) / ba >= frac:
                    return True
            return False

        graphic_text = [L for L in lines if inside(L, graphics)]
        table_text = [L for L in lines if inside(L, grid) and L not in graphic_text]
        body_text = [L for L in lines if L not in graphic_text and L not in table_text]

        gutters = find_gutters(body_text, p["px_width"], p["px_height"])
        bands = column_bands(body_text, p["px_width"], p["px_height"])

        sizes = [s["size"] for s in spans]
        body = float(np.median(sizes)) if sizes else 0.0

        ref[p["page_id"]] = {
            "page_id": p["page_id"], "doc": p["doc"], "page": p["page"],
            "width": p["px_width"], "height": p["px_height"],
            "text_lines": lines, "text_blocks": blocks,
            "body_text_lines": body_text, "graphic_text_lines": graphic_text,
            "table_text_lines": table_text,
            "image_rects": images, "graphic_areas": graphics,
            "ruling_h": len(hl), "ruling_v": len(vl), "grid_candidates": grid,
            "gutters": gutters, "column_bands": bands,
            "n_columns_est": max(len(bands), len(gutters) + 1),
            "body_font_px": round(body * sy, 2),
            "large_spans": [s["bbox"] for s in spans if s["size"] > body * 1.25],
            "stratum": p["stratum"],
        }
        print(f"  {p['page_id']}: lines={len(lines)}(body={len(body_text)},"
              f"gfx={len(graphic_text)},tbl={len(table_text)}) blocks={len(blocks)} "
              f"graphics={len(graphics)} grids={len(grid)} gutters={len(gutters)} "
              f"cols={len(bands)}")
    with open(out, "w", encoding="utf8") as f:
        json.dump(ref, f, indent=1)
    print("wrote", out)
    return ref


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--corpus", default=None)
    a = ap.parse_args()
    build(paths.resolve(a.workspace) if a.workspace else None,
          paths.resolve(a.corpus) if a.corpus else None)


if __name__ == "__main__":
    main()
