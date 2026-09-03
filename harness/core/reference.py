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

import fitz

# Import the harness packages regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths  # noqa: E402
# The reference itself lives in `validation`, which depends on nothing in this
# repository.  The dependency runs this way round on purpose: the PSR is a
# validation concept that the benchmark reuses, not the other way about, and
# keeping it here would tie a library other software builds on to a harness it
# has no use for.  Re-exported below so `core.reference.page_reference` keeps
# working for anything that already imports it.
from validation.reference import (  # noqa: E402,F401
    cluster_boxes, column_bands, find_gutters, page_reference)
ROOT = paths.ROOT
BENCH = paths.WORKSPACE
OUT_NAME = os.path.join("inventory", "pdf_structural_reference.json")

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
        r = page_reference(page, p["px_width"], p["px_height"])
        r.update({"page_id": p["page_id"], "doc": p["doc"], "page": p["page"],
                  "stratum": p["stratum"]})
        ref[p["page_id"]] = r
        lines, body_text = r["text_lines"], r["body_text_lines"]
        graphic_text, table_text = r["graphic_text_lines"], r["table_text_lines"]
        blocks, graphics = r["text_blocks"], r["graphic_areas"]
        grid, gutters, bands = r["grid_candidates"], r["gutters"], r["column_bands"]
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
