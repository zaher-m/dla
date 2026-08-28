#!/usr/bin/env python3
"""Objective layout metrics against the PDF Structural Reference (PSR).

No human ground truth exists for this corpus, so nothing here claims to be
mAP against annotated labels.  What is computed is a set of *geometric*
measures that the source PDFs answer unambiguously, plus cross-system
consensus.  Every metric states exactly what it measures.

Per system, per page:

  text_recall        area of real glyph coverage captured by text-bearing regions
  text_precision     share of predicted text-region area that sits on real glyphs
  text_spill         share of predicted text-region area on no glyph and no graphic
  line_capture       share of real text lines fully inside some text-bearing region
  line_split         real text lines cut by a predicted region boundary
  fragmentation      predicted text regions per real text line
  graphic_recall/iou localisation of real images + vector graphics as `figure`
  table_recall       localisation of real ruled grids as `table`
  column_bleed       text regions spanning two real column bands
  overlap_ratio      share of predicted area covered by >1 region (double-counting)
  page_coverage      share of page area covered by any predicted region
  class_diversity    number of distinct canonical classes predicted
"""
import json, os, sys
from collections import defaultdict, Counter
import numpy as np

# Import the harness package regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths  # noqa: E402
ROOT = paths.ROOT
BENCH = paths.WORKSPACE
NORM = os.path.join(BENCH, "normalized_outputs")
REF = os.path.join(BENCH, "inventory", "pdf_structural_reference.json")

TEXT_CLASSES = {"text", "title", "heading", "list", "caption", "footnote",
                "header", "footer", "sidebar", "page_number", "formula"}
FIGURE_CLASSES = {"figure"}
TABLE_CLASSES = {"table"}

GRID = 4          # rasterisation stride in px; 4 keeps 2481x3508 pages cheap and exact enough


def _mask(shape, boxes, polygons=None):
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    for b in boxes:
        x1 = max(0, int(b[0] // GRID)); y1 = max(0, int(b[1] // GRID))
        x2 = min(w, int(np.ceil(b[2] / GRID))); y2 = min(h, int(np.ceil(b[3] / GRID)))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
    return m


def _count_mask(shape, boxes):
    h, w = shape
    m = np.zeros((h, w), dtype=np.int16)
    for b in boxes:
        x1 = max(0, int(b[0] // GRID)); y1 = max(0, int(b[1] // GRID))
        x2 = min(w, int(np.ceil(b[2] / GRID))); y2 = min(h, int(np.ceil(b[3] / GRID)))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] += 1
    return m


def iou_box(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cover_frac(inner, boxes):
    """Fraction of `inner`'s area covered by the union of `boxes`.

    Area-based rather than strict containment: PDF line boxes include
    ascender/descender and Arabic diacritic extents that routinely poke a few
    pixels outside an otherwise perfect block prediction, and strict
    containment would score that as a miss.
    """
    ia = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-6)
    hits = []
    for b in boxes:
        ix1, iy1 = max(b[0], inner[0]), max(b[1], inner[1])
        ix2, iy2 = min(b[2], inner[2]), min(b[3], inner[3])
        if ix2 > ix1 and iy2 > iy1:
            hits.append((ix1, iy1, ix2, iy2))
    if not hits:
        return 0.0
    if len(hits) == 1:
        h = hits[0]
        return ((h[2]-h[0]) * (h[3]-h[1])) / ia
    # union via a small local raster (lines are tiny, so this is cheap)
    x0, y0 = inner[0], inner[1]
    w = max(1, int(np.ceil(inner[2] - x0))); h_ = max(1, int(np.ceil(inner[3] - y0)))
    if w * h_ > 400000:
        return min(1.0, sum((a[2]-a[0]) * (a[3]-a[1]) for a in hits) / ia)
    m = np.zeros((h_, w), dtype=bool)
    for a in hits:
        m[int(a[1]-y0):int(np.ceil(a[3]-y0)), int(a[0]-x0):int(np.ceil(a[2]-x0))] = True
    return float(m.sum()) / (w * h_)


def crosses(box, inner):
    """True when `box` cuts `inner` (partial overlap, not containment)."""
    ov = iou_box(box, inner)
    if ov == 0:
        return False
    ia = max((inner[2]-inner[0]) * (inner[3]-inner[1]), 1e-6)
    ix1, iy1 = max(box[0], inner[0]), max(box[1], inner[1])
    ix2, iy2 = min(box[2], inner[2]), min(box[3], inner[3])
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    frac = inter / ia
    return 0.12 < frac < 0.88


def page_metrics(pred, ref):
    W, H = ref["width"], ref["height"]
    shape = (int(np.ceil(H / GRID)), int(np.ceil(W / GRID)))
    page_cells = shape[0] * shape[1]

    regions = pred["regions"]
    txt_boxes = [r["bbox"] for r in regions if r["class"] in TEXT_CLASSES]
    fig_boxes = [r["bbox"] for r in regions if r["class"] in FIGURE_CLASSES]
    tab_boxes = [r["bbox"] for r in regions if r["class"] in TABLE_CLASSES]
    all_boxes = [r["bbox"] for r in regions]

    body_lines = ref.get("body_text_lines", ref["text_lines"])
    gfx_lines = ref.get("graphic_text_lines", [])
    tbl_lines = ref.get("table_text_lines", [])
    ref_text = _mask(shape, body_lines)
    ref_graph = _mask(shape, ref["graphic_areas"])
    ref_grid = _mask(shape, ref["grid_candidates"])

    pred_txt = _mask(shape, txt_boxes)
    pred_fig = _mask(shape, fig_boxes)
    pred_tab = _mask(shape, tab_boxes)
    pred_all = _mask(shape, all_boxes)
    # Unruled tables are common in this corpus; a system that wraps body lines in
    # a `table` region has structured them, not lost them, so report both a
    # strict text recall and a text-or-table recall.
    pred_txt_tab = _mask(shape, txt_boxes + tab_boxes)

    rt = ref_text.sum()
    pt = pred_txt.sum()
    inter_t = np.logical_and(ref_text, pred_txt).sum()
    inter_tt = np.logical_and(ref_text, pred_txt_tab).sum()
    # text region area that is neither glyphs nor a legitimate graphic
    spill = np.logical_and(pred_txt, np.logical_not(np.logical_or(ref_text, ref_graph))).sum()

    lines = body_lines
    cf = [cover_frac(L, txt_boxes) for L in lines]
    # completeness of the page decomposition, independent of class
    any_cf = [cover_frac(L, all_boxes) for L in ref["text_lines"]]
    gfx_cf = [cover_frac(L, fig_boxes) for L in gfx_lines]
    tbl_cf = [cover_frac(L, tab_boxes) for L in tbl_lines]
    captured = sum(1 for c in cf if c >= 0.9)
    partial = sum(1 for c in cf if 0.05 < c < 0.9)
    split = sum(1 for L in lines if any(crosses(b, L) for b in txt_boxes))

    rg = ref_graph.sum()
    fig_inter = np.logical_and(ref_graph, pred_fig).sum()
    fig_union = np.logical_or(ref_graph, pred_fig).sum()

    rgr = ref_grid.sum()
    tab_inter = np.logical_and(ref_grid, pred_tab).sum()
    tab_union = np.logical_or(ref_grid, pred_tab).sum()

    bands = ref.get("column_bands") or []
    bleed = gutter_cross = 0
    if len(bands) >= 2:
        for b in txt_boxes:
            hit = sum(1 for bd in bands
                      if min(b[2], bd[1]) - max(b[0], bd[0]) > (bd[1] - bd[0]) * 0.2)
            if hit >= 2:
                bleed += 1
        # softer signal: a text region that spans most of the whitespace corridor
        # between two columns has merged across the gutter even if it only clips
        # the far column
        gaps = [(bands[i][1], bands[i + 1][0]) for i in range(len(bands) - 1)
                if bands[i + 1][0] > bands[i][1]]
        for b in txt_boxes:
            for g0, g1 in gaps:
                w = g1 - g0
                if w > 0 and (min(b[2], g1) - max(b[0], g0)) > w * 0.6:
                    gutter_cross += 1
                    break

    cnt = _count_mask(shape, all_boxes)
    covered = (cnt > 0).sum()
    overlapped = (cnt > 1).sum()

    return {
        "n_regions": len(regions),
        "text_recall": round(float(inter_t / rt), 4) if rt else None,
        "text_or_table_recall": round(float(inter_tt / rt), 4) if rt else None,
        "text_precision": round(float(inter_t / pt), 4) if pt else None,
        "text_spill": round(float(spill / pt), 4) if pt else None,
        "line_capture": round(captured / len(lines), 4) if lines else None,
        "line_partial_rate": round(partial / len(lines), 4) if lines else None,
        "line_cover_mean": round(float(np.mean(cf)), 4) if lines else None,
        "any_region_capture": round(float(np.mean([c >= 0.9 for c in any_cf])), 4) if any_cf else None,
        "graphic_text_capture": round(float(np.mean([c >= 0.9 for c in gfx_cf])), 4) if gfx_cf else None,
        "table_text_capture": round(float(np.mean([c >= 0.9 for c in tbl_cf])), 4) if tbl_cf else None,
        "line_split_rate": round(split / len(lines), 4) if lines else None,
        "regions_per_line": round(len(txt_boxes) / len(lines), 4) if lines else None,
        "graphic_recall": round(float(fig_inter / rg), 4) if rg else None,
        "graphic_iou": round(float(fig_inter / fig_union), 4) if fig_union else None,
        "table_recall": round(float(tab_inter / rgr), 4) if rgr else None,
        "table_iou": round(float(tab_inter / tab_union), 4) if tab_union else None,
        "n_ref_columns": len(bands),
        "column_bleed": bleed,
        "gutter_cross": gutter_cross,
        "gutter_cross_rate": round(gutter_cross / len(txt_boxes), 4) if txt_boxes else None,
        "column_bleed_rate": round(bleed / len(txt_boxes), 4) if txt_boxes else None,
        "page_coverage": round(float(covered / page_cells), 4),
        "overlap_ratio": round(float(overlapped / covered), 4) if covered else 0.0,
        "class_diversity": len({r["class"] for r in regions}),
        "has_polygons": any(_nonrect_polygon(r) for r in regions),
        "has_polygon_field": any(r.get("polygon") for r in regions),
        "has_reading_order": any(r.get("reading_order") is not None for r in regions),
        "inference_s": pred["timing"].get("inference"),
        "total_s": pred["timing"].get("total_s"),
    }



def _nonrect_polygon(r, tol=1.0):
    """True only for a genuinely non-rectangular outline.

    Several detectors fill `polygon` with the four corners of their own bounding
    box (surya's `_poly`, Docling's converters).  Counting that as segmentation
    would credit a box detector with mask output, which is exactly the claim the
    "best segmentation" ranking turns on.  A polygon counts only if it has more
    than four vertices, or if its four vertices are not the axis-aligned corners
    of its own bbox within `tol` pixels.
    """
    poly = r.get("polygon")
    if not poly or len(poly) < 3:
        return False
    if len(poly) > 4:
        return True
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    corners = {(round(x), round(y)) for x, y in zip(xs, ys)}
    box = {(round(min(xs)), round(min(ys))), (round(max(xs)), round(min(ys))),
           (round(max(xs)), round(max(ys))), (round(min(xs)), round(max(ys)))}
    if len(corners) != 4:
        return True
    return any(min(abs(cx - bx) + abs(cy - by) for bx, by in box) > tol
               for cx, cy in corners)


def main():
    ref = json.load(open(REF))
    out = {}
    for rid in sorted(os.listdir(NORM)):
        man = os.path.join(NORM, rid, "_run.json")
        if not os.path.exists(man):
            continue
        m = json.load(open(man))
        if m.get("status") != "ok":
            out[rid] = {"status": m.get("status"), "note": m.get("note", "")}
            continue
        per = {}
        for f in sorted(os.listdir(os.path.join(NORM, rid))):
            if f.startswith("_"):
                continue
            pid = f[:-5]
            if pid not in ref:
                continue
            pred = json.load(open(os.path.join(NORM, rid, f)))
            per[pid] = page_metrics(pred, ref[pid])
        agg = {}
        keys = list(next(iter(per.values())).keys()) if per else []
        for k in keys:
            vals = [float(v[k]) for v in per.values()
                    if isinstance(v.get(k), (int, float)) and not isinstance(v.get(k), bool)]
            if vals:
                agg[k] = {"mean": round(float(np.mean(vals)), 4),
                          "median": round(float(np.median(vals)), 4),
                          "p10": round(float(np.percentile(vals, 10)), 4),
                          "p90": round(float(np.percentile(vals, 90)), 4)}
        out[rid] = {"status": "ok", "pages": per, "aggregate": agg,
                    "model": m.get("model", {}), "model_load_s": m.get("model_load_s"),
                    "resources": m.get("resources", {}), "torch_env": m.get("torch_env", {}),
                    "config": m.get("config", {})}
        a = agg
        print(f"{rid:34s} txtR={a.get('text_recall',{}).get('median')} "
              f"txtTblR={a.get('text_or_table_recall',{}).get('median')} "
              f"txtP={a.get('text_precision',{}).get('median')} "
              f"spill={a.get('text_spill',{}).get('median')} "
              f"lineCap={a.get('line_capture',{}).get('median')} "
              f"lineCov={a.get('line_cover_mean',{}).get('median')} "
              f"anyCap={a.get('any_region_capture',{}).get('median')} "
              f"figIoU={a.get('graphic_iou',{}).get('median')} "
              f"gutterX={a.get('gutter_cross_rate',{}).get('mean')}")
    os.makedirs(os.path.join(BENCH, "metrics"), exist_ok=True)
    json.dump(out, open(os.path.join(BENCH, "metrics", "layout_metrics.json"), "w"), indent=1)
    print("\nwrote metrics/layout_metrics.json")


if __name__ == "__main__":
    main()
