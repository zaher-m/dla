#!/usr/bin/env python3
"""Raw per-page signals read straight from the PDF.

Everything here is a measurement, not a judgement: `router.py` turns these into
a routing decision and a PSR-trust verdict, and the deterministic checks consume
them further downstream.  Splitting the two means a threshold change never
requires re-reading the corpus.

The signals exist to answer one question -- *can the PDF be trusted as a
reference for this page* -- so they concentrate on the ways a content stream
lies: text that is present but never painted, text that is really another
model's OCR output sitting under a scan, fonts that give correct geometry and
useless characters, and glyphs drawn outside the visible area.
"""
from collections import Counter

import numpy as np
import fitz

GRID = 8          # pt stride for area unions; coarse on purpose, these are ratios

# Arabic, Arabic Supplement/Extended, Presentation Forms, Hebrew, Syriac, Thaana.
_RTL_RANGES = ((0x0590, 0x05FF), (0x0600, 0x06FF), (0x0700, 0x074F),
               (0x0750, 0x077F), (0x07C0, 0x08FF), (0xFB1D, 0xFDFF),
               (0xFE70, 0xFEFF))


# Codepoints that mean extraction failed rather than that the page says this.
_PUA = ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD))


def _is_rtl_cp(cp):
    return any(lo <= cp <= hi for lo, hi in _RTL_RANGES)


def _is_pua_cp(cp):
    return any(lo <= cp <= hi for lo, hi in _PUA)


def _area_frac(boxes, w, h):
    """Union area of `boxes` as a fraction of the page, via a coarse raster."""
    if not boxes or w <= 0 or h <= 0:
        return 0.0
    gw, gh = max(1, int(w // GRID)), max(1, int(h // GRID))
    m = np.zeros((gh, gw), dtype=bool)
    for b in boxes:
        x1 = max(0, int(b[0] // GRID)); y1 = max(0, int(b[1] // GRID))
        x2 = min(gw, int(np.ceil(b[2] / GRID))); y2 = min(gh, int(np.ceil(b[3] / GRID)))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
    return float(m.sum()) / (gw * gh)


def _font_signals(doc, page):
    """Type3 and missing-ToUnicode share, per font rather than per glyph.

    A single Type3 font used for one decorative capital should not condemn the
    page, so the caller weighs this against the character counts.
    """
    fonts = page.get_fonts(full=True)
    if not fonts:
        return {"n_fonts": 0, "type3_frac": 0.0, "no_tounicode_frac": 0.0}
    n3 = nt = 0
    for f in fonts:
        xref, ftype = f[0], f[2]
        if str(ftype).lower() == "type3":
            n3 += 1
        try:
            key = doc.xref_get_key(xref, "ToUnicode")[0]
            if not key or key == "null":
                nt += 1
        except Exception:
            nt += 1
    n = len(fonts)
    return {"n_fonts": n, "type3_frac": round(n3 / n, 4),
            "no_tounicode_frac": round(nt / n, 4)}


def _trace_signals(page):
    """Visibility and text direction, from the low-level text trace.

    `get_text()` cannot answer either question: it reports what the content
    stream *says*, including text painted in render mode 3 (invisible) or in the
    page's own background colour.  A detector sees none of that, so scoring a
    model against it manufactures misses that never existed.  `get_texttrace()`
    exposes the paint type, the opacity and the colour.

    It also exposes `bidi`, which on this corpus is 0 for every span of a page
    that is 99% Arabic -- the writer never set it.  Direction is therefore taken
    from codepoint ranges in `page_signals`, and the trace value is reported
    alongside only so the disagreement stays visible.
    """
    try:
        trace = page.get_texttrace()
    except Exception:
        return {"trace_ok": False}
    n = inv = rtl = white = 0
    for sp in trace:
        k = len(sp.get("chars") or ())
        if not k:
            continue
        n += k
        # type 3 is "ignore text" -- painted nowhere.  Zero opacity reaches the
        # same outcome by a different route.
        if sp.get("type") == 3 or (sp.get("opacity") if sp.get("opacity") is not None else 1.0) <= 0.01:
            inv += k
        elif sp.get("color") == 0xFFFFFF:
            white += k
        if (sp.get("bidi") or 0) % 2 == 1:
            rtl += k
    if not n:
        return {"trace_ok": True, "n_trace_chars": 0, "invisible_frac": 0.0,
                "white_frac": 0.0, "bidi_rtl_frac": 0.0}
    return {"trace_ok": True, "n_trace_chars": n,
            "invisible_frac": round(inv / n, 4),
            "white_frac": round(white / n, 4),
            "bidi_rtl_frac": round(rtl / n, 4)}


def page_signals(doc, pno):
    """Every raw measurement for one page.  Pure read, no decisions."""
    page = doc[pno]
    rect = page.rect
    W, H = float(rect.width), float(rect.height)
    page_area = max(W * H, 1e-6)

    raw = page.get_text("rawdict")
    lines, spans, cps = [], [], Counter()
    n_chars = 0
    for blk in raw["blocks"]:
        if blk["type"] != 0:
            continue
        for ln in blk["lines"]:
            b = ln["bbox"]
            if b[2] - b[0] > 1 and b[3] - b[1] > 1:
                lines.append(list(b))
            for sp in ln["spans"]:
                spans.append(sp)
                for ch in sp.get("chars", ()):
                    c = ch.get("c") or ""
                    if c and not c.isspace():
                        n_chars += 1
                        cps[ord(c[0])] += 1

    images = []
    for im in page.get_images(full=True):
        try:
            for r in page.get_image_rects(im[0]):
                images.append([r.x0, r.y0, r.x1, r.y1])
        except Exception:
            pass
    big_images = [b for b in images
                  if (b[2] - b[0]) * (b[3] - b[1]) > 0.30 * page_area]

    def _in_big_image(box):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in big_images)

    drawings = page.get_drawings()
    # Vector ink, so that a page whose text was converted to outlines can be
    # told apart from a page that simply has few words on it.
    draw_rects = [[d["rect"].x0, d["rect"].y0, d["rect"].x1, d["rect"].y1]
                  for d in drawings
                  if d["rect"].width > 1 and d["rect"].height > 1
                  and d["rect"].width < W * 0.98 and d["rect"].height < H * 0.98]

    sizes = [s["size"] for s in spans if s.get("size")]
    heights = [b[3] - b[1] for b in lines]
    per_span = [len(sp.get("chars", ())) for sp in spans]

    rtl_cp = sum(v for k, v in cps.items() if _is_rtl_cp(k))
    # Direct evidence that decoding failed, rather than the ToUnicode proxy:
    # a font can lack a ToUnicode map and still decode correctly through a
    # standard encoding, which is the common case here.  Private-use and
    # replacement codepoints mean the characters really are unusable.
    pua_cp = sum(v for k, v in cps.items() if _is_pua_cp(k))
    bad_cp = sum(v for k, v in cps.items() if k in (0xFFFD, 0) or (k < 0x20 and k != 0x09))

    # Content drawn outside the visible area is in the stream and not on the
    # page.  CropBox is what a renderer shows; MediaBox is only the paper.
    crop = page.cropbox
    outside = sum(1 for b in lines
                  if b[2] < crop.x0 or b[0] > crop.x1
                  or b[3] < crop.y0 or b[1] > crop.y1)

    out = {
        "page": pno + 1,
        "width_pt": round(W, 2), "height_pt": round(H, 2),
        "rotation": page.rotation,
        "landscape": W > H,
        "n_chars": n_chars,
        "n_lines": len(lines),
        "n_spans": len(spans),
        "n_images": len(images),
        "n_big_images": len(big_images),
        "n_drawings": len(drawings),
        "glyph_area_frac": round(_area_frac(lines, W, H), 4),
        "vector_area_frac": round(_area_frac(draw_rects, W, H), 4),
        "image_area_frac": round(_area_frac(images, W, H), 4),
        "big_image_area_frac": round(_area_frac(big_images, W, H), 4),
        "lines_in_big_image_frac": round(
            sum(1 for b in lines if _in_big_image(b)) / len(lines), 4) if lines else 0.0,
        "median_font_pt": round(float(np.median(sizes)), 2) if sizes else 0.0,
        "median_line_h_pt": round(float(np.median(heights)), 2) if heights else 0.0,
        "median_chars_per_span": int(np.median(per_span)) if per_span else 0,
        "rtl_cp_frac": round(rtl_cp / n_chars, 4) if n_chars else 0.0,
        "pua_cp_frac": round(pua_cp / n_chars, 4) if n_chars else 0.0,
        "undecoded_cp_frac": round(bad_cp / n_chars, 4) if n_chars else 0.0,
        "lines_outside_crop_frac": round(outside / len(lines), 4) if lines else 0.0,
    }
    out.update(_font_signals(doc, page))
    out.update(_trace_signals(page))
    return out


def doc_signals(path, max_pages=None):
    doc = fitz.open(path)
    try:
        n = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        return [page_signals(doc, i) for i in range(n)]
    finally:
        doc.close()
