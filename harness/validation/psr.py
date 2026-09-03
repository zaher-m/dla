#!/usr/bin/env python3
"""Build a PDF Structural Reference for a page without needing a workspace.

`core.reference.build` reads a benchmark's `selected_pages.json` and writes a
corpus-wide file.  Validation needs the same reference for one arbitrary page of
one uploaded document, so this wraps `core.reference.page_reference` with the
render-size convention the rest of the pipeline uses: pixel dimensions come from
the same `get_pixmap(dpi=...)` call that produces the image every model sees, so
that geometry from a model and geometry from the PDF land in the same space.
"""
import os, sys

import fitz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.reference import page_reference  # noqa: E402

DPI = 300


def render_size(page, dpi=DPI):
    """Pixel size of `page` at `dpi`, matching `select_pages.render`.

    Taken from the pixmap rather than computed from the rect: PyMuPDF rounds,
    and a one-pixel disagreement between the reference and the image every model
    was given is a silent source of edge-of-page mismatches.
    """
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    return pix.width, pix.height


def page_psr(doc, pno, dpi=DPI, size=None):
    w, h = size or render_size(doc[pno], dpi)
    r = page_reference(doc[pno], w, h)
    r["page"] = pno + 1
    return r


def doc_psr(path, pages=None, dpi=DPI):
    doc = fitz.open(path)
    try:
        idx = pages if pages is not None else range(doc.page_count)
        return {i + 1: page_psr(doc, i, dpi) for i in idx}
    finally:
        doc.close()
