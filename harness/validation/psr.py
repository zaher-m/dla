#!/usr/bin/env python3
"""Build a PDF Structural Reference for a page without needing a workspace.

`validation.reference.page_reference` needs a page and a render size.  This
supplies the render-size convention the pipeline uses: pixel dimensions come
from the same `get_pixmap(dpi=...)` call that produces the image every model
sees, so that geometry from a model and geometry from the PDF land in the same
space.  `core.reference.build` is the corpus-wide equivalent, over a benchmark's
`selected_pages.json`; this one works on any page of any file.
"""
import fitz

from validation.reference import page_reference

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
