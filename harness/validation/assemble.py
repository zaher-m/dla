#!/usr/bin/env python3
"""Layout + PSR -> the reading stream a downstream consumer would receive.

For a born-digital page the characters are already in the file, so the layout
stage contributes exactly three things: which glyph lines group together, in
what order those groups are read, and which store each group goes to.  All three
are computable without running OCR, which is why this module exists -- it makes
the downstream effect of a layout measurable, exactly and repeatably, with no
model and no annotation.

The output is deliberately expressed in *line ids*, not text.  Comparing two
layouts then reduces to comparing two partitions and two orderings of the same
set, which needs no alignment step and stays meaningful on pages whose
characters did not decode (router check C0-06).
"""
import numpy as np

from validation.buckets import bucket as to_bucket, TEXT

MIN_COVER = 0.5          # a line belongs to the region covering at least this much of it
FULL_WIDTH = 0.62        # a region this wide spans columns rather than sitting in one


def _cover(line, box):
    ix1, iy1 = max(line[0], box[0]), max(line[1], box[1])
    ix2, iy2 = min(line[2], box[2]), min(line[3], box[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    a = max((line[2] - line[0]) * (line[3] - line[1]), 1e-6)
    return ((ix2 - ix1) * (iy2 - iy1)) / a


def assign(lines, regions):
    """Each line to the region covering most of it, or None."""
    out = []
    for L in lines:
        best, bi = 0.0, None
        for i, r in enumerate(regions):
            c = _cover(L, r["bbox"])
            if c > best:
                best, bi = c, i
        out.append(bi if best >= MIN_COVER else None)
    return out


def _band_of(box, bands, width):
    """Index of the column band a region sits in, or None when it spans them."""
    if not bands or (box[2] - box[0]) > width * FULL_WIDTH:
        return None
    hits = [i for i, b in enumerate(bands)
            if min(box[2], b[1]) - max(box[0], b[0]) > (box[2] - box[0]) * 0.5]
    return hits[0] if len(hits) == 1 else None


def derive_order(regions, psr, direction="ltr"):
    """Column-aware reading order when the model did not supply one.

    A simplified XY-cut: full-width regions partition the page, and the column
    regions between two partitions are read one band at a time.  Band order is
    reversed for right-to-left pages, which is the single decision that a
    per-page language guess gets wrong on numeric tables -- hence `direction`
    is supplied by the document, never inferred here.

    Reversing the bands is not on its own enough.  Two regions sharing a row
    inside one band, or falling outside every band, are ordered by the sort
    below, and sorting them by left edge reads an Arabic row backwards.  So the
    horizontal key mirrors for RTL, exactly as the line key in `assemble` does
    -- the two disagreeing was a real defect: on a corpus where no model emits
    a reading order, this function *is* the pipeline's reading order.
    """
    W = psr["width"]
    bands = psr.get("column_bands") or []
    rtl = direction == "rtl"
    # Rows are quantised before the horizontal key applies.  Two cells of one
    # table row differ by a point or two at the top edge, and on exact y they
    # sort by that jitter rather than by position, which discards the direction
    # key entirely on the pages that need it most.
    hs = [L[3] - L[1] for L in psr.get("text_lines") or []]
    q = max(float(np.median(hs)) * 0.6, 1.0) if hs else 6.0
    order = sorted(range(len(regions)),
                   key=lambda i: (round(regions[i]["bbox"][1] / q),
                                  -regions[i]["bbox"][2] if rtl
                                  else regions[i]["bbox"][0]))
    band_seq = list(range(len(bands)))
    if rtl:
        band_seq.reverse()

    out, pending = [], {b: [] for b in band_seq}
    def flush():
        for b in band_seq:
            out.extend(pending[b])
            pending[b] = []

    for i in order:
        b = _band_of(regions[i]["bbox"], bands, W)
        if b is None:
            flush()
            out.append(i)
        else:
            pending.setdefault(b, []).append(i)
    flush()
    return out


def assemble(regions, psr, direction="ltr", lines=None):
    """-> {"blocks": [...], "orphans": [...], "line_block": [...], "order": [...]}"""
    lines = psr["text_lines"] if lines is None else lines
    owner = assign(lines, regions)

    if regions and all(r.get("reading_order") is not None for r in regions):
        order = sorted(range(len(regions)), key=lambda i: regions[i]["reading_order"])
        order_source = "model"
    else:
        order = derive_order(regions, psr, direction)
        order_source = "derived"

    rtl = direction == "rtl"
    blocks, seq, line_block = [], [], [None] * len(lines)
    for rank, ri in enumerate(order):
        mine = [i for i, o in enumerate(owner) if o == ri]
        # within a block, top to bottom then along the writing direction
        mine.sort(key=lambda i: (round(lines[i][1], 1),
                                 -lines[i][0] if rtl else lines[i][0]))
        for i in mine:
            line_block[i] = rank
        seq.extend(mine)
        r = regions[ri]
        blocks.append({"region": ri, "rank": rank,
                       "class": r.get("class"), "bucket": to_bucket(r.get("class")),
                       "source": r.get("source"), "bbox": r["bbox"], "lines": mine})
    orphans = [i for i, o in enumerate(owner) if o is None]
    return {"blocks": blocks, "orphans": orphans, "line_block": line_block,
            "sequence": seq, "order_source": order_source, "n_lines": len(lines)}
