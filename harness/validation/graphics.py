#!/usr/bin/env python3
"""Is this filled cluster a figure, or the document's own content?

`reference.py` collects filled vector drawings into graphic areas and moves the
glyph lines inside them out of `body_text_lines`, on the reasoning that a
chart's axis labels are the figure's, not the page's.  That is right for a
chart.  It is wrong for a table drawn with banded cell fills, for a framed
paragraph, for a heading in a coloured box and for a footnote on a tint -- and
those are 56% of the clusters on this corpus, holding 8% of all its glyph lines
on 17% of its pages.  Everything they hold is invisible to the coverage family,
which is a blind spot on checks that block.

Text density was tried and does not separate them (E4), and neither does the
table score: a third of the misattributed clusters are prose, which scores as
"not a table" exactly like a chart does.  The question is not table-vs-chart.

What does separate them is what the cluster is *drawn from*, which the content
stream states and `reference.graphic_shape` now records: a chart is built from
curves and overlapping fills and labelled in type smaller than the body, while a
shaded table or a tinted paragraph is built from plain rectangles and holds text
at body size laid out in rows.  Eight features, logistic regression, weights in
`config/graphic_model.json`.  Inference is a dot product.
"""
import json
import os

import numpy as np

NAMES = ("rect_frac", "curve_frac", "fill_cover", "n_fill_col",
         "small_text", "text_ink", "left_align", "span_frac")

_MODEL = None


def _a(b):
    return max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)


def _inter(p, q):
    return (max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
            * max(0.0, min(p[3], q[3]) - max(p[1], q[1])))


def held_lines(cluster, psr, cover=0.6):
    """The glyph lines the cluster owns, on the same test `reference` uses."""
    return [L for L in psr["text_lines"] if _inter(L, cluster) >= cover * _a(L)]


def features(i, psr):
    """-> a vector aligned to NAMES for graphic area `i`, or None.

    None means the question cannot be put: a reference built before
    `graphic_shape` existed, or a cluster holding too little text to describe.
    """
    shapes = psr.get("graphic_shape") or []
    areas = psr.get("graphic_areas") or []
    if i >= len(shapes) or i >= len(areas):
        return None
    g, sh = areas[i], shapes[i]
    held = held_lines(g, psr)
    if len(held) < 4:
        return None
    ga = _a(g)
    allh = [L[3] - L[1] for L in psr["text_lines"]]
    med = float(np.median(allh)) if allh else 1.0
    med = med or 1.0
    small = sum(1 for L in held if (L[3] - L[1]) < 0.7 * med) / len(held)
    ink = min(sum(_a(L) for L in held) / ga, 1.0)

    # How much of the text stacks at one left edge.  A chart's tick labels sit
    # in a column at a single x; a table's rows start at one x per column and a
    # paragraph's at one, which is why this reads the other way round from the
    # `left_edges` count in `tablefeat`.
    tol = max(med * 0.5, 1.0)
    v = sorted(L[0] for L in held)
    groups, cur = [], [v[0]]
    for x in v[1:]:
        if x - cur[-1] <= tol:
            cur.append(x)
        else:
            groups.append(cur)
            cur = [x]
    groups.append(cur)
    left = max(len(x) for x in groups) / len(held)
    span = sum(1 for L in held
               if (L[2] - L[0]) > 0.5 * (g[2] - g[0])) / len(held)

    return np.array([sh.get("rect", 0.0), sh.get("curve", 0.0),
                     min(sh.get("fill_area", 0.0), 2.0),
                     min(sh.get("fills", 0) / 8.0, 1.0),
                     small, ink, left, span], dtype=float)


def load_model(path=None):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    path = path or os.environ.get("DLA_GRAPHIC_MODEL")
    if not path:
        here = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        path = os.path.join(here, "config", "graphic_model.json")
    try:
        with open(path, encoding="utf8") as f:
            m = json.load(f)
    except FileNotFoundError:
        return None
    if list(m.get("features") or ()) != list(NAMES):
        raise ValueError("graphic_model.json was trained on different features")
    _MODEL = {k: np.array(m[k], dtype=float) if k in ("mu", "sd", "w") else m[k]
              for k in m}
    return _MODEL


def score(i, psr, model=None):
    """P(graphic area `i` holds document content), or None if unanswerable."""
    m = model or load_model()
    v = features(i, psr)
    if m is None or v is None:
        return None
    z = (v - m["mu"]) / m["sd"]
    return float(1.0 / (1.0 + np.exp(-(z @ m["w"] + m["b"]))))


def split(psr, floor=0.8):
    """-> (figure indices, content indices, unanswerable indices).

    A cluster nothing can be said about stays a figure, which is the behaviour
    that existed before this module: the repair only ever moves a cluster the
    model positively identifies as content.
    """
    fig, content, unknown = [], [], []
    m = load_model()
    for i in range(len(psr.get("graphic_areas") or [])):
        s = score(i, psr, m)
        if s is None:
            unknown.append(i)
        elif s >= floor:
            content.append(i)
        else:
            fig.append(i)
    return fig, content, unknown


def repaired_body(psr, floor=0.8):
    """`body_text_lines` with the content clusters' lines put back.

    Returned in `text_lines` order so the result is stable and a consumer can
    still match a line to its index by coordinate.
    """
    body = psr.get("body_text_lines") or []
    _, content, _ = split(psr, floor)
    if not content:
        return list(body)
    areas = psr.get("graphic_areas") or []
    have = {tuple(round(v, 2) for v in L) for L in body}
    out = list(body)
    for i in content:
        for L in held_lines(areas[i], psr):
            k = tuple(round(v, 2) for v in L)
            if k not in have:
                have.add(k)
                out.append(L)
    order = {tuple(round(v, 2) for v in L): j
             for j, L in enumerate(psr["text_lines"])}
    out.sort(key=lambda L: order.get(tuple(round(v, 2) for v in L), 0))
    return out
