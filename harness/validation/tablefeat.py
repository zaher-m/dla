#!/usr/bin/env python3
"""Features for "is this region actually a table".

Two documented dead ends motivate this. Text density does not separate a shaded
table from a chart (p50 0.27 against 0.28), and ruling lines do not separate a
page column from a table column (1 of 13). Both failures are of a single
threshold on a single quantity, and both questions are instant for a human eye,
which is the signature of a problem with a decision boundary that no one
coordinate has.

The features are computed from the PDF's own geometry, never from pixels. That
is a deliberate constraint, not a shortcut: the validation package ships as a
314 MB image with four dependencies, and a convolutional model would make it six
gigabytes and require a GPU. Inference here is a dot product in numpy.

The load-bearing feature is factorisation. A table is a grid, so its lines
should resolve into r row bands and c column bands with r*c near the line count.
Prose does not factorise: its lines make one column and as many rows as there
are lines. Density never saw this because it collapses layout to one number.
"""
import numpy as np

NAMES = ("rules_h", "rules_v", "numeric", "short_tokens", "col_bands",
         "row_bands", "factorises", "left_edges", "width_var", "height_var",
         "fill", "line_density", "aspect", "area_frac")


def _clusters(vals, tol):
    """How many distinct positions, merging anything within tol."""
    if not vals:
        return 0
    v = sorted(vals)
    n, last = 1, v[0]
    for x in v[1:]:
        if x - last > tol:
            n += 1
            last = x
    return n


def _inside(box, region, cover=0.6):
    ix1, iy1 = max(box[0], region[0]), max(box[1], region[1])
    ix2, iy2 = min(box[2], region[2]), min(box[3], region[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    a = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    return ((ix2 - ix1) * (iy2 - iy1)) / a >= cover


def features(region, psr, texts=None):
    """-> a vector aligned to NAMES, or None when the region holds too little."""
    x0, y0, x1, y1 = region
    w, h = x1 - x0, y1 - y0
    if w <= 1 or h <= 1:
        return None
    lines = [L for L in psr["text_lines"] if _inside(L, region)]
    if len(lines) < 4:
        return None
    hs = [L[3] - L[1] for L in lines]
    med_h = float(np.median(hs)) or 1.0

    rh = sum(1 for r in (psr.get("rules_h") or [])
             if r[1] >= y0 and r[1] <= y1 and min(r[2], x1) - max(r[0], x0) > 0.5 * w)
    rv = sum(1 for r in (psr.get("rules_v") or [])
             if r[0] >= x0 and r[0] <= x1 and min(r[3], y1) - max(r[1], y0) > 0.5 * h)

    txt = ""
    if texts:
        idx = {tuple(round(v, 2) for v in L): i
               for i, L in enumerate(psr["text_lines"])}
        for L in lines:
            i = idx.get(tuple(round(v, 2) for v in L))
            if i is not None and i < len(texts):
                txt += " " + texts[i]
    toks = txt.split()
    numeric = (sum(1 for t in toks if any(c.isdigit() for c in t)) / len(toks)
               if toks else 0.0)
    short = sum(1 for t in toks if len(t) <= 4) / len(toks) if toks else 0.0

    # rows and columns, from where the lines actually start and sit
    cols = _clusters([L[0] for L in lines], max(w * 0.04, med_h))
    rows = _clusters([L[1] for L in lines], med_h * 0.6)
    # a grid's line count is about rows x columns; prose is rows x 1
    fact = min(rows * max(cols, 1) / max(len(lines), 1), 4.0)
    lefts = _clusters([L[0] for L in lines], med_h * 0.5)

    widths = np.array([L[2] - L[0] for L in lines], dtype=float)
    wvar = float(widths.std() / max(widths.mean(), 1e-6))
    hvar = float(np.std(hs) / max(np.mean(hs), 1e-6))

    fill = sum((min(g[2], x1) - max(g[0], x0)) * (min(g[3], y1) - max(g[1], y0))
               for g in (psr.get("graphic_areas") or [])
               if min(g[2], x1) > max(g[0], x0) and min(g[3], y1) > max(g[1], y0))
    fill = min(fill / max(w * h, 1e-6), 1.0)

    return np.array([
        min(rh / max(h / med_h, 1), 2.0),
        min(rv / max(cols, 1), 4.0),
        numeric, short,
        min(cols / 8.0, 1.0), min(rows / 40.0, 1.0), fact,
        min(lefts / 8.0, 1.0), min(wvar, 2.0), min(hvar, 1.0),
        fill,
        min(len(lines) / max(h / med_h, 1), 2.0),
        min(w / max(h, 1e-6), 4.0),
        min((w * h) / max(psr["width"] * psr["height"], 1e-6), 1.0),
    ], dtype=float)


# --- the learned part -------------------------------------------------------
#
# Logistic regression over the features above, weights in config/table_model.json.
# Inference is a dot product, which is the whole reason the features are computed
# from geometry rather than pixels: a convolutional model would take the shipped
# package from 314 MB to several gigabytes and require a GPU to answer one
# question about one region.
#
# Trained on regions where at least three systems agree, held out by document,
# AUC 0.929. Measured separately against an independent geometric label set
# (ruled grids as tables, rule-free prose as not) at AUC 1.000, which is what
# rules out the model having merely memorised the consensus.
#
# Only the negative direction is used. "This is certainly not a table" carries a
# 0.7% false-accusation rate below 0.05; "this is a table nothing has labelled"
# is still wrong 15-25% of the time even at 0.98, and is not shipped.

import json as _json
import os as _os

_MODEL = None


def load_model(path=None):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    path = path or _os.environ.get("DLA_TABLE_MODEL")
    if not path:
        here = _os.path.dirname(_os.path.dirname(_os.path.dirname(
            _os.path.abspath(__file__))))
        path = _os.path.join(here, "config", "table_model.json")
    try:
        with open(path, encoding="utf8") as f:
            m = _json.load(f)
    except FileNotFoundError:
        return None
    if list(m.get("features") or ()) != list(NAMES):
        raise ValueError("table_model.json was trained on different features")
    _MODEL = {k: np.array(m[k], dtype=float) if k in ("mu", "sd", "w") else m[k]
              for k in m}
    return _MODEL


def score(region, psr, texts=None, model=None):
    """P(this region is a table), or None when it cannot be judged."""
    m = model or load_model()
    v = features(region, psr, texts)
    if m is None or v is None:
        return None
    z = (v - m["mu"]) / m["sd"]
    return float(1.0 / (1.0 + np.exp(-(z @ m["w"] + m["b"]))))
