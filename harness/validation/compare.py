#!/usr/bin/env python3
"""Compare two reading streams for the same page.

Three questions, in the order they matter: did we lose content, did we group it
correctly, and did we read it in the right order.  Class agreement is fourth and
is scored only at bucket granularity, because that is the only granularity that
changes where the content is stored.

Grouping is measured by counting line *pairs* rather than by matching boxes.
Box matching needs a one-to-one assignment and breaks down exactly when a
prediction merges or splits a block -- the case this whole framework exists to
catch.  Pair counting has no matching step: precision falls when a prediction
puts lines together that belong apart (a merge), recall falls when it splits
lines that belong together.  They are reported separately because in a pipeline
that stores blocks, a merge is far more expensive than a split.
"""
import os
from collections import Counter

import numpy as np

import yaml


def _pairs(labels):
    a = np.asarray(labels)
    return (a[:, None] == a[None, :])


def structural_keys(psr):
    """Per line, the structural unit the PDF says it belongs to.

    Grouping precision alone cannot separate a harmless merge from a fatal one.
    A reference built by proximity works at paragraph granularity, so a model
    that emits one region per *section* scores badly while being perfectly
    usable downstream -- measured on a dense page, every one of 53 systems fell
    below a 0.85 precision bar that way.  What actually costs anything is
    merging across a boundary the PDF itself draws: a column gutter, the edge of
    a ruled table, the edge of a figure.  This returns that key so the
    comparison can count only those merges.
    """
    lines = psr["text_lines"]
    bands = psr.get("column_bands") or []
    grids = psr.get("grid_candidates") or []
    gfx = psr.get("graphic_areas") or []

    def _which(box, holders, frac=0.6):
        a = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
        for i, h in enumerate(holders):
            ix1, iy1 = max(box[0], h[0]), max(box[1], h[1])
            ix2, iy2 = min(box[2], h[2]), min(box[3], h[3])
            if ix2 > ix1 and iy2 > iy1 and ((ix2 - ix1) * (iy2 - iy1)) / a >= frac:
                return i
        return None

    keys, band_of = [], []
    for L in lines:
        hits = [i for i, b in enumerate(bands)
                if min(L[2], b[1]) - max(L[0], b[0]) > (L[2] - L[0]) * 0.5]
        band = hits[0] if len(hits) == 1 else None
        band_of.append(band)
        keys.append((band, _which(L, grids), _which(L, gfx)))
    return keys, band_of


def compare(pred, ref, psr=None, buckets_pred=None, buckets_ref=None):
    """Two `assemble()` outputs for one page -> the page's error shape."""
    keys, band_of = structural_keys(psr) if psr else (None, None)
    n = pred["n_lines"]
    pb, rb = pred["line_block"], ref["line_block"]
    common = [i for i in range(n) if pb[i] is not None and rb[i] is not None]

    out = {
        "n_lines": n,
        "n_compared": len(common),
        "orphan_rate": round(len(pred["orphans"]) / n, 4) if n else 0.0,
        "ref_orphan_rate": round(len(ref["orphans"]) / n, 4) if n else 0.0,
        "order_source": pred["order_source"],
    }
    if len(common) < 2:
        out.update({"grouping_precision": None, "grouping_recall": None,
                    "order_tau": None, "bucket_confusion": {}})
        return out

    sa, sb = _pairs([pb[i] for i in common]), _pairs([rb[i] for i in common])
    iu = np.triu_indices(len(common), 1)
    a, b = sa[iu], sb[iu]
    tp = int((a & b).sum())
    out["grouping_precision"] = round(tp / int(a.sum()), 4) if a.sum() else None
    out["grouping_recall"] = round(tp / int(b.sum()), 4) if b.sum() else None

    # The merges that cost something: lines the PDF puts in different columns,
    # different ruled tables or inside/outside a figure, which the prediction
    # nonetheless placed in one region.
    if keys is not None:
        kc = [keys[i] for i in common]
        bnd = np.array([-1 if k[0] is None else k[0] for k in kc])
        grd = np.array([-1 if k[1] is None else k[1] for k in kc])
        gfx = np.array([-1 if k[2] is None else k[2] for k in kc])
        # A full-width line belongs to no band, so it conflicts with nothing --
        # merging a page-wide heading into the column beneath it is correct, and
        # counting it as a column merge would flag every well-formed page.
        both = (bnd[:, None] >= 0) & (bnd[None, :] >= 0)
        band_conflict = both & (bnd[:, None] != bnd[None, :])
        # Table and figure membership is different: merging a line inside a
        # ruled grid with one outside it really does span a structural boundary.
        diff_unit = (band_conflict | (grd[:, None] != grd[None, :])
                     | (gfx[:, None] != gfx[None, :]))[iu]
        offending = a & diff_unit
        out["cross_merge_rate"] = round(
            float(offending.sum()) / int(a.sum()), 4) if a.sum() else 0.0
        # attributed to regions, so a reviewer is told which ones to open
        li = np.array(common)[iu[0]][offending]
        out["cross_merge_regions"] = sorted({int(pb[i]) for i in li})

    # Order: both streams order the same lines, so this is a permutation
    # distance and needs no alignment.
    ppos = {ln: k for k, ln in enumerate(pred["sequence"])}
    rpos = {ln: k for k, ln in enumerate(ref["sequence"])}
    pa = np.array([ppos.get(i, -1) for i in common])
    pr = np.array([rpos.get(i, -1) for i in common])
    d = np.sign(pa[:, None] - pa[None, :]) * np.sign(pr[:, None] - pr[None, :])
    out["order_tau"] = round(float(d[iu].mean()), 4)
    out["order_inversion_rate"] = round(float((d[iu] < 0).mean()), 4)

    # Column ping-pong: how many times the predicted stream leaves one column
    # band and enters another.  A clean two-column page needs exactly one
    # transition; interleaved columns produce fluent, unusable text and show up
    # here and nowhere else.
    if band_of is not None:
        seq = [band_of[i] for i in pred["sequence"] if band_of[i] is not None]
        trans = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        nb = len({b for b in band_of if b is not None})
        out["band_transitions"] = trans
        out["excess_band_transitions"] = max(0, trans - max(nb - 1, 0))

    # Bucket agreement, per line, only where the reference can speak.
    bp = buckets_pred or {b["rank"]: b["bucket"] for b in pred["blocks"]}
    br = buckets_ref or {b["rank"]: b["bucket"] for b in ref["blocks"]}
    # The PSR reference has real evidence for TABLE (ruling lines) and MEDIA
    # (graphic areas) and none at all for the rest, which it calls TEXT because
    # something has to be the default.  Scoring against that default punishes a
    # model for correctly identifying a running header, so only blocks the
    # reference can actually vouch for are compared.  A non-PSR reference -- a
    # gold layout, or another model -- carries no `source` and is trusted whole.
    evidenced = {b["rank"] for b in ref["blocks"]
                 if b.get("source") in (None, "psr.grid", "psr.graphic")}
    conf = Counter()
    n_scored = 0
    for i in common:
        if rb[i] not in evidenced:
            continue
        n_scored += 1
        x, y = bp.get(pb[i]), br.get(rb[i])
        if x and y and x != y:
            conf[f"{y}->{x}"] += 1
    out["bucket_confusion"] = dict(conf)
    out["bucket_scored"] = n_scored
    out["bucket_move_rate"] = round(sum(conf.values()) / n_scored, 4) if n_scored else 0.0
    return out


# Tier 1 in the error model: content lost, blocks merged, order broken, or a
# block routed to the wrong store.  Everything else is recorded, not blocking.
# Tier 1 deliberately does not use grouping precision or order tau.  Both
# compare against a heuristic reference and both mostly measure granularity;
# the spec calls them features, not rules.  What blocks a page is content lost,
# a merge across a boundary the PDF draws, columns read in ping-pong, or a block
# routed to the wrong store.
# Fallbacks only.  The live values live in config/checks.yaml with the corpus
# they were fitted on recorded beside them.
TIER1 = {
    "orphan_rate": 0.12,
    "cross_merge_rate": 0.25,
    "excess_band_transitions": 2,
    "bucket_move_rate": 0.35,
    "bucket_moves": ("TABLE->TEXT", "TEXT->TABLE", "TEXT->DISCARD", "TEXT->MEDIA"),
}


def load_tier1(path=None):
    path = path or os.environ.get("DLA_CHECKS_CONFIG")
    if not path:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(here, "config", "checks.yaml")
    t = dict(TIER1)
    try:
        with open(path, encoding="utf8") as f:
            t.update((yaml.safe_load(f) or {}).get("compare") or {})
    except FileNotFoundError:
        pass
    return t


def verdict(cmp_, t=None):
    t = {**load_tier1(), **(t or {})}
    why = []
    if cmp_["orphan_rate"] > t["orphan_rate"]:
        why.append(f"content lost: {cmp_['orphan_rate']:.1%} of lines sit in no region")
    cm = cmp_.get("cross_merge_rate")
    if cm is not None and cm > t["cross_merge_rate"]:
        regs = cmp_.get("cross_merge_regions") or []
        why.append(f"merged across a column, table or figure boundary: "
                   f"{cm:.1%} of grouped line pairs"
                   + (f", regions {regs[:4]}" if regs else ""))
    xb = cmp_.get("excess_band_transitions")
    if xb is not None and xb > t["excess_band_transitions"]:
        why.append(f"columns read in ping-pong: {xb} more band changes than the "
                   f"page structure allows")
    moved = sum(v for k, v in cmp_["bucket_confusion"].items() if k in t["bucket_moves"])
    rate = moved / max(cmp_.get("bucket_scored") or 0, 1) if cmp_.get("bucket_scored") else 0.0
    if rate > t["bucket_move_rate"]:
        why.append(f"{moved} lines ({rate:.0%}) routed to the wrong store")
    return {"tier1": bool(why), "reasons": why}
