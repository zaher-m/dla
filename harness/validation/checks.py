#!/usr/bin/env python3
"""Deterministic checks C1-C8: pure functions of a layout and the PDF.

No model, no labels, no GPU.  Each check returns findings attributed to specific
regions, because a reviewer needs to be told *which region to open*, and a
sentence describing the defect beats a score every time.

Three rules hold throughout.

  Severity is not a weight.  BLOCK means the page cannot be correct and
  escalates regardless of any score; MAJOR and ADV are features the risk model
  weighs.  Only BLOCK checks carry that authority, and each must earn it by
  measuring under a 0.5% false-positive rate once gold exists.

  A check that cannot run reports nothing and is recorded as unavailable.  It
  never reports a pass.  `run()` returns the set it skipped, and the caller is
  expected to escalate on an unavailable BLOCK check rather than accept.

  Thresholds live in config/checks.yaml.  The values here are fallbacks.
"""
import os
from collections import Counter, defaultdict

import numpy as np
import yaml

from validation import assemble as assemblemod
from validation import compare as comparemod
from validation import document as docmod
from validation import graphics as gfxmod
from validation import lines as linesmod
from validation import orderlm
from validation import psr_layout as psrlayout
from validation import tablefeat
from validation import tables as tablemod
from validation.buckets import bucket as to_bucket, TEXT, TABLE, MEDIA, DISCARD

BLOCK, MAJOR, ADV = "BLOCK", "MAJOR", "ADV"

REGISTRY = []


def check(cid, severity, family, needs=()):
    """Register a check.  `needs` names the inputs it cannot run without."""
    def deco(fn):
        REGISTRY.append({"id": cid, "severity": severity, "family": family,
                         "needs": tuple(needs), "fn": fn, "doc": fn.__doc__})
        return fn
    return deco


DEFAULTS = {
    "min_text_lines": 10,
    "orphan_line_rate": 0.12,
    "orphan_area_rate": 0.12,
    "lm_min_lines": 12,
    "lm_min_prose": 0.15,
    "lm_margin": 0.5,
    "orphan_cluster": 3,
    "coverage_min_lost": 4,
    "margin_band": 0.08,
    "orphan_in_column_rate": 0.12,
    "orphan_column_min_lines": 10,
    "footnote_lines": 4,
    "graphic_miss_count": 2,
    "overlap_regions": 4,
    "overlap_region_rate": 0.30,
    "duplicate_region_rate": 0.20,
    "footnote_band": 0.12,
    "graphic_miss_area": 0.02,
    "line_cut_rate": 0.15,
    "cut_min": 0.15,
    "cut_max": 0.85,
    "gutter_cover": 0.6,
    "band_span": 0.2,
    "band_imbalance": 0.85,
    "double_emit_rate": 0.02,
    "overlap_frac": 0.50,
    "grid_cover": 0.5,
    "media_text_lines": 5,
    "media_graphic_frac": 0.15,
    "media_max_text_frac": 0.18,
    "table_min_lines": 6,
    "table_score_floor": 0.05,
    "graphic_content_floor": 0.80,
    "discard_body_lines": 3,
    "discard_line_rate": 0.15,
    "out_of_bounds": 0.02,
    "page_dominating": 0.85,
    "coverage_floor": 0.40,
    "excess_band_transitions": 2,
    "order_inversions": 2,
    "order_tau": 0.90,
    "contiguity_min_lines": 6,
    "contiguity_slack": 0.25,
    "backward_jumps": 3,
    "rtl_row_min_pairs": 3,
    "rtl_row_wrong": 0.5,
    "doc_min_pages": 5,
    "running_rate": 0.8,
    "doc_body_lines": 20,
    "region_z": 4.0,
    "font_shift": 0.4,
    "aspect_ratio": 60.0,
}


def load_thresholds(path=None):
    path = path or os.environ.get("DLA_CHECKS_CONFIG")
    if not path:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(here, "config", "checks.yaml")
    t = dict(DEFAULTS)
    try:
        with open(path, encoding="utf8") as f:
            t.update((yaml.safe_load(f) or {}).get("checks") or {})
    except FileNotFoundError:
        pass
    return t


# ---------------------------------------------------------------- geometry --
def _area(b):
    return max((b[2] - b[0]) * (b[3] - b[1]), 0.0)


def _inter(a, b):
    return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))


def _cover(inner, outer):
    return _inter(inner, outer) / max(_area(inner), 1e-6)


def _band_of(box, bands, W, full=0.62):
    if not bands or (box[2] - box[0]) > W * full:
        return None
    hits = [i for i, b in enumerate(bands)
            if min(box[2], b[1]) - max(box[0], b[0]) > (box[2] - box[0]) * 0.5]
    return hits[0] if len(hits) == 1 else None


def figure_areas(psr, floor=None, max_text_frac=0.18):
    """Graphic areas that are really figures.

    `graphic_areas` clusters filled vector drawings, and a financial table drawn
    with coloured header rows and banded cell fills is a cluster of filled
    drawings.  Whole tables therefore arrive here as figures, and a check asking
    "where is the figure region for this graphic" then charges a detector for
    not boxing a table as an image.  A third of the misattributed clusters are
    not tables either but prose on a tint -- a heading, a source note, a framed
    paragraph.

    `validation.graphics` decides it from what the cluster is drawn from.  The
    density band below is the fallback for a reference built before
    `graphic_shape` existed; it is documented in E4 as not separating the two,
    and `selftest` asserts the model is present so a package that quietly loses
    it fails rather than degrading to this.
    """
    areas = psr.get("graphic_areas") or []
    if psr.get("graphic_shape") and gfxmod.load_model() is not None:
        fig, _content, unknown = gfxmod.split(
            psr, load_thresholds()["graphic_content_floor"]
            if floor is None else floor)
        return [areas[i] for i in sorted(fig + unknown)]
    out = []
    for g in areas:
        ga = max(_area(g), 1e-6)
        ta = sum(_area(L) for L in psr["text_lines"] if _cover(L, g) >= 0.6)
        if ta / ga <= max_text_frac:
            out.append(g)
    return out


def _standalone_content(psr, floor, tol=0.25):
    """Content clusters that are a block, not one column of a wider table.

    A tinted paragraph or a whole shaded table has to be read in one go.  One
    column of a banded table does not -- its cells are read one per row, so
    requiring them to arrive together fires on every table that is right, which
    it did on 36 pages.  A cluster whose lines share rows with lines outside it
    is one of those, the same test `page_columns` uses to tell a table's columns
    from a page's.

    Unioning row-sharing clusters back into whole tables first was tried and is
    worse: on a two-panel page the union reaches across both panels, and C4-08's
    firing on the reference's own layout, where there is no defect to find, went
    from 1 page to 7.
    """
    if not psr.get("graphic_shape") or gfxmod.load_model() is None:
        return []
    areas = psr.get("graphic_areas") or []
    _fig, content, _unk = gfxmod.split(psr, floor)
    out = []
    for i in content:
        g = areas[i]
        held = gfxmod.held_lines(g, psr)
        if not held:
            continue
        outside = [L for L in psr["text_lines"]
                   if _cover(L, g) < 0.6 and _inter(L, g) == 0.0]
        paired = 0
        for L in held:
            h = max(L[3] - L[1], 1e-6)
            if any(min(L[3], o[3]) - max(L[1], o[1]) > h * 0.5 for o in outside):
                paired += 1
        if paired / len(held) <= tol:
            out.append(g)
    return out


def page_columns(psr, tol=0.5):
    """Column bands that separate independent flows of running text.

    `column_bands` comes from the horizontal ink profile, which on a financial
    page resolves a table's numeric columns into separate bands.  Every check
    built on those bands then misreads the page: reading a table row by row
    registers as 57-88 switches between "columns", and a region spanning a row
    looks like a merge across two of them.

    Adjacent bands are merged back together when the text on either side of the
    boundary belongs to shared rows -- the same test that separates a column
    gutter from a gap between table columns.  Two columns of an article leave
    the boundary intact because their lines sit at unrelated heights.
    """
    bands = psr.get("column_bands") or []
    if len(bands) < 2:
        return list(bands)
    body = psr.get("body_text_lines") or []
    out = [list(bands[0])]
    for b in bands[1:]:
        left = [L for L in body if L[2] <= out[-1][1] + 2 and L[0] >= out[-1][0] - 2]
        right = [L for L in body if L[0] >= b[0] - 2 and L[2] <= b[1] + 2]
        paired = 0
        for a in left:
            h = max(a[3] - a[1], 1e-6)
            if any(min(a[3], c[3]) - max(a[1], c[1]) > h * 0.5 for c in right):
                paired += 1
        if left and right and paired / len(left) > tol:
            out[-1][1] = b[1]          # same rows on both sides: one table
        else:
            out.append(list(b))
    return out


def column_gutters(psr, tol=0.5):
    """Whitespace corridors that really do separate two columns of running text.

    `find_gutters` reports vertical whitespace in the body text, and on a
    financial page the widest such corridors are the gaps between a table's own
    numeric columns.  Filtering by ruled-table membership is not enough: unruled
    tables are common here and produce no grid candidate at all, so rendering
    the findings still showed every one of them was a detector correctly boxing
    a table.

    What separates the two is whether the sides flow independently.  Across a
    real column gutter the left and right text are separate flows and their
    lines sit at unrelated heights; across a gap between table columns the
    lines on both sides belong to the same row and share a y band.
    """
    out = []
    for g in (psr.get("gutters") or []):
        left = [L for L in (psr.get("body_text_lines") or [])
                if L[2] <= g[0] + 2 and L[2] > g[0] - (g[2] - g[0]) * 6]
        right = [L for L in (psr.get("body_text_lines") or [])
                 if L[0] >= g[2] - 2 and L[0] < g[2] + (g[2] - g[0]) * 6]
        if not left or not right:
            continue
        paired = 0
        for a in left:
            h = max(a[3] - a[1], 1e-6)
            if any(min(a[3], b[3]) - max(a[1], b[1]) > h * 0.5 for b in right):
                paired += 1
        if paired / len(left) <= tol:
            out.append(g)
    return out


def _f(cid, sev, msg, regions=(), value=None):
    return {"id": cid, "severity": sev, "message": msg,
            "regions": sorted(regions), "value": value}


# ------------------------------------------------------- C1  coverage -------
@check("C1-01", BLOCK, "C1", needs=("psr", "body_lines"))
def c1_01(x):
    """Body glyph lines that no region covers.

    Guarded on how much was lost, not on how much there was.  A rate alone
    reported 100% of the page's content lost when a page whose text all sits
    inside ruled tables had two body lines and both were missed.  Guarding on
    the size of the body set instead suppressed the real findings, because a
    dense table page has few body lines by construction -- the misses that
    matter are large in absolute terms as well as in proportion.
    """
    lost = len(x["read_orphans"])
    if lost < x["t"]["coverage_min_lost"]:
        return
    r = lost / max(len(x["read_lines"]), 1)
    if r > x["t"]["orphan_line_rate"]:
        return [_f("C1-01", BLOCK,
                   f"content lost: {r:.1%} of body text lines sit in no region",
                   value=round(r, 4))]


@check("C1-02", BLOCK, "C1", needs=("psr", "body_lines"))
def c1_02(x):
    """Glyph ink area no region covers -- catches one big miss C1-01 dilutes.

    Deliberately measured on the reference's own boxes rather than on
    reconstructed lines: glyph area does not change when a line is fragmented,
    which is what makes this the check that still means something on a page
    whose line count does not.
    """
    if len(x["read_orphans"]) < x["t"]["coverage_min_lost"]:
        return
    tot = sum(_area(L) for L in x["body_lines"])
    lost = sum(_area(x["body_lines"][i]) for i in x["orphan_content"])
    r = lost / max(tot, 1e-6)
    if r > x["t"]["orphan_area_rate"]:
        return [_f("C1-02", BLOCK,
                   f"content lost: {r:.1%} of body text area is uncovered",
                   value=round(r, 4))]


@check("C1-03", BLOCK, "C1", needs=("psr", "body_lines"))
def c1_03(x):
    """Vertically contiguous run of orphans -- a missed block, not stray glyphs."""
    if not x["read_orphans"]:
        return
    lines = x["read_lines"]
    orph = sorted(x["read_orphans"], key=lambda i: lines[i][1])
    lh = x["line_h"]
    run, best, cur = 1, 1, [orph[0]]
    bestset = list(cur)
    for a, b in zip(orph, orph[1:]):
        if lines[b][1] - lines[a][3] <= 1.8 * lh:
            run += 1; cur.append(b)
        else:
            if run > best:
                best, bestset = run, list(cur)
            run, cur = 1, [b]
    if run > best:
        best, bestset = run, list(cur)
    if best >= x["t"]["orphan_cluster"]:
        y = lines[bestset[0]][1]
        return [_f("C1-03", BLOCK,
                   f"a block of {best} consecutive text lines was missed "
                   f"(from y={y:.0f})", value=best)]


@check("C1-04", BLOCK, "C1", needs=("psr", "body_lines", "bands"))
def c1_04(x):
    """Orphans inside a real text column, as opposed to page margins.

    Measured as a share of the lines in those columns, not as a count.  Eight
    uncovered lines is a defect on a 40-line page and noise on a 300-line one,
    and the count form fired on a quarter of a random sample.
    """
    inband = [i for i, L in enumerate(x["read_lines"])
              if _band_of(L, x["bands"], x["W"]) is not None]
    if len(inband) < x["t"]["orphan_column_min_lines"]:
        return
    orph = set(x["read_orphans"])
    n = sum(1 for i in inband if i in orph)
    r = n / len(inband)
    if r > x["t"]["orphan_in_column_rate"]:
        return [_f("C1-04", BLOCK,
                   f"{r:.0%} of the text inside real columns ({n} lines) was "
                   f"missed", value=round(r, 4))]


@check("C1-05", BLOCK, "C1", needs=("psr", "body_lines"))
def c1_05(x):
    """No text region at all on a page that plainly has text.

    Counted over the lines no TEXT *or TABLE* region covers, not over every line
    on the page.  A page that is one large table, correctly boxed as a table,
    holds plenty of text and needs no text region; scoring this on the page's
    whole line count charges the system for being right, which it did on four
    pages once a shaded table's cells were recovered into the body.  Counting
    only uncovered lines was tried instead and gives the defect back: when every
    region is relabelled `figure` the lines are still covered, and the check
    that caught that on 94% of injected pages fell to 31%.  A table's cells
    belong in a table region; body text swallowed by a figure does not.
    """
    keep = [r["bbox"] for r, b in zip(x["regions"], x["region_bucket"])
            if b in (TEXT, TABLE)]
    n = sum(1 for L in x["read_lines"]
            if not any(_cover(L, b) >= 0.6 for b in keep))
    if n >= x["t"]["min_text_lines"] and not any(
            b == TEXT for b in x["region_bucket"]):
        return [_f("C1-05", BLOCK,
                   f"no text region on a page carrying {n} text lines that no "
                   f"text or table region covers", value=n)]


@check("C1-06", MAJOR, "C1", needs=("psr", "body_lines"))
def c1_06(x):
    """Uncovered lines in the footnote band at the foot of the page."""
    cut = x["H"] * (1 - x["t"]["footnote_band"])
    n = sum(1 for i in x["read_orphans_all"] if x["read_lines"][i][1] >= cut)
    if n >= x["t"]["footnote_lines"]:
        return [_f("C1-06", MAJOR,
                   f"{n} uncovered text lines in the bottom "
                   f"{x['t']['footnote_band']:.0%} of the page -- footnotes are "
                   f"a common systematic drop", value=n)]


@check("C1-07", MAJOR, "C1", needs=("psr", "graphics"))
def c1_07(x):
    """A real graphic with no MEDIA region on it."""
    page = x["W"] * x["H"]
    miss = [g for g in x["graphics"]
            if _area(g) > x["t"]["graphic_miss_area"] * page
            and not any(_cover(g, r["bbox"]) > 0.5
                        for r, b in zip(x["regions"], x["region_bucket"]) if b == MEDIA)]
    if len(miss) >= x["t"]["graphic_miss_count"]:
        return [_f("C1-07", MAJOR,
                   f"{len(miss)} graphic area(s) have no figure region and will "
                   f"never reach the object store", value=len(miss))]


# ------------------------------------------------------- C2  boundaries -----
@check("C2-01", MAJOR, "C2", needs=("psr", "body_lines"))
def c2_01(x):
    """Region boundaries cutting through glyph lines.

    Counts *lines that are cut*, not (line, region) pairs.  Summing pairs let a
    line clipped by three regions count three times and produced "rates" above
    190%.

    A cut is measured horizontally.  Area coverage cannot tell a genuine split
    from Arabic diacritics and ascenders poking a few pixels past an otherwise
    perfect boundary -- `core.metrics` documents the same trap -- and on an
    area test every line of a correctly boxed paragraph reads as cut.  A region
    that ends partway along a line's width has really divided it.
    """
    bad = defaultdict(int)
    cut = 0
    lo, hi = x["t"]["cut_min"], x["t"]["cut_max"]
    for L in x["body_lines"]:
        w = max(L[2] - L[0], 1e-6)
        hit = [i for i, r in enumerate(x["regions"])
               if lo < (max(0.0, min(L[2], r["bbox"][2]) - max(L[0], r["bbox"][0])) / w) < hi
               and _cover(L, r["bbox"]) > 0.05]
        if hit:
            cut += 1
            for i in hit:
                bad[i] += 1
    rate = cut / max(len(x["body_lines"]), 1)
    if rate > x["t"]["line_cut_rate"]:
        return [_f("C2-01", MAJOR,
                   f"{rate:.0%} of text lines are cut by a region boundary",
                   regions=sorted(bad, key=lambda i: -bad[i])[:8],
                   value=round(rate, 4))]


# Demoted from MAJOR.  Rendering the findings showed most are empty form cells
# and decorative banners whose text is baked into an image -- both genuinely
# hold no extractable text, and neither is a layout error.  It stays as a weak
# feature because a region with no text is still worth knowing about.
@check("C2-06", ADV, "C2", needs=("psr",))
def c2_06(x):
    """A text region holding no glyphs at all.

    Tested geometrically rather than through the reading stream.  A line is
    assigned to exactly one region -- whichever covers most of it -- so where
    two regions overlap the loser holds no assigned lines while plainly sitting
    on text.  Scoring that as an empty region fired on 72% of pages.
    """
    empty = []
    for i, (r, b) in enumerate(zip(x["regions"], x["region_bucket"])):
        if b != TEXT:
            continue
        # Either the region holds a line, or it sits on glyphs: a box tighter
        # than one line of text contains no line and is not therefore empty.
        holds = any(_cover(L, r["bbox"]) >= 0.5 for L in x["all_lines"])
        sits = any(_cover(r["bbox"], L) >= 0.5 for L in x["all_lines"])
        if not holds and not sits:
            empty.append(i)
    if empty:
        return [_f("C2-06", ADV,
                   f"{len(empty)} text region(s) contain no text",
                   regions=empty, value=len(empty))]


# ------------------------------------------------------- C3  columns --------
# Demoted from BLOCK.  Fifteen findings were rendered across three rounds of
# fixes -- filtering gutters inside ruled tables, then inside qualified tables,
# then by whether the two sides flow independently -- and not one was a real
# column merge.  On this corpus the widest vertical whitespace is almost always
# inside a table, ruled or not, and a detector boxing that table spans it
# correctly.  C3-02, which works from the page-level ink profile rather than
# local whitespace, covers the same failure without the false positives.  This
# stays as a weak feature and must never gate a page.
@check("C3-01", ADV, "C3", needs=("psr", "gutters"))
def c3_01(x):
    """A text region straddling a real whitespace corridor between columns.

    Only corridors outside tables count.  `find_gutters` looks for vertical
    whitespace in the body text, and on a financial page the widest such
    corridors are the gaps between a table's own numeric columns -- rendering
    the findings showed most of them were a detector correctly boxing a table
    that happens to contain white space.  A region spanning a table's internal
    gap has merged nothing.
    """
    bad = []
    for i, (r, b) in enumerate(zip(x["regions"], x["region_bucket"])):
        if b != TEXT:
            continue
        for g in x["page_gutters"]:
            w = g[2] - g[0]
            if w > 0 and (min(r["bbox"][2], g[2]) - max(r["bbox"][0], g[0])) \
                    > w * x["t"]["gutter_cover"]:
                bad.append(i); break
    if bad:
        return [_f("C3-01", ADV,
                   f"{len(bad)} text region(s) span the whitespace corridor "
                   f"between two columns", regions=bad, value=len(bad))]


@check("C3-02", BLOCK, "C3", needs=("psr", "bands"))
def c3_02(x):
    """A text region overlapping two column bands."""
    if len(x["bands"]) < 2:
        return
    bad = []
    for i, (r, b) in enumerate(zip(x["regions"], x["region_bucket"])):
        if b != TEXT:
            continue
        hit = sum(1 for bd in x["bands"]
                  if min(r["bbox"][2], bd[1]) - max(r["bbox"][0], bd[0])
                  > (bd[1] - bd[0]) * x["t"]["band_span"])
        if hit >= 2:
            bad.append(i)
    if bad:
        return [_f("C3-02", BLOCK,
                   f"{len(bad)} text region(s) cover two column bands",
                   regions=bad, value=len(bad))]


@check("C3-05", MAJOR, "C3", needs=("psr", "bands"))
def c3_05(x):
    """One column detected, the other silently ignored."""
    if len(x["bands"]) < 2:
        return
    per = Counter()
    for L in x["body_lines"]:
        b = _band_of(L, x["bands"], x["W"])
        if b is not None:
            per[b] += 1
    if len(per) < 2 or min(per.values()) < 5:
        return
    got = Counter()
    for i, r in enumerate(x["regions"]):
        b = _band_of(r["bbox"], x["bands"], x["W"])
        if b is not None:
            got[b] += len(x["region_lines"][i])
    tot = sum(got.values()) or 1
    share = max(got.values()) / tot if got else 1.0
    if share > x["t"]["band_imbalance"]:
        return [_f("C3-05", MAJOR,
                   f"one column holds {share:.0%} of the detected text while the "
                   f"page splits its lines {dict(per)}", value=round(share, 3))]


# ------------------------------------------------------- C4  order ----------
@check("C4-03", BLOCK, "C4", needs=("psr", "bands", "stream"))
def c4_03(x):
    """Reading order ping-ponging between column bands."""
    if len(x["bands"]) < 2:
        return
    seq = []
    for i in x["stream"]["sequence"]:
        b = _band_of(x["all_lines"][i], x["bands"], x["W"])
        if b is not None:
            seq.append(b)
    trans = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    excess = trans - (len(x["bands"]) - 1)
    if excess > x["t"]["excess_band_transitions"]:
        return [_f("C4-03", BLOCK,
                   f"columns are read in ping-pong: {trans} switches between "
                   f"columns where the page structure allows "
                   f"{len(x['bands']) - 1}", value=excess)]


@check("C4-02", BLOCK, "C4", needs=("psr", "bands", "stream"))
def c4_02(x):
    """Within one column, blocks are not read top to bottom."""
    per = defaultdict(list)
    for blk in x["stream"]["blocks"]:
        b = _band_of(blk["bbox"], x["bands"], x["W"])
        if b is not None and blk["lines"]:
            per[b].append((blk["rank"], blk["bbox"][1], blk["region"]))
    bad, regions = 0, []
    # A block is out of order when the stream reaches it after a block that
    # starts materially lower on the page.  The tolerance is a whole block, not
    # a line: side-by-side blocks in one column differ by a few pixels of top
    # edge and are not a reading-order error.
    tol = max(x["line_h"] * 3, 40.0)
    for b, items in per.items():
        items.sort()
        for (r1, y1, g1), (r2, y2, g2) in zip(items, items[1:]):
            if y2 < y1 - tol:
                bad += 1
                regions.append(g2)
    if bad >= x["t"]["order_inversions"]:
        return [_f("C4-02", BLOCK,
                   f"{bad} block(s) are read out of top-to-bottom order within "
                   f"their own column", regions=regions, value=bad)]


# ------------------------------------------------------- C5  duplication ----
# ---- order invariants -------------------------------------------------------
#
# A bounding-box algorithm can always produce *an* order; nothing about it can
# say the order is right, which is why comparing one derivation against another
# measures only their difference in granularity.  But an order is not only right
# or wrong relative to a reference -- it can be *invalid on its own terms*, and
# those constraints need no reference and no model-supplied order.  These three
# hold whoever produced the sequence, so unlike C4-07 they cover the systems
# that emit no order at all, which on this corpus is most of them.


@check("C4-08", BLOCK, "C4", needs=("psr", "stream"))
def c4_08(x):
    """A table or figure is read interleaved with the text around it.

    The lines inside one structural unit have to come out together.  When a
    table's rows arrive split around a paragraph, the extracted text is fluent
    and the table is destroyed -- the failure mode that survives every metric
    based on coverage or on boxes.
    """
    pos = x["read_all_pos"]
    # Figures, plus the filled clusters that hold content and stand alone.
    # Taking every content cluster was tried and rendered: a banded table is
    # clustered one column at a time, and a column's cells are *meant* to arrive
    # spread out, one per row -- 36 findings, all on tables read correctly.  A
    # cluster whose lines share rows with lines outside it is one of those, the
    # same test `page_columns` uses to tell a table's columns from a page's.
    units = ([q["bbox"] for q in tablemod.qualified(x["psr"])]
             + x["graphics"] + x["content_units"])
    bad = []
    for u in units:
        inside = [pos[k] for k, L in enumerate(x["read_all"])
                  if k in pos and _cover(L, u) >= 0.6]
        if len(inside) < x["t"]["contiguity_min_lines"]:
            continue
        span = max(inside) - min(inside) + 1
        intruders = span - len(inside)
        if intruders > len(inside) * x["t"]["contiguity_slack"]:
            bad.append((len(inside), intruders))
    if bad:
        n, intr = max(bad, key=lambda b: b[1])
        # "Spliced into" described the count but not the defect: the lines of one
        # table or figure are not read consecutively, and a reviewer needs to be
        # told that rather than to picture an insertion.
        return [_f("C4-08", BLOCK,
                   f"the {n} lines of one table or figure are not read together: "
                   f"{intr} lines of other content fall between the first and "
                   f"the last", value=intr)]


@check("C4-09", ADV, "C4", needs=("psr", "stream"))
def c4_09(x):
    """The stream jumps back up the page without changing column.

    Only counted where the two blocks sit in the same column band, or the page
    has none: moving from the foot of one column to the head of the next is a
    jump backwards and is correct.

    Advisory, on evidence: run against the PSR reference layout -- regions taken
    straight from the PDF's geometry -- it fires on 16.3% of pages, twice the
    9.2% it fires on a model.  A check that fires more often on the correct
    region set is measuring `assemble.derive_order`, not the layout, and cannot
    attribute the defect to anything.
    """
    blocks = [b for b in sorted(x["stream"]["blocks"], key=lambda b: b["rank"])
              if b["lines"]]
    tol = max(x["line_h"] * 3, 40.0)
    bad = []
    for a, b in zip(blocks, blocks[1:]):
        if b["bbox"][1] >= a["bbox"][3] - tol:
            continue
        ba = _band_of(a["bbox"], x["bands"], x["W"])
        bb = _band_of(b["bbox"], x["bands"], x["W"])
        if x["bands"] and ba is not None and bb is not None and ba != bb:
            continue
        bad.append(b["region"])
    if len(bad) >= x["t"]["backward_jumps"]:
        return [_f("C4-09", ADV,
                   f"the reading order jumps back up the page {len(bad)} times "
                   f"without moving to another column",
                   regions=bad, value=len(bad))]


@check("C4-10", MAJOR, "C4", needs=("psr", "stream"))
def c4_10(x):
    """On a right-to-left page, blocks sharing a row are read left to right.

    The most damaging order error available on this corpus, and invisible to
    every other check: the text is complete, the boxes are right, and the
    columns of a row come out reversed.

    Not blocking, on the same control as C4-09 though less clear cut: 4.1% on
    the PSR reference layout against 3.1% on a model.  The reference is row-first
    and so splits a table row into side-by-side cells, which is exactly the
    configuration this check tests -- the comparison is unfair to it rather than
    damning.  It stays a feature until annotated right-to-left pages can settle
    which of the two is being measured.
    """
    if x["route"].get("direction") != "rtl":
        return
    blocks = [b for b in sorted(x["stream"]["blocks"], key=lambda b: b["rank"])
              if b["lines"]]
    wrong = tot = 0
    seen = []
    for a, b in zip(blocks, blocks[1:]):
        h = min(a["bbox"][3] - a["bbox"][1], b["bbox"][3] - b["bbox"][1])
        ov = min(a["bbox"][3], b["bbox"][3]) - max(a["bbox"][1], b["bbox"][1])
        if h <= 0 or ov < h * 0.5:
            continue                      # not the same row
        tot += 1
        if b["bbox"][0] > a["bbox"][0]:   # moved rightwards on an RTL page
            wrong += 1
            seen.append(b["region"])
    if tot >= x["t"]["rtl_row_min_pairs"] and wrong / tot > x["t"]["rtl_row_wrong"]:
        return [_f("C4-10", MAJOR,
                   f"{wrong} of {tot} side-by-side blocks are read left to "
                   f"right on a right-to-left page",
                   regions=seen, value=round(wrong / tot, 3))]


@check("C4-07", MAJOR, "C4", needs=("psr", "reference_stream"))
def c4_07(x):
    """The reading order disagrees with the order the page itself implies.

    Runs only where the model supplied an order of its own.  Where it did not,
    the pipeline derives one from the page geometry -- and so does the reference,
    so the check would be comparing a derivation against itself and reporting
    the difference in region granularity as a reading-order error.  It fired on
    31% of pages that way, across four systems none of which emits an order at
    all.

    Where a model does supply one this is the only order check that survives on
    a single-column page, which after the column bands were corrected is nearly
    every page here.  The reference is heuristic, so this is a feature and not a
    gate: it catches a right-to-left page read left to right, or two flows
    interleaved, without claiming authority in a disputed case.
    """
    if x["stream"]["order_source"] != "model":
        return
    c = comparemod.compare(x["stream"], x["reference_stream"], psr=x["psr"])
    tau = c.get("order_tau")
    if tau is None:
        return
    if tau < x["t"]["order_tau"]:
        inv = c.get("order_inversion_rate") or 0.0
        return [_f("C4-07", MAJOR,
                   f"reading order disagrees with the page's own geometry: "
                   f"{inv:.0%} of line pairs are read in the opposite order "
                   f"(tau {tau:.3f}, order from the {x['stream']['order_source']})",
                   value=round(tau, 4))]


@check("C4-11", ADV, "C4", needs=("psr", "stream", "words"))
def c4_11(x):
    """The order reads worse as language than a plainly available alternative.

    The only order check that reads the words.  It compares the stream's own
    junctions against the same lines taken in simple top-to-bottom order: if the
    page really is one flowing column the two are close, and if the stream has
    interleaved two columns the alternative reads better.

    Advisory and gated on prose.  A junction between two numeric table cells
    carries no language at all, and the measured accuracy falls from 94% on
    pages with prose to 80% on table-like ones (E5).  80% is a feature; nothing
    at that rate may gate a page.
    """
    txt = x["line_text"]
    seq = [i for i in x["stream"]["sequence"] if i < len(txt)]
    if len(seq) < x["t"]["lm_min_lines"]:
        return
    words = [txt[i] for i in seq]
    if orderlm.prosiness(words) < x["t"]["lm_min_prose"]:
        return                     # no language to read; the score means nothing
    got = orderlm.score(x["lm"], words)
    alt_idx = sorted(seq, key=lambda i: (round(x["all_lines"][i][1], 1),
                                         x["all_lines"][i][0]))
    if alt_idx == seq or got is None:
        return
    alt = orderlm.score(x["lm"], [txt[i] for i in alt_idx])
    if alt is None:
        return
    if alt - got > x["t"]["lm_margin"]:
        return [_f("C4-11", ADV,
                   f"the reading order joins words less plausibly than simply "
                   f"reading the page top to bottom ({got:.2f} against "
                   f"{alt:.2f} per junction)", value=round(alt - got, 3))]


@check("C5-01", MAJOR, "C5", needs=())
def c5_01(x):
    """Two same-bucket regions overlapping substantially."""
    bad = set()
    n = len(x["regions"])
    for i in range(n):
        for j in range(i + 1, n):
            if x["region_bucket"][i] != x["region_bucket"][j]:
                continue
            a, b = x["regions"][i]["bbox"], x["regions"][j]["bbox"]
            small = min(_area(a), _area(b))
            if not small:
                continue
            ov = _inter(a, b) / small
            # Containment is hierarchy, not duplication: a heading box inside a
            # block is how several systems express structure, and a consumer
            # resolves it by nesting.  Only siblings that partly overlap are a
            # defect.
            if ov > 0.9:
                continue
            if ov > x["t"]["overlap_frac"]:
                bad.add(i); bad.add(j)
    rate = len(bad) / max(n, 1)
    if rate > x["t"]["overlap_region_rate"] and len(bad) >= x["t"]["overlap_regions"]:
        return [_f("C5-01", MAJOR,
                   f"{len(bad)} of {n} regions ({rate:.0%}) partly overlap "
                   f"another region of the same kind",
                   regions=bad, value=round(rate, 4))]


@check("C5-03", MAJOR, "C5", needs=("psr",))
def c5_03(x):
    """A glyph line covered by two text regions is stored twice."""
    dup = 0
    seen = set()
    for li, L in enumerate(x["all_lines"]):
        owners = [i for i, (r, b) in enumerate(zip(x["regions"], x["region_bucket"]))
                  if b == TEXT and _cover(L, r["bbox"]) >= 0.5]
        if len(owners) < 2:
            continue
        # Nested regions are one record after the consumer resolves containment;
        # a line is only emitted twice when two regions genuinely sit side by
        # side over it.  Counting nesting put this at 90% of lines on some pages.
        boxes = [x["regions"][i]["bbox"] for i in owners]
        siblings = [(i, j) for a, i in enumerate(owners)
                    for j in owners[a + 1:]
                    if _cover(x["regions"][i]["bbox"], x["regions"][j]["bbox"]) < 0.9
                    and _cover(x["regions"][j]["bbox"], x["regions"][i]["bbox"]) < 0.9]
        if siblings:
            dup += 1
            seen.update(i for pair in siblings for i in pair)
    rate = dup / max(len(x["all_lines"]), 1)
    if rate > x["t"]["double_emit_rate"]:
        return [_f("C5-03", MAJOR,
                   f"{rate:.1%} of text lines fall inside two text regions and "
                   f"will be written to the index twice",
                   regions=seen, value=round(rate, 4))]


# ------------------------------------------------------- C6  buckets --------
@check("C5-04", BLOCK, "C5", needs=("psr",))
def c5_04(x):
    """Wholesale duplication: the detector emitted its own output twice.

    C5-01 and C5-03 measure how much is duplicated and stay advisory of a policy
    question -- whether the consumer deduplicates.  This one is not a question.
    A page whose regions are near-exact copies of each other is a broken
    detector, and no downstream arrangement makes that acceptable.

    Added after a defect-injection sweep (validation/sensitivity.py) found that
    duplicating every region on the page changed the block rate by 0.0%: the
    entire page written to its store twice, and nothing in the gate objected.
    """
    n = len(x["regions"])
    if n < 4:
        return
    exact = 0
    seen = []
    tol = max(x["line_h"] * 0.5, 4.0)
    for i, r in enumerate(x["regions"]):
        b = r["bbox"]
        for j, q in seen:
            c = q["bbox"]
            if (x["region_bucket"][i] == x["region_bucket"][j]
                    and all(abs(b[k] - c[k]) <= tol for k in range(4))):
                exact += 1
                break
        else:
            seen.append((i, r))
    rate = exact / max(n, 1)
    if rate > x["t"]["duplicate_region_rate"]:
        return [_f("C5-04", BLOCK,
                   f"{exact} of {n} regions ({rate:.0%}) repeat another region of "
                   f"the same kind at the same place: the page would be written "
                   f"to its store twice", value=round(rate, 4))]


@check("C6-01", BLOCK, "C6", needs=("psr", "grids"))
def c6_01(x):
    """A ruled table with no table region on it.

    Runs against qualified grids only.  The raw PSR grid candidates are clusters
    of ruling strokes filtered by size, and on a random sample 90% of the ones a
    strong detector "missed" were charts, running headers, framed paragraphs or
    empty boxes.  `tables.is_table_like` is what makes this check reportable.
    """
    miss = [g for g in x["table_grids"]
            if not any(_cover(g, r["bbox"]) > x["t"]["grid_cover"]
                       for r, b in zip(x["regions"], x["region_bucket"]) if b == TABLE)]
    if miss:
        return [_f("C6-01", BLOCK,
                   f"{len(miss)} ruled table(s) have no table region: their rows "
                   f"will be stored as prose", value=len(miss))]


# Demoted from BLOCK.  Three formulations were rendered -- line count against
# filtered graphics, text density, then line count against raw graphics -- and
# every finding any of them produced was a chart the detector had boxed
# correctly.  The failure it targets is severe (text sent to object storage
# never reaches the index) so it is kept as a feature, but nothing has shown it
# can identify that failure, and it must not gate a page until something does.
@check("C6-02", BLOCK, "C6", needs=("psr",))
def c6_02(x):
    """A table region with nothing about the page to support it.

    The mirror of C6-01 and the more expensive error of the two in this
    pipeline: a table that is really prose becomes rows in relational storage,
    where a missed table merely becomes prose in the text index.  Evidence is
    ruling lines, or rows made of several aligned cells.

    Two thresholds failed at this before a model was fitted.  Text density does
    not separate a table from a chart, and the structural half of
    `tables.is_table_like` was fitted on grid candidates -- a ruled area plus its
    padding -- so it rejected five unmistakable financial tables when carried
    across to a model's tight box.  A calibration belongs to the kind of box it
    was measured on, and no single quantity had the boundary.

    Blocking on evidence from both directions.  It reports nothing on 424 real
    (system, page) pairs across four systems, and firing on nothing is what an
    inert check also does -- so it was also measured under injected defect:
    reclassifying a third of a page's regions as tables makes it fire on 62% of
    pages against an 11% baseline.  Silent because there is nothing to say, not
    silent because it cannot speak.

    So the evidence is a small learned score over fourteen geometric features
    (validation/tablefeat.py).  Only its negative end is used: below 0.05 the
    statement "this is certainly not a table" is wrong 0.7% of the time, while
    the same model asserting a table it cannot see labelled is wrong 15-25% and
    is not used at all.  The structural test is kept as a second opinion, so a
    region must fail both to be reported.
    """
    bad = []
    for i, (r, b) in enumerate(zip(x["regions"], x["region_bucket"])):
        if b != TABLE or len(x["region_lines"][i]) < x["t"]["table_min_lines"]:
            continue
        if any(_cover(r["bbox"], g) > 0.5 or _cover(g, r["bbox"]) > 0.5
               for g in x["grids"]):
            continue
        ev = tablemod.evidence(r["bbox"], x["psr"])
        rows = ev["h_positions"] >= 3 or ev["n_rows"] >= 3
        cols = ev["v_positions"] >= 2 or ev["multi_cell_rows"] >= 2
        if rows and cols:
            continue
        p = tablefeat.score(r["bbox"], x["psr"], x.get("line_text"))
        # No model, or a region too small to feature: the structural test stands
        # alone, as it did before the model existed.
        if p is None or p < x["t"]["table_score_floor"]:
            bad.append(i)
    if bad:
        return [_f("C6-02", BLOCK,
                   f"{len(bad)} table region(s) sit on no ruling lines and no "
                   f"repeated column structure: their content would be stored "
                   f"as table rows", regions=bad, value=len(bad))]


# C6-04 in the specification -- "a real graphic with no figure region" -- is the
# same check as C1-07 and is not implemented twice.


@check("C6-03", ADV, "C6", needs=("psr",))
def c6_03(x):
    """A figure region that is really text -- content sent to object storage.

    Tested against the *unfiltered* graphic areas.  Two other attempts failed
    here and both failed the same way, by flagging correctly boxed charts: text
    density does not separate a chart from a mis-boxed paragraph at region
    scale, because a chart's axis labels and legend are dense inside a tight
    box.  What does separate them is whether any vector art or image sits under
    the region at all.  A shaded table is filtered out of `figure_areas` for
    other checks, but here it counts as graphic evidence, so a table boxed as a
    figure is missed -- C6-01 and C6-02 are the checks for that.
    """
    bad = []
    for i, (r, b) in enumerate(zip(x["regions"], x["region_bucket"])):
        if b != MEDIA or len(x["region_lines"][i]) <= x["t"]["media_text_lines"]:
            continue
        gfrac = max((_cover(r["bbox"], g) for g in x["raw_graphics"]), default=0.0)
        if gfrac < x["t"]["media_graphic_frac"]:
            bad.append(i)
    if bad:
        return [_f("C6-03", ADV,
                   f"{len(bad)} figure region(s) hold text and no graphic: that "
                   f"text leaves the index entirely", regions=bad, value=len(bad))]


@check("C6-05", BLOCK, "C6", needs=("psr", "bands"))
def c6_05(x):
    """Body content classified as running furniture and dropped.

    Position is tested vertically as well as horizontally.  A column band spans
    the full page height, so a running header sits inside one by construction
    and every finding this check produced was a correctly identified header.
    """
    m = x["t"]["margin_band"]
    top, bot = m * x["H"], (1 - m) * x["H"]
    bad = []
    for i, (r, b) in enumerate(zip(x["regions"], x["region_bucket"])):
        if b != DISCARD or len(x["region_lines"][i]) <= x["t"]["discard_body_lines"]:
            continue
        cy = (r["bbox"][1] + r["bbox"][3]) / 2
        if not (top <= cy <= bot):
            continue
        if _band_of(r["bbox"], x["bands"], x["W"]) is not None:
            bad.append(i)
    if bad:
        return [_f("C6-05", BLOCK,
                   f"{len(bad)} region(s) inside a text column are marked header "
                   f"or footer and their content will be deleted",
                   regions=bad, value=len(bad))]


@check("C6-06", BLOCK, "C6", needs=("psr",))
def c6_06(x):
    """Wholesale deletion: too much of the page routed to DISCARD."""
    n = sum(len(x["region_lines"][i])
            for i, b in enumerate(x["region_bucket"]) if b == DISCARD)
    rate = n / max(len(x["all_lines"]), 1)
    if rate > x["t"]["discard_line_rate"]:
        return [_f("C6-06", BLOCK,
                   f"{rate:.0%} of the page's text is routed to header/footer "
                   f"and dropped", value=round(rate, 4))]


# ------------------------------------------------------- C7  sanity ---------
@check("C7-01", BLOCK, "C7", needs=())
def c7_01(x):
    """A region outside the page -- a coordinate-space or rescaling bug."""
    m = x["t"]["out_of_bounds"]
    bad = [i for i, r in enumerate(x["regions"])
           if r["bbox"][0] < -m * x["W"] or r["bbox"][1] < -m * x["H"]
           or r["bbox"][2] > (1 + m) * x["W"] or r["bbox"][3] > (1 + m) * x["H"]]
    if bad:
        return [_f("C7-01", BLOCK,
                   f"{len(bad)} region(s) extend beyond the page",
                   regions=bad, value=len(bad))]


@check("C7-03", ADV, "C7", needs=("psr",))
def c7_03(x):
    """A ruling line detected as a text region."""
    bad = [i for i, r in enumerate(x["regions"])
           if _area(r["bbox"]) > 0
           and max(r["bbox"][2] - r["bbox"][0], r["bbox"][3] - r["bbox"][1])
           / max(min(r["bbox"][2] - r["bbox"][0], r["bbox"][3] - r["bbox"][1]), 1e-6)
           > x["t"]["aspect_ratio"] and not x["region_lines"][i]]
    if bad:
        return [_f("C7-03", ADV,
                   f"{len(bad)} extremely thin empty region(s) -- probably rules "
                   f"detected as content", regions=bad, value=len(bad))]


@check("C7-04", BLOCK, "C7", needs=("psr", "bands"))
def c7_04(x):
    """One region swallowing a multi-column page."""
    if len(x["bands"]) < 3:
        return
    page = x["W"] * x["H"]
    bad = [i for i, r in enumerate(x["regions"])
           if _area(r["bbox"]) > x["t"]["page_dominating"] * page]
    if bad:
        return [_f("C7-04", BLOCK,
                   f"a single region covers the whole of a "
                   f"{len(x['bands'])}-column page", regions=bad)]


# Demoted from BLOCK and superseded.  It compares predicted area against the
# bounding box of the body lines, which on a sparse page -- a title page, a
# section divider -- is mostly the whitespace between a header and a footer, so
# every one of its findings was a correctly covered sparse page.  Measuring
# coverage against glyph area rather than a bounding box is exactly what C1-02
# already does, so this is a worse duplicate rather than a check to repair.
@check("C7-05", ADV, "C7", needs=("psr", "body_lines"))
def c7_05(x):
    """Gross under-detection against the page's own ink extent."""
    if not x["body_lines"]:
        return
    ink = [min(L[0] for L in x["body_lines"]), min(L[1] for L in x["body_lines"]),
           max(L[2] for L in x["body_lines"]), max(L[3] for L in x["body_lines"])]
    cov = sum(_area(r["bbox"]) for r in x["regions"])
    r = cov / max(_area(ink), 1e-6)
    if r < x["t"]["coverage_floor"]:
        return [_f("C7-05", ADV,
                   f"predicted regions cover only {r:.0%} of the page's ink area",
                   value=round(r, 4))]


# ------------------------------------------------------- C8  document -------
#
# These are the only checks that see more than one page.  Each compares a page
# against what the rest of its own document looks like under the same system, so
# a finding means "this page is unlike its neighbours", not "this page is unlike
# what we expected".


@check("C8-01", MAJOR, "C8", needs=("psr", "doc", "body_lines"))
def c8_01(x):
    """A running header or footer the document almost always has is missing.

    Only on a page with a body.  Covers and section title pages carry no running
    furniture by design, and every finding this produced without the guard was
    one of those -- correct about the page and useless about the detector.
    """
    d = x["doc"]
    if d["running_rate"] < x["t"]["running_rate"] or d["n_pages"] < x["t"]["doc_min_pages"]:
        return
    if len(x["body_lines"]) < x["t"]["doc_body_lines"]:
        return
    if not x["facts"]["running"]:
        return [_f("C8-01", MAJOR,
                   f"no running header or footer, on a document that has one on "
                   f"{d['running_rate']:.0%} of its pages",
                   value=round(d["running_rate"], 3))]


# C8-02 and C8-06 read the PDF and never look at the prediction, so they cannot
# be evidence about a detector -- a page can be unlike its document while the
# layout for it is perfect.  They are page-novelty signals: useful as risk
# features, and wrong as findings, so both are advisory.
@check("C8-02", ADV, "C8", needs=("psr", "doc", "bands"))
def c8_02(x):
    """The column count differs from the rest of the document.

    A property of the page, not of the prediction: it says this page is unlike
    its neighbours, which is a reason to look, not a defect that was found.
    """
    d = x["doc"]
    if d["n_pages"] < x["t"]["doc_min_pages"]:
        return
    got = x["facts"]["n_columns"]
    if got != d["columns_mode"]:
        return [_f("C8-02", ADV,
                   f"{got} column(s) where the document normally has "
                   f"{d['columns_mode']}", value=got)]


@check("C8-03", ADV, "C8", needs=("psr", "doc"))
def c8_03(x):
    """The region count is a gross outlier against the document's own spread."""
    d = x["doc"]
    if d["n_pages"] < x["t"]["doc_min_pages"] or d["regions_std"] < 1:
        return
    z = abs(x["facts"]["n_regions"] - d["regions_mean"]) / d["regions_std"]
    if z > x["t"]["region_z"]:
        return [_f("C8-03", ADV,
                   f"{x['facts']['n_regions']} regions against a document mean "
                   f"of {d['regions_mean']:.0f} (z={z:.1f})", value=round(z, 2))]


@check("C8-05", MAJOR, "C8", needs=("psr", "doc"))
def c8_05(x):
    """The class vocabulary collapsed on this page alone."""
    d = x["doc"]
    if d["n_pages"] < x["t"]["doc_min_pages"] or d["classes_median"] < 4:
        return
    if x["facts"]["classes"] <= 1 and x["facts"]["n_regions"] > 3:
        return [_f("C8-05", MAJOR,
                   f"every region on the page has the same class, on a document "
                   f"that normally uses {d['classes_median']:.0f}",
                   value=x["facts"]["classes"])]


@check("C8-06", ADV, "C8", needs=("psr", "doc", "body_lines"))
def c8_06(x):
    """The body font size differs sharply from the rest of the document.

    As with C8-02 this describes the page and not the prediction.  Rendering its
    findings showed appendix dividers and note pages -- all genuinely unlike
    their document, none of them a layout failure.
    """
    d = x["doc"]
    got = x["facts"]["body_font"]
    if d["n_pages"] < x["t"]["doc_min_pages"] or not got or not d["font_mode"]:
        return
    if len(x["body_lines"]) < x["t"]["doc_body_lines"]:
        return
    r = abs(got - d["font_mode"]) / d["font_mode"]
    if r > x["t"]["font_shift"]:
        return [_f("C8-06", ADV,
                   f"body font {got:.0f}px against a document norm of "
                   f"{d['font_mode']:.0f}px", value=round(r, 3))]


# ------------------------------------------------------------- runner -------
def _outside_margins(lines, idx, H, m):
    """Orphans in the top/bottom margin are running furniture the pipeline
    discards anyway, so failing to box them costs nothing.  On a random sample
    they were 22% of all uncovered lines."""
    return [i for i in idx if m * H <= (lines[i][1] + lines[i][3]) / 2 <= (1 - m) * H]


def context(regions, psr, stream, route, t=None, doc=None,
            line_text=None, lm=None):
    t = t or load_thresholds()
    W, H = psr["width"], psr["height"]
    all_lines = psr["text_lines"]
    # Body text, with the lines a filled cluster wrongly swallowed put back.
    # The reference moves everything inside a graphic area out of the body, and
    # 56% of the clusters holding text on this corpus are not figures but shaded
    # tables and tinted prose -- 8% of all glyph lines, on 17% of pages, that
    # the coverage family could not see and so could never charge anyone for.
    # `page_columns` deliberately keeps reading the raw body: a table's numeric
    # columns re-entering the ink profile is the separate gap E4 records.
    body = gfxmod.repaired_body(psr, t["graphic_content_floor"])
    heights = [b[3] - b[1] for b in all_lines] or [10.0]

    owner = stream["line_block"]
    rank_to_region = {b["rank"]: b["region"] for b in stream["blocks"]}
    region_lines = defaultdict(list)
    for li, rk in enumerate(owner):
        if rk is not None:
            region_lines[rank_to_region[rk]].append(li)

    # Matched by value, not identity: `body_text_lines` and `text_lines` are the
    # same objects inside `reference.py` and separate ones after a JSON round
    # trip, so an identity lookup silently resolves every body line to the same
    # index and reports whatever that one line happened to be.
    ix = {tuple(round(v, 2) for v in L): i for i, L in enumerate(all_lines)}
    orphan_body = [k for k, L in enumerate(body)
                   if owner[ix.get(tuple(round(v, 2) for v in L), 0)] is None] if body else []

    ctx = {
        "regions": regions, "psr": psr, "stream": stream, "route": route, "t": t,
        "W": W, "H": H, "all_lines": all_lines, "body_lines": body,
        "line_h": float(np.median(heights)),
        "bands": page_columns(psr),
        "raw_bands": psr.get("column_bands") or [],
        "gutters": psr.get("gutters") or [],
        "page_gutters": column_gutters(psr),
        "grids": psr.get("grid_candidates") or [],
        "table_grids": [q["bbox"] for q in tablemod.qualified(psr)],
        "graphics": figure_areas(psr),
        "raw_graphics": psr.get("graphic_areas") or [],
        "content_units": _standalone_content(psr, t["graphic_content_floor"]),
        "region_bucket": [to_bucket(r.get("class")) for r in regions],
        "region_lines": [region_lines.get(i, []) for i in range(len(regions))],
        "orphan_body": orphan_body,
        "orphan_content": _outside_margins(body, orphan_body, H, t["margin_band"]),
        "doc": doc,
        # The words on each line, aligned to psr["text_lines"], supplied by a
        # caller that has the PDF open.  The PSR stays geometric: putting the
        # text in it would multiply every workspace's size for one check.
        "line_text": line_text,
        "lm": lm,
    }
    # Coverage is scored against reconstructed reading lines, not against the
    # reference's boxes.  On this corpus a "line" from PyMuPDF is often a single
    # glyph -- the median page's boxes are nearly twice as fine as its lines, and
    # a vertical marginal label arrives as one box per character.  Counting those
    # reads a lost label as a lost section.  See validation/lines.py.
    # Whitespace-only lines are not content and a model is right to miss them.
    # They were 51% of everything the coverage family called lost.  The PSR
    # records which they are (`blank_line_idx`, into `text_lines`); matched here
    # by rounded coordinate, the same way body lines are, because the two lists
    # hold separate objects after a JSON round trip.
    blank_boxes = {tuple(round(v, 2) for v in all_lines[i])
                   for i in (psr.get("blank_line_idx") or [])
                   if i < len(all_lines)}
    body_ink = [k for k, L in enumerate(body)
                if tuple(round(v, 2) for v in L) not in blank_boxes]
    ink = set(body_ink)
    ctx["blank_body_lines"] = len(body) - len(ink)

    read_lines, read_owner = linesmod.reading_lines(body)

    def _orphan_lines(lost_boxes):
        lost_boxes = set(lost_boxes)
        covered = {k for i, k in enumerate(read_owner)
                   if k is not None and i not in lost_boxes}
        # A reading line built only from blank boxes is not lost content.
        has_ink = {k for i, k in enumerate(read_owner)
                   if k is not None and i in ink}
        return [k for k in range(len(read_lines))
                if k not in covered and k in has_ink]

    # The same reconstruction over *every* text line, not only body text, and a
    # reading-line-level view of the stream.  The order checks below measure how
    # far apart a structural unit's lines are dragged, and measuring that in
    # glyph boxes reports a table of 17 lines read with 171 lines spliced into
    # it on a page that holds nothing like 171 lines.
    read_all, read_all_owner = linesmod.reading_lines(all_lines)
    seen, seq_lines = set(), []
    for li in stream["sequence"]:
        k = read_all_owner[li] if li < len(read_all_owner) else None
        if k is not None and k not in seen:
            seen.add(k)
            seq_lines.append(k)
    ctx["read_all"] = read_all
    ctx["read_all_pos"] = {k: i for i, k in enumerate(seq_lines)}

    ctx["read_lines"] = read_lines
    # Two views, because the margin filter is not the same question twice.
    # `read_orphans` drops the top and bottom bands, where running furniture
    # lives and where a model is right to emit nothing.  `read_orphans_all`
    # keeps them, because C1-06 exists to look inside the bottom band -- scoring
    # it on the filtered set guarantees it never fires, which is exactly what
    # happened when it shared one view with the rest of the family.
    ctx["read_orphans"] = _orphan_lines(ctx["orphan_content"])
    ctx["read_orphans_all"] = _orphan_lines(ctx["orphan_body"])
    ctx["fragmentation"] = round(len(body) / max(len(read_lines), 1), 3) if body else 1.0
    ctx["facts"] = docmod.page_facts(regions, psr, ctx["bands"], t["margin_band"])
    # The order the PDF itself implies, for C4-07.  Built here rather than in
    # the check so the cost is paid once per page even if more order checks
    # arrive later.
    try:
        ref_regions, meta = psrlayout.build(psr)
        ctx["reference_stream"] = (
            assemblemod.assemble(ref_regions, psr, route.get("direction", "ltr"))
            if meta["confidence"] == "usable" else None)
    except Exception:
        ctx["reference_stream"] = None
    return ctx


def available(x):
    """Which inputs this page has, and which are absent because the page simply
    has no such structure.

    The distinction matters more than it looks.  A column check that cannot run
    because the page is single-column is *inapplicable* and says nothing; one
    that cannot run because the column structure could not be determined is
    *unverifiable* and must escalate.  Collapsing the two either escalates a
    third of an ordinary corpus or silently accepts the pages hardest to read.
    """
    have, na = set(), set()
    if x["route"]["psr_trust"] != "unusable":
        have.add("psr")
    if x["body_lines"]:
        have.add("body_lines")
    if len(x["bands"]) >= 1:
        have.add("bands")
    elif (x["psr"].get("n_columns_est") or 1) <= 1:
        na.add("bands")          # a genuinely single-column page
    if x["page_gutters"]:
        have.add("gutters")
    elif len(x["bands"]) <= 1 or x["gutters"]:
        na.add("gutters")        # single column, or every corridor is inside a table
    if x["table_grids"]:
        have.add("grids")
    else:
        na.add("grids")          # the page has no qualified table to miss
    if x["graphics"]:
        have.add("graphics")
    else:
        na.add("graphics")
    if x["stream"]:
        have.add("stream")
    # An order check needs both a reference and an order the model actually
    # produced.  A derived order is the pipeline's own work, not the model's,
    # and checking it against another derivation says nothing.
    if x.get("doc"):
        have.add("doc")
    else:
        na.add("doc")            # a single page cannot be unlike its document
    if x.get("line_text") and x.get("lm"):
        have.add("words")
    else:
        na.add("words")          # no caller supplied the text, or no model
    if x.get("reference_stream") and x["stream"]["order_source"] == "model":
        have.add("reference_stream")
    else:
        na.add("reference_stream")
    return have, na


def run(regions, psr, stream, route, t=None, doc=None,
        line_text=None, lm=None):
    x = context(regions, psr, stream, route, t, doc, line_text, lm)
    have, na = available(x)
    findings, skipped, inapplicable = [], [], []
    for c in REGISTRY:
        missing = [n for n in c["needs"] if n not in have]
        if missing:
            row = {"id": c["id"], "severity": c["severity"],
                   "reason": "no " + ", ".join(missing)}
            (inapplicable if all(n in na for n in missing) else skipped).append(row)
            continue
        try:
            out = c["fn"](x)
        except Exception as e:                       # a broken check must not
            skipped.append({"id": c["id"], "severity": c["severity"],  # pass a page
                            "reason": f"{type(e).__name__}: {e}"})
            continue
        if out:
            findings.extend(out)
    # C1-01..04 are four views of the same missed content and C3-01/02 two views
    # of the same column merge.  A reviewer gets one task per defect, not one per
    # check, so only the strongest finding in each group survives into the report.
    GROUPS = (("C1-01", "C1-02", "C1-03", "C1-04"), ("C3-01", "C3-02"))
    for g in GROUPS:
        hits = [f for f in findings if f["id"] in g]
        if len(hits) > 1:
            keep = max(hits, key=lambda f: g.index(f["id"]))
            keep["also"] = [f["id"] for f in hits if f is not keep]
            findings = [f for f in findings if f not in hits or f is keep]

    blocking = [f for f in findings if f["severity"] == BLOCK]
    # An unavailable BLOCK check is a reason to escalate, never to accept.
    unverifiable = [s for s in skipped if s["severity"] == BLOCK]
    return {"findings": findings, "skipped": skipped,
            "inapplicable": inapplicable,
            "n_block": len(blocking), "n_major": sum(1 for f in findings
                                                     if f["severity"] == MAJOR),
            "unverifiable": len(unverifiable),
            "families": sorted({f["id"].split("-")[0] for f in findings})}
