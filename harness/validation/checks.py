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
    "orphan_line_rate": 0.12,
    "orphan_area_rate": 0.12,
    "orphan_cluster": 6,
    "margin_band": 0.08,
    "orphan_in_column_rate": 0.12,
    "orphan_column_min_lines": 20,
    "footnote_lines": 4,
    "graphic_miss_count": 2,
    "overlap_regions": 6,
    "footnote_band": 0.12,
    "graphic_miss_area": 0.02,
    "line_cut_rate": 0.15,
    "gutter_cover": 0.6,
    "band_span": 0.2,
    "band_imbalance": 0.85,
    "double_emit_rate": 0.02,
    "overlap_frac": 0.50,
    "grid_cover": 0.5,
    "media_text_lines": 5,
    "media_graphic_frac": 0.15,
    "discard_body_lines": 3,
    "discard_line_rate": 0.15,
    "out_of_bounds": 0.02,
    "page_dominating": 0.85,
    "coverage_floor": 0.40,
    "excess_band_transitions": 2,
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


def _f(cid, sev, msg, regions=(), value=None):
    return {"id": cid, "severity": sev, "message": msg,
            "regions": sorted(regions), "value": value}


# ------------------------------------------------------- C1  coverage -------
@check("C1-01", BLOCK, "C1", needs=("psr", "body_lines"))
def c1_01(x):
    """Body glyph lines that no region covers."""
    r = len(x["orphan_content"]) / max(len(x["body_lines"]), 1)
    if r > x["t"]["orphan_line_rate"]:
        return [_f("C1-01", BLOCK,
                   f"content lost: {r:.1%} of body text lines sit in no region",
                   value=round(r, 4))]


@check("C1-02", BLOCK, "C1", needs=("psr", "body_lines"))
def c1_02(x):
    """Glyph ink area no region covers -- catches one big miss C1-01 dilutes."""
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
    if not x["orphan_content"]:
        return
    lines = x["body_lines"]
    orph = sorted(x["orphan_content"], key=lambda i: lines[i][1])
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
    inband = [i for i, L in enumerate(x["body_lines"])
              if _band_of(L, x["bands"], x["W"]) is not None]
    if len(inband) < x["t"]["orphan_column_min_lines"]:
        return
    orph = set(x["orphan_content"])
    n = sum(1 for i in inband if i in orph)
    r = n / len(inband)
    if r > x["t"]["orphan_in_column_rate"]:
        return [_f("C1-04", BLOCK,
                   f"{r:.0%} of the text inside real columns ({n} lines) was "
                   f"missed", value=round(r, 4))]


@check("C1-05", BLOCK, "C1", needs=("psr", "body_lines"))
def c1_05(x):
    """No text region at all on a page that plainly has text."""
    if len(x["body_lines"]) > 20 and not any(
            b == TEXT for b in x["region_bucket"]):
        return [_f("C1-05", BLOCK,
                   f"no text region on a page carrying {len(x['body_lines'])} text lines")]


@check("C1-06", MAJOR, "C1", needs=("psr", "body_lines"))
def c1_06(x):
    """Uncovered lines in the footnote band at the foot of the page."""
    cut = x["H"] * (1 - x["t"]["footnote_band"])
    n = sum(1 for i in x["orphan_body"] if x["body_lines"][i][1] >= cut)
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
    """Region boundaries cutting through glyph lines."""
    bad = defaultdict(int)
    for L in x["body_lines"]:
        for i, r in enumerate(x["regions"]):
            c = _cover(L, r["bbox"])
            if 0.12 < c < 0.88:
                bad[i] += 1
    n = sum(bad.values())
    rate = n / max(len(x["body_lines"]), 1)
    if rate > x["t"]["line_cut_rate"]:
        return [_f("C2-01", MAJOR,
                   f"{rate:.0%} of text lines are cut by a region boundary",
                   regions=list(bad), value=round(rate, 4))]


@check("C2-06", MAJOR, "C2", needs=("psr",))
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
        if not any(_cover(L, r["bbox"]) >= 0.5 for L in x["all_lines"]):
            empty.append(i)
    if empty:
        return [_f("C2-06", MAJOR,
                   f"{len(empty)} text region(s) contain no text",
                   regions=empty, value=len(empty))]


# ------------------------------------------------------- C3  columns --------
@check("C3-01", BLOCK, "C3", needs=("psr", "gutters"))
def c3_01(x):
    """A text region straddling a real whitespace corridor between columns."""
    bad = []
    for i, (r, b) in enumerate(zip(x["regions"], x["region_bucket"])):
        if b != TEXT:
            continue
        for g in x["gutters"]:
            w = g[2] - g[0]
            if w > 0 and (min(r["bbox"][2], g[2]) - max(r["bbox"][0], g[0])) \
                    > w * x["t"]["gutter_cover"]:
                bad.append(i); break
    if bad:
        return [_f("C3-01", BLOCK,
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
    bad = 0
    per = defaultdict(list)
    for blk in x["stream"]["blocks"]:
        b = _band_of(blk["bbox"], x["bands"], x["W"])
        if b is not None:
            per[b].append((blk["rank"], blk["bbox"][1]))
    for b, items in per.items():
        items.sort()
        ys = [y for _, y in items]
        bad += sum(1 for a, c in zip(ys, ys[1:]) if c < a - x["line_h"])
    if bad:
        return [_f("C4-02", BLOCK,
                   f"{bad} block(s) are read out of top-to-bottom order within "
                   f"their own column", value=bad)]


# ------------------------------------------------------- C5  duplication ----
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
            if small > 0 and _inter(a, b) / small > x["t"]["overlap_frac"]:
                bad.add(i); bad.add(j)
    if len(bad) >= x["t"]["overlap_regions"]:
        return [_f("C5-01", MAJOR,
                   f"{len(bad)} regions of the same kind overlap each other",
                   regions=bad, value=len(bad))]


@check("C5-03", MAJOR, "C5", needs=("psr",))
def c5_03(x):
    """A glyph line covered by two text regions is stored twice."""
    dup = 0
    seen = set()
    for li, L in enumerate(x["all_lines"]):
        owners = [i for i, (r, b) in enumerate(zip(x["regions"], x["region_bucket"]))
                  if b == TEXT and _cover(L, r["bbox"]) >= 0.5]
        if len(owners) > 1:
            dup += 1
            seen.update(owners)
    rate = dup / max(len(x["all_lines"]), 1)
    if rate > x["t"]["double_emit_rate"]:
        return [_f("C5-03", MAJOR,
                   f"{rate:.1%} of text lines fall inside two text regions and "
                   f"will be written to the index twice",
                   regions=seen, value=round(rate, 4))]


# ------------------------------------------------------- C6  buckets --------
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


@check("C6-03", BLOCK, "C6", needs=("psr",))
def c6_03(x):
    """A figure region that is really text -- content sent to object storage."""
    bad = []
    for i, (r, b) in enumerate(zip(x["regions"], x["region_bucket"])):
        if b != MEDIA:
            continue
        gfrac = max((_cover(r["bbox"], g) for g in x["graphics"]), default=0.0)
        if len(x["region_lines"][i]) > x["t"]["media_text_lines"] \
                and gfrac < x["t"]["media_graphic_frac"]:
            bad.append(i)
    if bad:
        return [_f("C6-03", BLOCK,
                   f"{len(bad)} figure region(s) hold text and no graphic: that "
                   f"text leaves the index entirely", regions=bad, value=len(bad))]


@check("C6-05", BLOCK, "C6", needs=("psr", "bands"))
def c6_05(x):
    """Body content classified as running furniture and dropped."""
    bad = []
    for i, (r, b) in enumerate(zip(x["regions"], x["region_bucket"])):
        if b != DISCARD or len(x["region_lines"][i]) <= x["t"]["discard_body_lines"]:
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


@check("C7-05", BLOCK, "C7", needs=("psr", "body_lines"))
def c7_05(x):
    """Gross under-detection against the page's own ink extent."""
    if not x["body_lines"]:
        return
    ink = [min(L[0] for L in x["body_lines"]), min(L[1] for L in x["body_lines"]),
           max(L[2] for L in x["body_lines"]), max(L[3] for L in x["body_lines"])]
    cov = sum(_area(r["bbox"]) for r in x["regions"])
    r = cov / max(_area(ink), 1e-6)
    if r < x["t"]["coverage_floor"]:
        return [_f("C7-05", BLOCK,
                   f"predicted regions cover only {r:.0%} of the page's ink area",
                   value=round(r, 4))]


# ------------------------------------------------------------- runner -------
def _outside_margins(lines, idx, H, m):
    """Orphans in the top/bottom margin are running furniture the pipeline
    discards anyway, so failing to box them costs nothing.  On a random sample
    they were 22% of all uncovered lines."""
    return [i for i in idx if m * H <= (lines[i][1] + lines[i][3]) / 2 <= (1 - m) * H]


def context(regions, psr, stream, route, t=None):
    t = t or load_thresholds()
    W, H = psr["width"], psr["height"]
    all_lines = psr["text_lines"]
    body = psr.get("body_text_lines") or []
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

    return {
        "regions": regions, "psr": psr, "stream": stream, "route": route, "t": t,
        "W": W, "H": H, "all_lines": all_lines, "body_lines": body,
        "line_h": float(np.median(heights)),
        "bands": psr.get("column_bands") or [],
        "gutters": psr.get("gutters") or [],
        "grids": psr.get("grid_candidates") or [],
        "table_grids": [q["bbox"] for q in tablemod.qualified(psr)],
        "graphics": psr.get("graphic_areas") or [],
        "region_bucket": [to_bucket(r.get("class")) for r in regions],
        "region_lines": [region_lines.get(i, []) for i in range(len(regions))],
        "orphan_body": orphan_body,
        "orphan_content": _outside_margins(body, orphan_body, H, t["margin_band"]),
    }


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
    if x["gutters"]:
        have.add("gutters")
    elif len(x["bands"]) <= 1:
        na.add("gutters")        # nothing to run between
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
    return have, na


def run(regions, psr, stream, route, t=None):
    x = context(regions, psr, stream, route, t)
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
