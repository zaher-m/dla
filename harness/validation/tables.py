#!/usr/bin/env python3
"""Does this ruled box actually hold a table?

`grid_candidates` in the PSR is a cluster of ruling strokes filtered by size.
That admits a great deal that is not a table: a framed callout, a signature box,
a pair of rules under a heading, a banner.  Measured on a random sample, half
the grid candidates a strong detector "missed" were of that kind, and a check
built on the raw candidates reported a table failure on 30% of all pages.

The qualifier is deliberately built from the PDF alone.  Fitting it against
which boxes a model called a table would teach it to agree with that model,
which is precisely the circular validation the framework exists to avoid.

A box is table-like when it shows *repeated row structure* and *repeated column
structure* and holds enough content to be worth storing:

  rows     three or more distinct horizontal rule positions, or three or more
           rows of text
  columns  two or more distinct vertical rule positions, or two or more text
           rows that each hold several cells

The column test has two arms because financial tables are very often ruled
horizontally only -- on this corpus a quarter of the boxes a detector agreed
were tables carry no vertical rule at all, so demanding one would reject real
tables.

Charts are the hard case.  A bar or line chart has axis rules that read as row
structure and axis tick labels that read as cells, so it satisfies every
structural test a table does; rendering the boxes that survived an earlier
version of this qualifier showed five of six were charts.

Two discriminators were tried.  Leaning on the PSR's own attribution -- a
chart's labels become `graphic_text_lines` -- fails, because a table drawn with
coloured header rows and banded cell fills is itself collected as a graphic
area, so all of its cell text is reattributed away and the table disappears.
Three confirmed tables had 97-100% of their lines counted as chart labelling
that way.

What does separate them is *text density inside the ruled box*, and it is a band
rather than a floor.  A chart spends its area on ink and puts a thin ring of
labels round the edge; a table leaves whitespace between short numeric cells;
a framed paragraph of prose fills the box solidly.  On visually confirmed
examples: charts cover 3-8% of the box with glyphs, tables 21-33%, and bordered
prose 36-63%.  Both ends matter -- an earlier version used a floor only and its
surviving findings turned out to be framed paragraphs and whole-page borders.

A caution for whoever tunes this next.  The band is fitted on 21 hand-checked
boxes, which is enough to reject the failure modes actually seen and not enough
to call it validated.  It is the first thing to re-fit against annotated pages.
And it is calibrated on *grid candidates* -- a ruled area plus its padding.  It
does not transfer to a model's own region box, which is drawn tight around the
content: carried across, it rejected five unmistakable financial tables.

A related defect is still open upstream, in `core.reference`.  A table drawn
with coloured header rows and banded cell fills is a cluster of filled vector
drawings, so it is collected as a `graphic_area` and all of its cell text is
reattributed to `graphic_text_lines` -- out of `body_text_lines`, which is what
`core.metrics` scores text recall against.  Measured across three workspaces the
effect reaches 6-8% of all glyph lines, on 9-34% of pages.

Two signals were measured against visually confirmed examples and neither
separates a shaded table from a chart:

  text density   shaded tables cover 20-31% of the box with glyphs; benchmark
                 charts cover a median of 28%.  The distributions coincide.  A
                 0.16 threshold fitted on one corpus deleted genuine charts from
                 eight pages of the other.
  vector shape   both are drawn from rectangles and straight lines with one to
                 four fill colours.  The best combined rule caught six of seven
                 tables and wrongly caught eleven of thirty-four charts.

The distinction is semantic rather than geometric, so the fix needs labelled
examples and not another threshold.  Until then the PSR is left alone -- it is
shared with the benchmark metrics -- and validation compensates locally in
`checks.figure_areas`.
"""
POS_TOL = 6.0        # px: strokes closer than this are one rule drawn twice
ROW_OVERLAP = 0.5


def _inter(a, b):
    return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))


def _inside(box, holder, frac=0.6):
    a = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    return _inter(box, holder) / a >= frac


def _distinct(vals, tol=POS_TOL):
    return len({round(v / tol) for v in vals})


def evidence(grid, psr):
    """Everything the PDF says about one ruled box."""
    h = [r for r in (psr.get("rules_h") or []) if _inside(r, grid, 0.5)]
    v = [r for r in (psr.get("rules_v") or []) if _inside(r, grid, 0.5)]
    lines = [L for L in psr["text_lines"] if _inside(L, grid)]
    gfx = {tuple(x) for x in (psr.get("graphic_text_lines") or [])}
    n_gfx = sum(1 for L in lines if tuple(L) in gfx)
    ga = max((grid[2] - grid[0]) * (grid[3] - grid[1]), 1e-6)
    ta = sum((L[2] - L[0]) * (L[3] - L[1]) for L in lines)

    rows = []
    for L in sorted(lines, key=lambda b: b[1]):
        placed = False
        for r in rows:
            ref = r[0]
            ov = max(0.0, min(L[3], ref[3]) - max(L[1], ref[1]))
            if ov / max(min(L[3] - L[1], ref[3] - ref[1]), 1e-6) >= ROW_OVERLAP:
                r.append(L); placed = True; break
        if not placed:
            rows.append([L])

    return {
        "h_positions": _distinct([r[1] for r in h]),
        "v_positions": _distinct([r[0] for r in v]),
        "n_lines": len(lines),
        "text_frac": round(ta / ga, 3),
        # kept as a diagnostic only: it cannot be used to reject, because a
        # shaded table is itself collected as a graphic area
        "graphic_line_share": round(n_gfx / max(len(lines), 1), 3),
        "n_rows": len(rows),
        "multi_cell_rows": sum(1 for r in rows if len(r) >= 2),
    }


DEFAULTS = {"min_lines": 6, "min_row_rules": 3, "min_rows": 3,
            "min_col_rules": 2, "min_multi_cell_rows": 2,
            "min_text_frac": 0.18, "max_text_frac": 0.35}


def is_table_like(ev, t=None):
    t = {**DEFAULTS, **(t or {})}
    if ev["n_lines"] < t["min_lines"]:
        return False, "too little content to be a table"
    if ev["text_frac"] < t["min_text_frac"]:
        return False, (f"a chart: text covers only {ev['text_frac']:.0%} of the "
                       f"box, a table covers a fifth or more")
    if ev["text_frac"] > t["max_text_frac"]:
        return False, (f"framed prose: text covers {ev['text_frac']:.0%} of the "
                       f"box, denser than a table's columns and gaps")
    rows_ok = (ev["h_positions"] >= t["min_row_rules"]
               or ev["n_rows"] >= t["min_rows"])
    cols_ok = (ev["v_positions"] >= t["min_col_rules"]
               or ev["multi_cell_rows"] >= t["min_multi_cell_rows"])
    if not rows_ok:
        return False, "no repeated row structure"
    if not cols_ok:
        return False, "no repeated column structure"
    return True, "ruled box, several rules, and text-dense like a table"


def qualified(psr, t=None):
    """The grid candidates that really do look like tables."""
    out = []
    for g in (psr.get("grid_candidates") or []):
        ev = evidence(g, psr)
        ok, why = is_table_like(ev, t)
        if ok:
            out.append({"bbox": g, "evidence": ev})
    return out
