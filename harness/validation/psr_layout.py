#!/usr/bin/env python3
"""A reference layout built from the PDF alone.

This is what makes phase 0 measurable before a single page has been annotated:
the content stream already says where the glyphs, the ruled grids and the
graphics are, so a layout can be compared against the page's own structure
rather than against a human's.

The grouping is row-first, which matters on this corpus.  A naive
sort-by-y-then-merge-if-close builds paragraphs, and on a financial statement
consecutive lines are not stacked at all -- they are cells sitting side by side,
with *negative* horizontal overlap and negative vertical gaps.  Measured on a
dense two-column page, prose-style grouping produced 181 regions for 273 lines
and made every model look like it had merged everything.  So lines are first
gathered into rows by vertical overlap, then runs of rows are classified:
multi-cell runs with stable column positions are one unruled table, single-cell
runs merge into paragraphs.

Its limits are still worth stating.  It speaks to *is this one block or two* and
*what order do these blocks come in*.  It cannot say whether a block is a
caption or a heading, and `meta["confidence"]` reports how much of the page it
had to guess at, so a comparison can decline to score grouping where the
reference is weak.
"""
import numpy as np

GAP = 1.7           # merge single-cell rows whose vertical gap is under this many line heights
ROW_OVERLAP = 0.5   # lines share a row when their y-spans overlap this much of the shorter
MIN_OVERLAP = 0.1   # ...and paragraphs need this much horizontal overlap to continue
COL_TOL = 0.02      # column positions agree within this fraction of page width


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _h_overlap(a, b):
    ov = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    return ov / max(min(a[2] - a[0], b[2] - b[0]), 1e-6)


def _v_overlap(a, b):
    ov = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ov / max(min(a[3] - a[1], b[3] - b[1]), 1e-6)


def _rows(lines):
    """Lines sharing a horizontal band, in reading-independent order."""
    idx = sorted(range(len(lines)), key=lambda i: (lines[i][1], lines[i][0]))
    rows, cur = [], [idx[0]] if idx else []
    for k in idx[1:]:
        if any(_v_overlap(lines[k], lines[j]) >= ROW_OVERLAP for j in cur):
            cur.append(k)
        else:
            rows.append(cur); cur = [k]
    if cur:
        rows.append(cur)
    return rows


def _same_columns(r1, r2, lines, W):
    """Two rows share a column structure when their cell left edges line up."""
    a = sorted(round(lines[i][0] / W, 3) for i in r1)
    b = sorted(round(lines[i][0] / W, 3) for i in r2)
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= COL_TOL for x, y in zip(a, b))


def build(psr):
    """PSR -> (regions in the normalized layout schema, meta)."""
    W = psr["width"]
    bands = psr.get("column_bands") or []
    body = psr.get("body_text_lines") or []
    heights = [b[3] - b[1] for b in psr["text_lines"]] or [10.0]
    lh = float(np.median(heights))

    regions = []

    grids = psr.get("grid_candidates") or []
    tbl_lines = psr.get("table_text_lines") or []
    for g in grids:
        held = [L for L in tbl_lines
                if _h_overlap(L, g) > 0.5 and _v_overlap(L, g) > 0.5]
        regions.append({"bbox": _union(held + [g]), "class": "table",
                        "reading_order": None, "source": "psr.grid"})

    for g in (psr.get("graphic_areas") or []):
        regions.append({"bbox": list(g), "class": "figure",
                        "reading_order": None, "source": "psr.graphic"})

    # Body text, kept inside a single column band so two columns never merge
    # merely because their lines are vertically adjacent.
    if bands:
        groups = {i: [] for i in range(len(bands))}
        groups[None] = []
        for L in body:
            hits = [i for i, b in enumerate(bands)
                    if min(L[2], b[1]) - max(L[0], b[0]) > (L[2] - L[0]) * 0.5]
            groups[hits[0] if len(hits) == 1 else None].append(L)
    else:
        groups = {None: list(body)}

    n_tabular = 0
    for key, lines in groups.items():
        if not lines:
            continue
        rows = _rows(lines)
        i = 0
        while i < len(rows):
            if len(rows[i]) >= 2:
                # a run of rows with matching column structure is one table
                j = i + 1
                while j < len(rows) and len(rows[j]) >= 2 and \
                        _same_columns(rows[j - 1], rows[j], lines, W):
                    j += 1
                members = [lines[k] for r in rows[i:j] for k in r]
                cls = "table" if (j - i) >= 2 else "text"
                n_tabular += len(members) if cls == "table" else 0
                regions.append({"bbox": _union(members), "class": cls,
                                "reading_order": None, "source": "psr.rows"})
                i = j
                continue
            # single-cell rows: merge downwards into a paragraph
            j = i + 1
            while j < len(rows) and len(rows[j]) == 1:
                prev, nxt = lines[rows[j - 1][0]], lines[rows[j][0]]
                if nxt[1] - prev[3] > GAP * lh or _h_overlap(nxt, prev) < MIN_OVERLAP:
                    break
                j += 1
            members = [lines[k] for r in rows[i:j] for k in r]
            regions.append({"bbox": _union(members), "class": "text",
                            "reading_order": None, "source": "psr.rows"})
            i = j

    meta = {"n_lines": len(psr["text_lines"]), "n_body": len(body),
            "n_regions": len(regions), "tabular_lines": n_tabular,
            "tabular_frac": round(n_tabular / len(body), 4) if body else 0.0,
            "lines_per_region": round(len(body) / max(
                sum(1 for r in regions if r["source"] == "psr.rows"), 1), 2)}
    # A reference that splits nearly every line into its own region cannot be
    # used to score grouping; the comparison checks this before reporting it.
    # A page whose text all sits inside ruled grids has no body lines to group
    # and is not weak -- it is answered, so it keeps full confidence.
    if not body:
        meta["confidence"] = "usable" if regions else "low"
    else:
        meta["confidence"] = "low" if meta["lines_per_region"] < 1.6 else "usable"
    return regions, meta
