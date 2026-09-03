#!/usr/bin/env python3
"""Reconstruct reading lines from the reference's text boxes.

The PSR takes its text lines from PyMuPDF's own line grouping, which on a large
share of this corpus is one box per glyph: the Arabic PDFs here place characters
with individual positioning operators, and PyMuPDF reports each as a line.  On
the random sample 44% of all body lines are taller than wide, which no
horizontal text line can be, and one page carries 58 "lines" holding 225
characters between them -- 3.9 characters each.

That breaks every count-based coverage threshold.  "A block of 36 consecutive
lines was missed" reads as a lost section and is in fact 36 glyphs of a vertical
label in the margin, which every layout model correctly ignores.  It was the
single largest driver of escalation on the random sample.

So coverage is scored against lines rebuilt here rather than against the boxes
as they arrive.  The PSR itself is left alone: `core.metrics` scores recall
against those fields and they must not move.  This is a validation-layer view of
the same data.

Merging is deliberately timid.  Boxes join only when they share a row and sit
within about one character of each other, so a fragmented word becomes one line
while two cells of a table row, or two columns of a page, stay apart -- a merge
across a column would let a model that captured one column look as though it had
missed nothing.

A second pass handles text set vertically.  A rotated marginal label arrives as
a column of single glyphs, one per row, which the horizontal pass cannot touch:
on the page that produced the worst finding on the sample it was 36 boxes, and
the check reported a lost section.  Boxes that are taller than wide, stacked in
a narrow column with small gaps, are one line read downwards.
"""
import numpy as np

MAX_GAP = 0.6        # of the median box height: about one character
ROW_OVERLAP = 0.5    # of the shorter box: same row


def _median_h(boxes):
    hs = [b[3] - b[1] for b in boxes if b[3] > b[1]]
    return float(np.median(hs)) if hs else 10.0


def _merge_vertical(lines, members, owner, h):
    """Second pass: columns of tall-thin boxes are one vertical line."""
    gap = h * 1.2
    changed = True
    while changed:
        changed = False
        for a in range(len(lines)):
            if lines[a] is None:
                continue
            wa, ha = lines[a][2] - lines[a][0], lines[a][3] - lines[a][1]
            if wa >= ha:                       # only tall-thin runs stack
                continue
            for b in range(a + 1, len(lines)):
                if lines[b] is None:
                    continue
                wb, hb = lines[b][2] - lines[b][0], lines[b][3] - lines[b][1]
                if wb >= hb:
                    continue
                # same narrow column, and vertically within a character
                ov = (min(lines[a][2], lines[b][2])
                      - max(lines[a][0], lines[b][0]))
                if ov <= 0.5 * min(wa, wb):
                    continue
                dy = max(lines[a][1], lines[b][1]) - min(lines[a][3], lines[b][3])
                if dy > gap:
                    continue
                lines[a] = [min(lines[a][0], lines[b][0]), min(lines[a][1], lines[b][1]),
                            max(lines[a][2], lines[b][2]), max(lines[a][3], lines[b][3])]
                members[a] += members[b]
                lines[b] = None
                changed = True
    keep = [k for k, L in enumerate(lines) if L is not None]
    remap = {k: i for i, k in enumerate(keep)}
    out = [lines[k] for k in keep]
    for k in keep:
        for i in members[k]:
            owner[i] = remap[k]
    return out, owner


def reading_lines(boxes):
    """-> (lines, owner) where owner[i] is the line index of input box i.

    Order within a line follows the page, not the writing direction: a line is
    a geometric union here, and nothing downstream reads its text.
    """
    if not boxes:
        return [], []
    h = _median_h(boxes)
    gap = h * MAX_GAP
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
    lines, owner = [], [None] * len(boxes)
    members = []
    for i in order:
        b = boxes[i]
        bh = b[3] - b[1]
        placed = False
        for k, L in enumerate(lines):
            lh = L[3] - L[1]
            ov = min(L[3], b[3]) - max(L[1], b[1])
            if ov <= ROW_OVERLAP * max(min(lh, bh), 1e-6):
                continue
            # horizontal gap to the line as it stands, either side
            if b[0] > L[2] + gap or b[2] < L[0] - gap:
                continue
            lines[k] = [min(L[0], b[0]), min(L[1], b[1]),
                        max(L[2], b[2]), max(L[3], b[3])]
            members[k].append(i)
            owner[i] = k
            placed = True
            break
        if not placed:
            lines.append(list(b))
            members.append([i])
            owner[i] = len(lines) - 1
    return _merge_vertical(lines, members, owner, h)


def fragmentation(boxes):
    """How much finer the reference's boxes are than the lines they belong to.

    1.0 means the boxes already are lines.  Higher means the page is fragmented,
    and any threshold counting boxes is reading that page in a different unit
    from the rest of the corpus.
    """
    if not boxes:
        return 1.0
    lines, _ = reading_lines(boxes)
    return round(len(boxes) / max(len(lines), 1), 3)
