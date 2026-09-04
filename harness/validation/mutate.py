#!/usr/bin/env python3
"""Inject known layout defects and measure which checks catch them.

The checks were each verified by rendering their findings and looking, which
answers "is what this check reports real". It does not answer the complementary
question -- what happens on a page that IS broken and no check fires -- and that
question has no natural source of examples, because a corpus does not come
labelled with its failures.

So the failures are manufactured. Start from a layout believed to be roughly
right, apply one defect of one intensity, and see whether anything blocks. Every
mutation here is a failure mode observed in real detector output on this corpus,
not an arbitrary perturbation.

Detection is reported as a *marginal* rate. The baseline layout already trips
some checks, so the number that means anything is how much more often a check
fires once the defect is present.
"""
import copy
import random


def _boxes(regions):
    return [r["bbox"] for r in regions]


def drop_region(regions, rng, frac=0.25):
    """A detector misses blocks. The most consequential error there is."""
    keep = [r for r in regions if rng.random() > frac]
    return keep


def drop_largest(regions, rng, n=1):
    """Misses the *biggest* block, which a random drop mostly misses."""
    if len(regions) <= n:
        return []
    order = sorted(range(len(regions)),
                   key=lambda i: -((regions[i]["bbox"][2] - regions[i]["bbox"][0])
                                   * (regions[i]["bbox"][3] - regions[i]["bbox"][1])))
    gone = set(order[:n])
    return [r for i, r in enumerate(regions) if i not in gone]


def merge_horizontal(regions, rng, frac=0.5):
    """Two side-by-side regions become one box: the column merge.

    The failure that motivated this module. Every glyph stays inside a region,
    so coverage is satisfied, and the lines are then read across the gutter.
    """
    out, used = [], set()
    pairs = []
    for i, a in enumerate(regions):
        for j, b in enumerate(regions):
            if j <= i or i in used or j in used:
                continue
            ay, by = a["bbox"], b["bbox"]
            h = min(ay[3] - ay[1], by[3] - by[1])
            ov = min(ay[3], by[3]) - max(ay[1], by[1])
            if h > 0 and ov > 0.5 * h and abs(ay[0] - by[0]) > 0.2 * (ay[2] - ay[0] + 1):
                pairs.append((i, j))
                used.add(i); used.add(j)
    take = [p for p in pairs if rng.random() < frac]
    merged = {i for p in take for i in p}
    for i, j in take:
        a, b = regions[i]["bbox"], regions[j]["bbox"]
        out.append({**regions[i], "bbox": [min(a[0], b[0]), min(a[1], b[1]),
                                           max(a[2], b[2]), max(a[3], b[3])]})
    out += [r for i, r in enumerate(regions) if i not in merged]
    return out


def merge_vertical(regions, rng, frac=0.5):
    """Stacked regions merged: a heading absorbed into the paragraph below."""
    out, used = [], set()
    order = sorted(range(len(regions)), key=lambda i: regions[i]["bbox"][1])
    for k in range(len(order) - 1):
        i, j = order[k], order[k + 1]
        if i in used or j in used or rng.random() >= frac:
            continue
        a, b = regions[i]["bbox"], regions[j]["bbox"]
        if min(a[2], b[2]) - max(a[0], b[0]) < 0.5 * min(a[2] - a[0], b[2] - b[0]):
            continue
        out.append({**regions[i], "bbox": [min(a[0], b[0]), min(a[1], b[1]),
                                           max(a[2], b[2]), max(a[3], b[3])]})
        used.add(i); used.add(j)
    out += [r for i, r in enumerate(regions) if i not in used]
    return out


def split_region(regions, rng, frac=0.5):
    """One block reported as two: over-segmentation."""
    out = []
    for r in regions:
        b = r["bbox"]
        if rng.random() < frac and b[3] - b[1] > 40:
            mid = (b[1] + b[3]) / 2
            out.append({**r, "bbox": [b[0], b[1], b[2], mid]})
            out.append({**r, "bbox": [b[0], mid, b[2], b[3]]})
        else:
            out.append(r)
    return out


def shrink(regions, rng, frac=0.15):
    """Boxes tightened until lines fall outside them: boundary cuts."""
    out = []
    for r in regions:
        b = r["bbox"]
        dx, dy = (b[2] - b[0]) * frac / 2, (b[3] - b[1]) * frac / 2
        out.append({**r, "bbox": [b[0] + dx, b[1] + dy, b[2] - dx, b[3] - dy]})
    return out


def grow(regions, rng, frac=0.15):
    """Boxes inflated until they swallow their neighbours."""
    out = []
    for r in regions:
        b = r["bbox"]
        dx, dy = (b[2] - b[0]) * frac / 2, (b[3] - b[1]) * frac / 2
        out.append({**r, "bbox": [b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy]})
    return out


def duplicate(regions, rng, frac=0.3):
    """The same block emitted twice: written to its store twice."""
    return regions + [copy.deepcopy(r) for r in regions if rng.random() < frac]


def _reclass(to):
    def f(regions, rng, frac=0.3):
        return [({**r, "class": to} if rng.random() < frac else r) for r in regions]
    f.__name__ = f"class_to_{to}"
    return f


def shuffle_order(regions, rng, frac=1.0):
    """A supplied reading order permuted. Inert where none is supplied."""
    out = copy.deepcopy(regions)
    idx = list(range(len(out)))
    rng.shuffle(idx)
    for pos, i in enumerate(idx):
        out[i]["reading_order"] = pos
    return out


def shift(regions, rng, frac=0.05):
    """Every box offset: a coordinate-space or rescaling bug."""
    return [{**r, "bbox": [r["bbox"][0] + frac * 2000, r["bbox"][1] + frac * 2000,
                           r["bbox"][2] + frac * 2000, r["bbox"][3] + frac * 2000]}
            for r in regions]


# name -> (function, intensities).  Intensities are shares or counts, chosen so
# the weakest is a defect a reviewer would still call one.
MUTATIONS = [
    ("drop_region",     drop_region,        (0.10, 0.25, 0.50, 1.00)),
    ("drop_largest",    drop_largest,       (1, 2, 3)),
    ("merge_horizontal", merge_horizontal,  (0.50, 1.00)),
    ("merge_vertical",  merge_vertical,     (0.50, 1.00)),
    ("split_region",    split_region,       (0.50, 1.00)),
    ("shrink",          shrink,             (0.10, 0.25, 0.40)),
    ("grow",            grow,               (0.15, 0.35)),
    ("duplicate",       duplicate,          (0.30, 1.00)),
    ("class_to_figure", _reclass("figure"), (0.30, 1.00)),
    ("class_to_header", _reclass("header"), (0.30, 1.00)),
    ("class_to_table",  _reclass("table"),  (0.30, 1.00)),
    ("shuffle_order",   shuffle_order,      (1.00,)),
    ("shift",           shift,              (0.02, 0.05)),
]


def apply(name, regions, rng, intensity):
    fn = dict((n, f) for n, f, _ in MUTATIONS)[name]
    return fn(copy.deepcopy(regions), rng, intensity)
