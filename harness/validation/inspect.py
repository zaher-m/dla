#!/usr/bin/env python3
"""Render the pages a check fires on, so a claim can be looked at.

Every threshold in this package was fitted against a distribution, and a
distribution cannot tell you whether the thing being counted is real.  The one
check that was carried through to a headline number without this step -- missed
tables -- turned out to be 97% false positives: charts, running headers, framed
paragraphs and empty boxes.  Rendering is what caught that.

    python -m validation.inspect --workspace data/sample120 \
        --corpus data/corpus_flat --system docling.heron --check C2-01 \
        --out /tmp/c2-01.png

Red is the region the finding names, green is every other region the system
predicted, blue is the PSR structure the check compared against.
"""
import argparse, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw  # noqa: E402

from validation import assemble, checks, evaluate  # noqa: E402

CELL = 560
RED, GREEN, BLUE = (220, 0, 0), (0, 150, 0), (30, 90, 220)


def collect(ws, corpus, system, check_id):
    """Findings for one system, with pages routed exactly as the pipeline does.

    The route is derived, never supplied.  An earlier version passed a fixed
    `psr_trust: full` and so rendered findings from pages the pipeline excludes
    -- including ones whose text is drawn as vector outlines, where the
    reference holds three lines and every check produces nonsense.  Verifying a
    check against findings it would never emit is worse than not verifying it.
    """
    ref = json.load(open(os.path.join(ws, "inventory",
                                      "pdf_structural_reference.json")))
    routes = evaluate.routes_for(ws, corpus)
    norm = os.path.join(ws, "normalized_outputs", system)
    profiles, docs = evaluate.doc_profiles(
        ws, ref, routes, os.path.join(ws, "normalized_outputs"), [system])
    hits = []
    for pid, psr in ref.items():
        f = os.path.join(norm, pid + ".json")
        r = routes.get(pid)
        if not os.path.exists(f) or r is None or r["psr_trust"] == "unusable":
            continue
        regions = json.load(open(f))["regions"]
        stream = assemble.assemble(regions, psr, direction=r["direction"])
        res = checks.run(regions, psr, stream, r,
                         doc=profiles.get((system, docs.get(pid))))
        for fd in res["findings"]:
            if fd["id"] == check_id:
                hits.append((pid, psr, regions, fd, stream))
    return hits


def _draw_order(dr, stream, colour=(230, 60, 0)):
    """The predicted reading sequence, as a numbered path between block centres.

    A box overlay cannot show an order error at all -- the boxes are identical
    whichever sequence they are read in -- so an order finding is unverifiable
    without this.
    """
    pts = []
    for b in sorted(stream["blocks"], key=lambda x: x["rank"]):
        if not b["lines"]:
            continue
        x1, y1, x2, y2 = b["bbox"]
        pts.append(((x1 + x2) / 2, (y1 + y2) / 2))
    if len(pts) > 1:
        dr.line(pts, fill=colour, width=7)
    for n, (px, py) in enumerate(pts, 1):
        dr.ellipse([px - 22, py - 22, px + 22, py + 22], fill=(255, 255, 255),
                   outline=colour, width=5)
        dr.text((px - 8, py - 8), str(n), fill=colour)


def sheet(ws, hits, out, cols=3, rows=2, seed=0, psr_key=None, order=False):
    pick = hits if len(hits) <= cols * rows else random.Random(seed).sample(
        hits, cols * rows)
    img = Image.new("RGB", (CELL * cols, CELL * rows), "white")
    dr = ImageDraw.Draw(img)
    for n, (pid, psr, regions, fd, stream) in enumerate(pick):
        page = Image.open(os.path.join(ws, "working", "pages_300dpi",
                                       pid + ".png")).convert("RGB")
        d2 = ImageDraw.Draw(page)
        for key in (psr_key or []):
            for b in (psr.get(key) or []):
                # column_bands are x-ranges, not boxes: draw them full height
                box = ([b[0], 0, b[1], psr["height"]] if len(b) == 2
                       else [b[0], b[1], b[2], b[3]])
                d2.rectangle(box, outline=BLUE, width=5)
        named = set(fd.get("regions") or [])
        for i, r in enumerate(regions):
            d2.rectangle(r["bbox"], outline=RED if i in named else GREEN,
                         width=9 if i in named else 3)
        if order:
            _draw_order(d2, stream)
        page.thumbnail((CELL - 10, CELL - 40))
        x, y = (n % cols) * CELL, (n // cols) * CELL
        img.paste(page, (x + 5, y + 34))
        dr.text((x + 6, y + 8), f"{pid}  {fd['id']} v={fd.get('value')}", fill=(0, 0, 0))
        dr.text((x + 6, y + 20), fd["message"][:110], fill=(80, 80, 80))
    img.save(out)
    return len(pick), len(hits)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--check", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--psr", nargs="*", default=[],
                    help="PSR keys to draw in blue, e.g. gutters column_bands")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order", action="store_true",
                    help="draw the predicted reading sequence")
    a = ap.parse_args()
    hits = collect(a.workspace, a.corpus, a.system, a.check)
    n, tot = sheet(a.workspace, hits, a.out, seed=a.seed, psr_key=a.psr,
                   order=a.order)
    print(f"{a.check}: {tot} findings, rendered {n} -> {a.out}")


if __name__ == "__main__":
    main()
