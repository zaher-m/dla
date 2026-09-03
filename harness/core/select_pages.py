#!/usr/bin/env python3
"""Choose the pages to evaluate, then render them.

Three modes, because the same code serves two very different callers:

  ``all``         every page of every document, capped at ``selection.max_pages``.
                  This is what the web application uses — someone who uploads a
                  document wants that document analysed, not a sample of it.
  ``stratified``  quota-based sampling across layout strata (multi-column, ruled
                  table, figure-heavy, sparse title page, …), spread across
                  source documents.  This is how the reference benchmark chose
                  its 29 pages: it stresses the dimensions that separate layout
                  models instead of over-weighting whatever is most common.
  ``first``       the first N pages in document order. Useful for a smoke test.
  ``random``      a uniform random sample, seeded and therefore reproducible.
                  Every other mode is biased on purpose: ``stratified`` seeks out
                  the pages that separate models, ``first`` takes whatever the
                  document opens with.  Both are useless for estimating how often
                  something happens *in the corpus*, because every statistic they
                  produce is conditioned on the selection rule.  This mode exists
                  so escalation rates and error rates can be quoted about the
                  corpus rather than about a sample chosen to be interesting.

Rendering is part of selection rather than a separate stage because the two must
agree exactly: every metric, every overlay and every model input is derived from
the same 300 dpi raster, and the pixel dimensions recorded here are what the
whole pipeline treats as the page's coordinate space.

    python -m core.select_pages --mode all
    python -m core.select_pages --mode stratified --workspace benchmark
    python -m core.select_pages --mode random --max-pages 120 --seed 7
"""
import argparse, hashlib, json, os, random, sys
from collections import Counter

import fitz

# Import the harness package regardless of how this module is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths  # noqa: E402

DEFAULT_QUOTAS = {
    "dense_table_portrait": 4, "dense_table_landscape": 3, "three_column": 2,
    "two_column_portrait": 4, "two_column_landscape": 3, "figure_heavy": 3,
    "sparse_title_page": 2, "graphics_dense": 2,
    "single_column_portrait": 3, "single_column_landscape": 2,
}


def stratum(p):
    """Coarse layout class, from the inventory's geometric signals alone."""
    land = p["width_pt"] > p["height_pt"]
    if p["table_likely"] and p["vlines"] >= 8:
        return "dense_table_landscape" if land else "dense_table_portrait"
    if p["est_columns"] >= 3:
        return "three_column"
    if p["est_columns"] == 2:
        return "two_column_landscape" if land else "two_column_portrait"
    if p["n_images"] >= 3 or p["image_area_frac"] > 0.05:
        return "figure_heavy"
    if p["char_count"] < 700:
        return "sparse_title_page"
    if p["n_drawings"] > 60:
        return "graphics_dense"
    return "single_column_landscape" if land else "single_column_portrait"


def pick_stratified(pages, quotas, cap):
    by = {}
    for p in pages:
        by.setdefault(p["stratum"], []).append(p)
    sel = []
    for s, q in quotas.items():
        cand = by.get(s, [])
        if not cand:
            continue
        cand.sort(key=lambda p: (-p["text_blocks"], -p["char_count"]))
        seen_docs, picked = {}, []
        for p in cand:                    # round-robin over documents for diversity
            k = p["doc"]
            if seen_docs.get(k, 0) < max(1, q // 2) and len(picked) < q:
                picked.append(p)
                seen_docs[k] = seen_docs.get(k, 0) + 1
        for p in cand:
            if len(picked) >= q:
                break
            if p not in picked:
                picked.append(p)
        sel.extend(picked[:q])

    # every source document must be represented, or the report describes a
    # corpus it did not actually look at
    cnt = Counter(p["doc"] for p in sel)
    for doc in sorted({p["doc"] for p in pages}):
        need = 2 - cnt.get(doc, 0)
        if need > 0:
            extra = sorted((p for p in pages if p["doc"] == doc and p not in sel),
                           key=lambda p: (-p["text_blocks"], -p["char_count"]))[:need]
            sel.extend(extra)
    return sel[:cap] if cap else sel


def render(sel, corpus, ws, dpi_primary, dpi_secondary):
    work = os.path.join(ws, "working")
    subs = {dpi_primary: "pages_300dpi", dpi_secondary: "pages_150dpi"}
    for sub in list(subs.values()) + ["pages_pdf"]:
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    opened = {}
    for p in sel:
        src = opened.get(p["doc"])
        if src is None:
            src = opened[p["doc"]] = fitz.open(os.path.join(corpus, p["doc"]))
        page = src[p["page"] - 1]
        for dpi, sub in subs.items():
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
            out = os.path.join(work, sub, f"{p['page_id']}.png")
            pix.save(out)
            if dpi == dpi_primary:
                p["render_300dpi"] = out
                p["px_width"], p["px_height"] = pix.width, pix.height
                with open(out, "rb") as f:
                    p["sha256"] = hashlib.sha256(f.read()).hexdigest()[:16]
        # single-page PDF for the PDF-native pipelines, byte-identical content
        one = fitz.open()
        one.insert_pdf(src, from_page=p["page"] - 1, to_page=p["page"] - 1)
        one.save(os.path.join(work, "pages_pdf", f"{p['page_id']}.pdf"))
        one.close()
        p["page_pdf"] = os.path.join(work, "pages_pdf", f"{p['page_id']}.pdf")
    for d in opened.values():
        d.close()
    return sel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default=None,
                    choices=["all", "stratified", "first", "random"])
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for --mode random; recorded so the sample is reproducible")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--workspace", default=None)
    a = ap.parse_args()

    corpus = paths.resolve(a.corpus) if a.corpus else paths.CORPUS
    ws = paths.ensure_workspace(paths.resolve(a.workspace) if a.workspace else None)
    mode = a.mode or paths.get("selection", "mode", "all")
    cap = a.max_pages if a.max_pages is not None else paths.get("selection", "max_pages", 25)
    quotas = paths.get("selection", "quotas") or DEFAULT_QUOTAS
    dpi1 = paths.get("render", "dpi_primary", 300)
    dpi2 = paths.get("render", "dpi_secondary", 150)

    inv_path = os.path.join(ws, "inventory", "corpus_inventory.json")
    if not os.path.exists(inv_path):
        raise SystemExit(f"no inventory at {inv_path} — run core.inventory first")
    with open(inv_path, encoding="utf8") as f:
        inv = json.load(f)
    pages = inv["pages"]
    for p in pages:
        p["stratum"] = stratum(p)

    if mode == "stratified":
        sel = pick_stratified(pages, quotas, cap)
    elif mode == "first":
        sel = pages[:cap] if cap else list(pages)
    elif mode == "random":
        seed = a.seed if a.seed is not None else paths.get("selection", "seed", 0)
        rng = random.Random(seed)
        sel = list(pages)
        rng.shuffle(sel)
        sel = sel[:cap] if cap else sel
        print(f"[select] uniform random sample, seed={seed}")
    else:
        sel = list(pages)
        if cap and len(sel) > cap:
            print(f"[select] {len(sel)} pages exceeds selection.max_pages={cap}; "
                  f"taking the first {cap}. Raise DLA_SELECTION_MAX_PAGES to include more.")
            sel = sel[:cap]

    sel.sort(key=lambda p: (p["doc"], p["page"]))
    for i, p in enumerate(sel, 1):
        p["page_id"] = f"page_{i:03d}"

    sel = render(sel, corpus, ws, dpi1, dpi2)
    out = os.path.join(ws, "inventory", "selected_pages.json")
    with open(out, "w", encoding="utf8") as f:
        json.dump(sel, f, ensure_ascii=False, indent=2)

    print(f"selected: {len(sel)} pages  mode={mode}  corpus={corpus}")
    for k, v in Counter(p["stratum"] for p in sel).most_common():
        print(f"  {k:26s} {v}")
    n_scanned = sum(1 for p in sel if not p.get("has_text_layer"))
    if n_scanned:
        print(f"  note: {n_scanned}/{len(sel)} pages carry no text layer. The PDF "
              f"Structural Reference cannot be built for those, so geometric "
              f"metrics will be absent; the side-by-side comparison still works.")


if __name__ == "__main__":
    main()
