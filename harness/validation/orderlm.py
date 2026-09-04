#!/usr/bin/env python3
"""Is this sequence of lines plausible as language?

Every other order check compares one derivation against another, or asks whether
an order is internally consistent. None of them reads the words. But for a
born-digital page the words are right there, and a correct order joins them into
running text while a wrong one splices unrelated fragments.

The model is a character 5-gram plus a word bigram, trained on the corpus's own
text -- no external model, no download, works for Arabic, CPU only. It is trained
on WITHIN-LINE text exclusively, so it never sees a line-to-line transition and
cannot have memorised any ordering; scoring then asks a question it was not
trained on.

Only the n-grams that straddle a junction are scored. Including a window either
side lets the model's opinion of ordinary Arabic dominate, and that opinion is
identical whichever line follows which -- an early version of this scored a
window and separated a true order from a shuffled one barely above chance.

Measured accuracy, choosing the true order over a shuffled one on 994 pages of
the reference corpus (docs/validation-experiments.md, E5):

    pages with prose        94%
    mostly-prose pages      88%
    table-like pages        80%
    all                     85%

That is a feature, not a gate. It must never block a page.
"""
import json, math, os
from collections import Counter

import fitz

N = 5                    # character n-gram order
MIN_COUNT = 2            # prune singletons: they are typos and page numbers
CHAR_FLOOR = 0.1
WORD_FLOOR = 0.05


def _norm(t):
    return " ".join(t.split())


def train(line_texts):
    """line_texts: an iterable of strings, each one line of a page."""
    counts, ctx, words, uni = Counter(), Counter(), Counter(), Counter()
    for t in line_texts:
        s = " " + _norm(t) + " "
        if len(s) < N:
            continue
        for i in range(len(s) - N + 1):
            counts[s[i:i + N]] += 1
            ctx[s[i:i + N - 1]] += 1
        w = s.split()
        for a, b in zip(w, w[1:]):
            words[a + "\t" + b] += 1
            uni[a] += 1
    counts = {k: v for k, v in counts.items() if v >= MIN_COUNT}
    words = {k: v for k, v in words.items() if v >= MIN_COUNT}
    ctx = {k: v for k, v in ctx.items() if v >= MIN_COUNT}
    return {"n": N, "counts": counts, "ctx": ctx, "words": words,
            "uni": dict(uni), "alphabet": len({k[-1] for k in counts}) or 1}


def save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(model, f, ensure_ascii=False)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf8") as f:
        return json.load(f)


def junction(model, a, b):
    """Log-probability of the text crossing the boundary from line a to line b."""
    ta, tb = _norm(a), _norm(b)
    if not ta or not tb:
        return None
    n, V = model["n"], model["alphabet"]
    counts, ctx = model["counts"], model["ctx"]
    s = ta[-(n - 1):] + " " + tb[:n - 1]
    p = len(ta[-(n - 1):])
    lp, k = 0.0, 0
    for i in range(max(0, p - n + 1), min(p + 1, len(s) - n + 1)):
        g = s[i:i + n]
        lp += math.log((counts.get(g, 0) + CHAR_FLOOR)
                       / (ctx.get(g[:-1], 0) + CHAR_FLOOR * V))
        k += 1
    wa, wb = ta.split()[-1], tb.split()[0]
    uni = model["uni"]
    wlp = math.log((model["words"].get(wa + "\t" + wb, 0) + WORD_FLOOR)
                   / (uni.get(wa, 0) + WORD_FLOOR * max(len(uni), 1)))
    return lp / max(k, 1) + wlp


def score(model, texts):
    """Mean junction score over a sequence of line texts.  None if unscorable."""
    v = [j for j in (junction(model, a, b) for a, b in zip(texts, texts[1:]))
         if j is not None]
    return sum(v) / len(v) if v else None


def prosiness(texts):
    """Share of lines carrying real words rather than figures and labels.

    The signal comes from language, and a junction between two numeric table
    cells has none: the same measurement is 94% accurate on pages with prose and
    80% on table-like ones.  Callers gate on this rather than pretending the
    score means the same thing everywhere.
    """
    if not texts:
        return 0.0
    n = sum(1 for t in texts
            if len([w for w in t.split()
                    if len(w) > 2 and not any(c.isdigit() for c in w)]) >= 4)
    return n / len(texts)


def line_texts(ws, corpus, ref):
    """The words on each line, per page, aligned to psr["text_lines"].

    Read here rather than stored in the PSR: the reference is geometric, and
    carrying the text of every line would multiply the size of every workspace
    for the one check that reads it.
    """
    out, docs = {}, {}
    for pid, psr in ref.items():
        name, pno = psr.get("doc"), psr.get("page")
        if not name or not pno:
            continue
        d = docs.get(name)
        if d is None:
            path = os.path.join(corpus, name)
            if not os.path.exists(path):
                continue
            d = docs[name] = fitz.open(path)
        rows = []
        for blk in d[pno - 1].get_text("rawdict")["blocks"]:
            if blk["type"] != 0:
                continue
            for ln in blk["lines"]:
                b = ln["bbox"]
                if b[2] - b[0] > 1 and b[3] - b[1] > 1:
                    rows.append("".join(c["c"] for sp in ln["spans"]
                                        for c in sp["chars"]))
        # The PSR keeps lines in this same order and filters on the same size
        # test, so the two lists line up index for index.
        out[pid] = rows if len(rows) == len(psr["text_lines"]) else None
    for d in docs.values():
        d.close()
    return out


def order_model(ws, texts):
    """Train the junction model on this workspace's own text, once."""
    path = os.path.join(ws, "validation", "order_lm.json")
    m = load(path)
    if m:
        return m
    lines = [t for rows in texts.values() if rows for t in rows]
    if len(lines) < 500:
        return None
    m = train(lines)
    save(m, path)
    return m


def main():
    """Report how plausibly each page's reading order joins its words.

        python -m validation.orderlm --workspace data/sample120 --corpus data/corpus_flat

    Independent of every other order check: those compare one derivation against
    another or test an order's internal consistency, and none of them reads what
    the page says.
    """
    import argparse, statistics
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from validation import assemble, checks
    from validation.evaluate import routes_for

    ap = argparse.ArgumentParser(description=main.__doc__.split("\n")[0])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--system")
    a = ap.parse_args()

    ref = json.load(open(os.path.join(a.workspace, "inventory",
                                      "pdf_structural_reference.json")))
    routes = routes_for(a.workspace, a.corpus)
    norm = os.path.join(a.workspace, "normalized_outputs")
    systems = [a.system] if a.system else [
        s for s in sorted(os.listdir(norm))
        if os.path.exists(os.path.join(norm, s, "_run.json"))]
    texts = line_texts(a.workspace, a.corpus, ref)
    model = order_model(a.workspace, texts)
    if model is None:
        raise SystemExit("not enough text in this workspace to train the model")
    print(f"model: {len(model['counts'])} char {model['n']}-grams, "
          f"{len(model['words'])} word bigrams")

    for s in systems:
        rows, mute = [], Counter()
        for pid, psr in sorted(ref.items()):
            r = routes.get(pid)
            f = os.path.join(norm, s, pid + ".json")
            if r is None or r["psr_trust"] == "unusable" or not os.path.exists(f):
                continue
            txt = texts.get(pid)
            if txt is None:
                mute["no text"] += 1
                continue
            regions = json.load(open(f))["regions"]
            st = assemble.assemble(regions, psr, direction=r["direction"])
            seq = [i for i in st["sequence"] if i < len(txt)]
            if len(seq) < 12:
                mute["too few lines"] += 1
                continue
            w = [txt[i] for i in seq]
            if prosiness(w) < 0.15:
                mute["no prose to read"] += 1
                continue
            alt = sorted(seq, key=lambda i: (round(psr["text_lines"][i][1], 1),
                                             psr["text_lines"][i][0]))
            got, other = score(model, w), score(model, [txt[i] for i in alt])
            if got is None or other is None:
                continue
            rows.append((other - got, pid))
        if not rows:
            print(f"\n{s}: nothing scorable  {dict(mute)}")
            continue
        d = sorted(r[0] for r in rows)
        print(f"\n{s}: {len(rows)} pages scored  {dict(mute)}")
        print(f"  alternative minus this order: p10 {d[int(.1*len(d))]:+.3f}  "
              f"median {statistics.median(d):+.3f}  p90 {d[int(.9*len(d))]:+.3f}")
        print("  positive means the page reads better in plain top-to-bottom order")
        for delta, pid in sorted(rows, reverse=True)[:3]:
            print(f"    {pid}  {delta:+.3f}")


if __name__ == "__main__":
    main()
