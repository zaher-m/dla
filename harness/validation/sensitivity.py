#!/usr/bin/env python3
"""Sensitivity: what share of injected defects does the gate actually block?

    python -m validation.sensitivity --workspace data/sample120 --corpus data/corpus_flat

For every page, take a layout believed to be roughly right, apply one defect at
one intensity, and record whether the page would now be blocked and which checks
caught it.  Reported against the unmutated baseline, because the baseline itself
trips checks and only the difference means anything.

A row near zero is a blind spot: a defect a reviewer would call obvious that the
gate would write to the store.
"""
import argparse, json, os, random, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation import assemble, checks, decide as decidemod  # noqa: E402
from validation import mutate, psr_layout  # noqa: E402
from validation.evaluate import routes_for  # noqa: E402


def baseline_layouts(ws, corpus, source, systems):
    """The layouts defects are injected into.

    `psr` uses the reference layout the PDF's own geometry implies: the closest
    thing to a correct layout available without annotation.  `model` injects on
    top of a real detector's output, which is more realistic and conflates the
    defect with whatever that detector already got wrong.
    """
    ref = json.load(open(os.path.join(ws, "inventory",
                                      "pdf_structural_reference.json")))
    routes = routes_for(ws, corpus)
    norm = os.path.join(ws, "normalized_outputs")
    out = []
    for pid, psr in sorted(ref.items()):
        r = routes.get(pid)
        if r is None or r["psr_trust"] == "unusable":
            continue
        if source == "psr":
            try:
                regions, meta = psr_layout.build(psr)
            except Exception:
                continue
            if meta["confidence"] != "usable" or len(regions) < 4:
                continue
            out.append((pid, psr, r, regions))
        else:
            for s in systems:
                f = os.path.join(norm, s, pid + ".json")
                if not os.path.exists(f):
                    continue
                regions = json.load(open(f))["regions"]
                if len(regions) >= 4:
                    out.append((pid, psr, r, regions))
    return out


def blocked(regions, psr, route, policy):
    """Would the gate stop this page, and on what?"""
    stream = assemble.assemble(regions, psr, direction=route["direction"])
    res = checks.run(regions, psr, stream, route)
    sev = {f["id"]: decidemod.severity(f["id"], f["severity"], policy)
           for f in res["findings"]}
    hits = {c for c, v in sev.items() if v == "BLOCK"}
    unver = {s["id"] for s in res["skipped"]
             if decidemod.severity(s["id"], s["severity"], policy) == "BLOCK"}
    return bool(hits or unver), hits, {f["id"] for f in res["findings"]}


def run(ws, corpus, source="psr", systems=None, seed=11, limit=None):
    norm = os.path.join(ws, "normalized_outputs")
    systems = systems or [s for s in sorted(os.listdir(norm))
                          if os.path.exists(os.path.join(norm, s, "_run.json"))]
    pages = baseline_layouts(ws, corpus, source, systems)
    if limit:
        pages = pages[:limit]
    policy = decidemod.load_policy()

    base_block = 0
    base_hits = Counter()
    for pid, psr, r, regions in pages:
        b, hits, _ = blocked(regions, psr, r, policy)
        base_block += b
        base_hits.update(hits)

    rows = []
    for name, _fn, intensities in mutate.MUTATIONS:
        for inten in intensities:
            rng = random.Random(seed)
            n = caught = same = 0
            catchers = Counter()
            for pid, psr, r, regions in pages:
                mutated = mutate.apply(name, regions, rng, inten)
                if mutated == regions:
                    same += 1
                    continue
                n += 1
                b, hits, _ = blocked(mutated, psr, r, policy)
                caught += b
                catchers.update(hits)
            rows.append({"mutation": name, "intensity": inten, "n": n,
                         "no_effect": same,
                         "blocked": caught / n if n else None,
                         "caught_by": dict(catchers.most_common(4))})
    return {"source": source, "pages": len(pages),
            "baseline_blocked": base_block / max(len(pages), 1),
            "baseline_checks": dict(base_hits.most_common(6)),
            "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--source", choices=("psr", "model"), default="psr")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--json")
    a = ap.parse_args()
    r = run(a.workspace, a.corpus, a.source, limit=a.limit)
    print(f"{r['pages']} pages, baseline layout: {r['source']}")
    print(f"baseline blocked without any defect: {r['baseline_blocked']:.1%}  "
          f"{r['baseline_checks']}")
    print(f"\n{'defect':18s} {'inten':>6s} {'n':>4s} {'blocked':>8s} {'lift':>7s}  caught by")
    for row in r["rows"]:
        if not row["n"]:
            print(f"{row['mutation']:18s} {row['intensity']:6} {0:4d}       --      --  "
                  f"(no page affected)")
            continue
        lift = row["blocked"] - r["baseline_blocked"]
        flag = "  <-- BLIND" if lift < 0.05 else ""
        print(f"{row['mutation']:18s} {row['intensity']:6} {row['n']:4d} "
              f"{row['blocked']:8.1%} {lift:+7.1%}  "
              f"{','.join(row['caught_by']) or '-'}{flag}")
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
