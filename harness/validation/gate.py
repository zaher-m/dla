#!/usr/bin/env python3
"""The gate: decide every page of a workspace and build the review queue.

    python -m validation.gate --workspace data/sample120 --corpus data/corpus_flat

Writes one decision record per (system, page) and one queue of reviewer tasks,
then prints the decision mix against the escalation budget.  The mix is the
number to watch: the budget in docs/validation/decision-and-escalation.md expects
70-80% accepted and 12-20% escalated by the deterministic checks, and a
deterministic share above 25% means the thresholds are too tight rather than the
corpus being bad.

The route is derived here exactly as it is in `validation.evaluate`, never
passed in.  Handing this tool a route is how scanned pages get scored as if they
were born-digital, which invalidated three earlier rounds of numbers.
"""
import argparse, json, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation import assemble, checks, decide as decidemod  # noqa: E402
from validation.orderlm import line_texts, order_model  # noqa: E402
from validation.evaluate import doc_profiles, routes_for  # noqa: E402


def gate(ws, corpus, systems=None, policy=None):
    ref = json.load(open(os.path.join(ws, "inventory",
                                      "pdf_structural_reference.json")))
    routes = routes_for(ws, corpus)
    norm = os.path.join(ws, "normalized_outputs")
    systems = systems or [s for s in sorted(os.listdir(norm))
                          if os.path.exists(os.path.join(norm, s, "_run.json"))
                          and json.load(open(os.path.join(norm, s, "_run.json"))
                                        ).get("status") == "ok"]
    p = policy or decidemod.load_policy()
    score = decidemod.scorer(p)
    texts = line_texts(ws, corpus, ref)
    lm = order_model(ws, texts)
    profiles, docs = doc_profiles(ws, ref, routes, norm, systems)
    out = defaultdict(list)
    for pid, psr in sorted(ref.items()):
        r = routes.get(pid)
        if r is None:
            continue
        for s in systems:
            f = os.path.join(norm, s, pid + ".json")
            if not os.path.exists(f):
                continue
            if r["psr_trust"] == "unusable":
                # No checks run, so the record carries an empty result: the
                # decision is made by the route alone and says so.
                res = {"findings": [], "skipped": [], "inapplicable": [],
                       "families": []}
            else:
                regions = json.load(open(f))["regions"]
                stream = assemble.assemble(regions, psr, direction=r["direction"])
                res = checks.run(regions, psr, stream, r,
                                 doc=profiles.get((s, docs.get(pid))),
                                 line_text=texts.get(pid), lm=lm)
            d = decidemod.decide(res, r, policy=p, score=score)
            out[s].append({"page_id": pid, "doc": docs.get(pid), **d})
    return dict(out), p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--systems", nargs="*")
    ap.add_argument("--write", action="store_true",
                    help="persist decision records and the review queue")
    a = ap.parse_args()
    dec, p = gate(a.workspace, a.corpus, a.systems)
    print(f"policy: discard={p['discard']} unusable={p['unusable']} "
          f"risk={p['risk']}")
    print(f"\n{'system':30s} {'pages':>6s} {'accept':>8s} {'escalate':>9s} "
          f"{'defer':>7s} {'reject':>7s}")
    tasks = Counter()
    for s, rows in sorted(dec.items()):
        c = Counter(r["decision"] for r in rows)
        n = max(len(rows), 1)
        tasks.update(r["task"] for r in rows if r["task"])
        print(f"{s:30s} {len(rows):6d} {c['accept']/n:8.1%} "
              f"{c['escalate']/n:9.1%} {c['defer']/n:7.1%} {c['reject']/n:7.1%}")
    n = sum(len(r) for r in dec.values()) or 1
    print(f"\ntask mix: " + "  ".join(f"{k} {v/n:.1%}" for k, v in sorted(tasks.items())))
    top = Counter(f["id"] for rows in dec.values() for r in rows
                  if r["decision"] == "escalate" for f in r["findings"]
                  if f["severity"] == "BLOCK")
    if top:
        print("escalated by: " + "  ".join(f"{k} {v}" for k, v in top.most_common(8)))
    if a.write:
        # The same writer the pipeline stage uses, so the files a tool produces
        # and the files a job produces cannot drift apart.
        from validation.stage import write
        _, k = write(a.workspace, dec, p)
        print(f"\n{k} escalations -> "
              f"{os.path.join(a.workspace, 'validation', 'queue.json')}")


if __name__ == "__main__":
    main()
