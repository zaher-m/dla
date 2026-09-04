#!/usr/bin/env python3
"""Run the checks over a workspace and report how often each one fires.

This exists because the obvious way to measure -- loop over normalized outputs,
call `checks.run`, count -- invites the mistake that invalidated the first three
rounds of numbers here: it is trivial to pass a fabricated route and silently
score scanned pages as if they were born-digital.  One page in the reference
sample carries three stray glyph lines under a full-page scan, and every
coverage and bucket check run against it produced nonsense that then landed in a
summary table.

So the route is not a parameter.  It is derived from the PDF, per page, by the
same router the pipeline uses, and pages whose PSR cannot be trusted are counted
as unverifiable rather than folded into a fire rate.

    python -m validation.evaluate --workspace data/sample120 --corpus data/corpus_flat
"""
import argparse, json, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # noqa: E402

from validation import assemble, checks, document, router, signals  # noqa: E402
from validation.orderlm import line_texts, order_model  # noqa: E402


def page_docs(ws):
    """page_id -> source document, for the document-level checks."""
    sel = json.load(open(os.path.join(ws, "inventory", "selected_pages.json")))
    return {p["page_id"]: p["doc"] for p in sel}


def doc_profiles(ws, ref, routes, norm, systems):
    """Per (system, document) profile, built in a pass before any check runs.

    The document checks compare a page against the rest of its own document, so
    the profile has to exist before the first page is judged.
    """
    docs = page_docs(ws)
    facts = defaultdict(list)
    for pid, psr in ref.items():
        r = routes.get(pid)
        if r is None or r["psr_trust"] == "unusable":
            continue
        for s in systems:
            f = os.path.join(norm, s, pid + ".json")
            if not os.path.exists(f):
                continue
            regions = json.load(open(f))["regions"]
            facts[(s, docs.get(pid))].append(
                document.page_facts(regions, psr, checks.page_columns(psr)))
    return {k: document.profile(v) for k, v in facts.items()}, docs


def routes_for(ws, corpus, cache=True):
    """Route every selected page through the real router.

    Cached in the workspace: routing reads every page of every source document
    to establish text direction, which costs about a minute on a 35-document
    corpus and is paid again by every tool that needs a route.
    """
    path = os.path.join(ws, "inventory", "page_routes.json")
    sel_path = os.path.join(ws, "inventory", "selected_pages.json")
    if cache and os.path.exists(path) and \
            os.path.getmtime(path) >= os.path.getmtime(sel_path):
        with open(path, encoding="utf8") as f:
            return json.load(f)
    sel = json.load(open(sel_path))
    by_doc = defaultdict(list)
    for p in sel:
        by_doc[p["doc"]].append(p)
    t = router.load_thresholds()
    out = {}
    for doc, pages in by_doc.items():
        d = fitz.open(os.path.join(corpus, doc))
        try:
            # Direction is a property of the document, so the context is built
            # from every page of it, not only the sampled ones.
            sigs = [signals.page_signals(d, i) for i in range(d.page_count)]
            ctx = router.document_context(sigs, t)
            for p in pages:
                out[p["page_id"]] = router.route(sigs[p["page"] - 1], t, ctx)
        finally:
            d.close()
    if cache:
        with open(path, "w", encoding="utf8") as f:
            json.dump(out, f)
    return out


def evaluate(ws, corpus, systems=None):
    ref = json.load(open(os.path.join(ws, "inventory",
                                      "pdf_structural_reference.json")))
    routes = routes_for(ws, corpus)
    norm = os.path.join(ws, "normalized_outputs")
    systems = systems or [s for s in sorted(os.listdir(norm))
                          if os.path.exists(os.path.join(norm, s, "_run.json"))
                          and json.load(open(os.path.join(norm, s, "_run.json"))
                                        ).get("status") == "ok"]
    profiles, docs = doc_profiles(ws, ref, routes, norm, systems)
    texts = line_texts(ws, corpus, ref)
    lm = order_model(ws, texts)
    fire, na, unver = Counter(), Counter(), Counter()
    per_sys = Counter(); per_sys_n = Counter(); per_sys_unver = Counter()
    kinds = Counter(); pairs = 0
    skipped_pages = Counter()
    for pid, psr in ref.items():
        r = routes.get(pid)
        if r is None:
            continue
        kinds[r["page_kind"]] += 1
        for s in systems:
            f = os.path.join(norm, s, pid + ".json")
            if not os.path.exists(f):
                continue
            per_sys_n[s] += 1
            if r["psr_trust"] == "unusable":
                # Not a pass and not a failure: the page cannot be checked.
                skipped_pages[s] += 1
                per_sys_unver[s] += 1
                continue
            pairs += 1
            regions = json.load(open(f))["regions"]
            stream = assemble.assemble(regions, psr, direction=r["direction"])
            res = checks.run(regions, psr, stream, r,
                             doc=profiles.get((s, docs.get(pid))),
                             line_text=texts.get(pid), lm=lm)
            for x in res["findings"]:
                fire[x["id"]] += 1
            for x in res["inapplicable"]:
                na[x["id"]] += 1
            for x in res["skipped"]:
                unver[x["id"]] += 1
            if res["n_block"]:
                per_sys[s] += 1
            if res["unverifiable"]:
                per_sys_unver[s] += 1
    return {"systems": systems, "pairs": pairs, "page_kinds": dict(kinds),
            "fire": fire, "na": na, "unver": unver, "per_sys": per_sys,
            "per_sys_n": per_sys_n, "per_sys_unver": per_sys_unver,
            "unusable_pages": dict(skipped_pages)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--systems", nargs="*")
    ap.add_argument("--json")
    a = ap.parse_args()
    r = evaluate(a.workspace, a.corpus, a.systems)
    n = max(r["pairs"], 1)
    print(f"{r['pairs']} checkable (system,page) pairs over "
          f"{len(r['systems'])} systems")
    print(f"page kinds: {r['page_kinds']}")
    print(f"\n{'check':8s} {'sev':6s} {'fires':>7s} {'n/a':>7s} {'unver':>7s}")
    reg = {c["id"]: c for c in checks.REGISTRY}
    for cid in sorted(reg, key=lambda k: -r["fire"][k]):
        print(f"{cid:8s} {reg[cid]['severity']:6s} {r['fire'][cid]/n:7.1%} "
              f"{r['na'][cid]/n:7.1%} {r['unver'][cid]/n:7.1%}")
    print(f"\n{'system':30s} pages  block  unverifiable")
    for s in r["systems"]:
        d = max(r["per_sys_n"][s], 1)
        print(f"{s:30s} {r['per_sys_n'][s]:5d}  {r['per_sys'][s]/d:5.1%}  "
              f"{r['per_sys_unver'][s]/d:11.1%}")
    if a.json:
        json.dump({k: (dict(v) if isinstance(v, Counter) else v)
                   for k, v in r.items()}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
