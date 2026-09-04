#!/usr/bin/env python3
"""Pipeline stage: decide every page of a workspace and write the results.

    python -m validation.stage --workspace data/jobs/abc --corpus data/uploads

This is the one entry point a job runs.  `validation.evaluate` and
`validation.gate` stay as measurement tools for a corpus; this writes the
products the rest of the system consumes:

    validation/decisions/<system>.json   one record per page, the full evidence
    validation/queue.json                the reviewer's work, escalations only
    validation/deferred.json             pages this pipeline cannot process yet
    validation/summary.json              small enough for a report bundle

and reads one back:

    validation/verdicts.jsonl            what reviewers sent back

The three files are the interface.  Nothing downstream imports this package: a
consumer reads JSON with a `schema` field on it, which is what lets the queue be
served to an annotation platform, the summary be embedded in a report, and both
be produced by a container that holds no models.
"""
import argparse, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation import audit as auditmod  # noqa: E402
from validation import decide as decidemod  # noqa: E402
from validation import verdicts as verdictsmod  # noqa: E402
from validation.gate import gate  # noqa: E402

# Bump on any change to the shape of a record below.  Consumers should refuse a
# major they do not know rather than silently misread a field.
SCHEMA = "dla.validation/1"


def summarise(decisions, policy):
    """Per system, what the gate decided.  Kept small: a report embeds this."""
    out = {"schema": SCHEMA, "policy": {k: policy[k] for k in
                                        ("discard", "unusable", "risk")},
           "systems": {}, "pages": {}}
    for s, rows in decisions.items():
        c = Counter(r["decision"] for r in rows)
        tasks = Counter(r["task"] for r in rows if r["task"])
        fired = Counter(f["id"] for r in rows for f in r["findings"])
        blocked = Counter(f["id"] for r in rows if r["decision"] == "escalate"
                          for f in r["findings"] if f["severity"] == "BLOCK")
        n = max(len(rows), 1)
        out["systems"][s] = {
            "pages": len(rows),
            "decisions": dict(c),
            "accept_rate": round(c["accept"] / n, 4),
            "escalate_rate": round(c["escalate"] / n, 4),
            "defer_rate": round(c["defer"] / n, 4),
            "tasks": dict(tasks),
            "fired": dict(fired.most_common()),
            "escalated_by": dict(blocked.most_common()),
        }
        # Per page, per system: enough for a viewer to colour a panel and say
        # why, and nothing more.  The full evidence stays in decisions/.
        for r in rows:
            out["pages"].setdefault(r["page_id"], {})[s] = {
                "decision": r["decision"], "task": r["task"],
                "risk": r["risk"], "reason": r["reason"],
                "n_findings": len(r["findings"]),
            }
    return out


def write(ws, decisions, policy):
    d_ = os.path.join(ws, "validation")
    os.makedirs(os.path.join(d_, "decisions"), exist_ok=True)
    for s, rows in decisions.items():
        with open(os.path.join(d_, "decisions", s + ".json"), "w", encoding="utf8") as f:
            json.dump({"schema": SCHEMA, "system": s, "pages": rows}, f, indent=1)

    # The audit stratum: accepted pages a reviewer looks at anyway.  Marked on
    # the decision record before the queue is built, so a record always says
    # whether the page was audited even if the queue is regenerated.
    for s, rows in decisions.items():
        for r in auditmod.select(rows, s, policy):
            r["audit"] = True

    queue = [{"system": s, "page_id": r["page_id"], "doc": r["doc"],
              "task": r["task"], "reason": r["reason"], "risk": r["risk"],
              # The reviewer is told what fired, in words, and which regions to
              # open.  A score tells them nothing they can act on.
              "findings": [{"id": f["id"], "severity": f["severity"],
                            "regions": f.get("regions", []),
                            "message": f["message"]} for f in r["findings"]]}
             for s, rows in decisions.items() for r in rows
             if r["decision"] == "escalate"]
    # E7 never competes with the rest of the queue and is never down-sampled
    # when the queue is busy: the moment it flexes with load it stops being a
    # random sample of what was accepted, and the estimate it exists to support
    # is gone.
    queue += [{"system": s, "page_id": r["page_id"], "doc": r["doc"],
               "task": "E7", "reason": "random audit of an accepted page",
               "risk": r["risk"], "findings": []}
              for s, rows in decisions.items() for r in rows if r.get("audit")]
    queue.sort(key=lambda q: (q["task"], -q["risk"]))
    with open(os.path.join(d_, "queue.json"), "w", encoding="utf8") as f:
        json.dump({"schema": SCHEMA, "tasks": queue}, f, indent=1)

    # The backlog.  A deferred page is not a failure and not a pass: it is work
    # this pipeline does not do yet, and without a list of it "deferred for now"
    # quietly becomes "dropped".  Keyed by page rather than by system, because
    # the reason is a property of the file and every system on that page shares
    # it.  This is the input the scanned path will be built against.
    seen, deferred = set(), []
    for rows in decisions.values():
        for r in rows:
            if r["decision"] != "defer" or r["page_id"] in seen:
                continue
            seen.add(r["page_id"])
            deferred.append({"page_id": r["page_id"], "doc": r["doc"],
                             "page_kind": r["page_kind"],
                             "psr_trust": r["psr_trust"],
                             "reason": r["reason"]})
    deferred.sort(key=lambda d: (d["doc"] or "", d["page_id"]))
    by_doc = Counter(d["doc"] for d in deferred)
    with open(os.path.join(d_, "deferred.json"), "w", encoding="utf8") as f:
        json.dump({"schema": SCHEMA, "n_pages": len(deferred),
                   "by_reason": dict(Counter(x["reason"] for x in deferred)),
                   "by_document": dict(by_doc.most_common()),
                   "pages": deferred}, f, indent=1)

    summary = summarise(decisions, policy)
    summary["deferred"] = {"n_pages": len(deferred),
                           "by_reason": dict(Counter(x["reason"] for x in deferred))}
    rate, seed = auditmod.settings(policy)
    summary["audit"] = {
        "rate": rate, "seed": seed,
        "n_sampled": sum(1 for rows in decisions.values()
                         for r in rows if r.get("audit"))}
    # Whatever reviewers have sent back so far, and what it is worth.  Absent on
    # a fresh workspace, which is the normal state and not an error.
    try:
        v = verdictsmod.load(ws)
    except ValueError as e:
        print(f"[validation] verdicts unreadable: {e}")
        v = []
    if v:
        summary["verdicts"] = verdictsmod.estimate(v)
    with open(os.path.join(d_, "summary.json"), "w", encoding="utf8") as f:
        json.dump(summary, f, indent=1)
    return summary, len(queue)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workspace", default=os.environ.get("DLA_WORKSPACE"))
    ap.add_argument("--corpus", default=None)
    a, _ = ap.parse_known_args()
    ws = a.workspace
    if not ws:
        sys.exit("[validation] no workspace")
    corpus = a.corpus or os.path.join(ws, "corpus")

    # A stage runs on whatever the job produced.  No reference, no systems, or
    # no pages is a job too small to validate, not a failure: say so and stop,
    # rather than aborting a pipeline that has a perfectly good report to build.
    need = os.path.join(ws, "inventory", "pdf_structural_reference.json")
    if not os.path.exists(need):
        print(f"[validation] no structural reference in {ws}; nothing to validate")
        return
    norm = os.path.join(ws, "normalized_outputs")
    if not os.path.isdir(norm) or not os.listdir(norm):
        print(f"[validation] no normalized outputs in {ws}; nothing to validate")
        return

    decisions, policy = gate(ws, corpus)
    if not decisions:
        print("[validation] no system produced a page to decide")
        return
    summary, n = write(ws, decisions, policy)
    print(f"[validation] policy discard={policy['discard']} "
          f"unusable={policy['unusable']}")
    for s, v in sorted(summary["systems"].items()):
        print(f"  {s:30s} {v['pages']:4d} pages  accept {v['accept_rate']:5.1%}  "
              f"escalate {v['escalate_rate']:5.1%}  defer {v['defer_rate']:5.1%}")
    print(f"[validation] {n} escalations -> {os.path.join(ws, 'validation', 'queue.json')}")
    a = summary["audit"]
    print(f"[validation] {a['n_sampled']} accepted page(s) sampled for audit "
          f"at {a['rate']:.1%} (seed {a['seed']})")
    est = summary.get("verdicts")
    if est:
        r = est["false_accept_rate"]
        print(f"[validation] verdicts: {est['audit_n']} audited, "
              + (f"false-accept {r:.1%} (95% CI {est['ci95'][0]:.1%}-{est['ci95'][1]:.1%})"
                 if r is not None else "no audit labels yet")
              + (f" -- {est['note']}" if est.get("note") else ""))
    dfr = summary.get("deferred") or {}
    if dfr.get("n_pages"):
        why = ", ".join(f"{k} ({v})" for k, v in dfr["by_reason"].items())
        print(f"[validation] {dfr['n_pages']} page(s) deferred, awaiting a path "
              f"this pipeline does not have: {why}")


if __name__ == "__main__":
    main()
