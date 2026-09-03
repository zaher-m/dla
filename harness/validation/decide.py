#!/usr/bin/env python3
"""Per-page decision: a deterministic veto, then a score.

The two stages never interfere, and that separation is the whole design.

  The veto is deterministic, label-free and permanent.  A blocking check fires,
  or a blocking check could not run, and the page escalates -- whatever any
  score says.  This is the part that carries authority precisely because it
  never needed labels to earn it.

  The score is a hand-weighted scorecard today and a calibrated model later.
  Its weights are invented.  It cannot reject a page and by default it cannot
  escalate one on its own either; it orders the queue and nothing more.  When
  gold data exists, `Scorecard` is replaced in place by a model with the same
  interface and the rest of the system does not change.

Three things are configuration rather than code, because each is a policy the
pipeline may revise without the checks being wrong:

  What the downstream does with DISCARD regions.  If they are archived where a
  reviewer can recover them, deleting a paragraph by mislabelling it a header is
  recoverable and does not block; if they are dropped at write time, it is
  permanent and does.  The check measures the same defect either way, so the
  severity belongs to the policy and not to the check.

  What happens to a page the born-digital path cannot process at all.  Today it
  is deferred, because a layout review of a page whose text is not extractable
  produces nothing this pipeline can act on.  When the scanned path exists this
  becomes a route, not a dead end.

  Where the risk thresholds sit.  Both are null until labels exist, which is the
  honest setting: an invented weight sum is not grounds for discarding a page.
"""
import os

import yaml

from validation.checks import BLOCK, MAJOR, ADV

ACCEPT, ESCALATE, REJECT, DEFER = "accept", "escalate", "reject", "defer"

DEFAULTS = {
    # archive: DISCARD regions are kept and recoverable.  drop: deleted at write
    # time.  The severity table below is keyed on this.
    "discard": "archive",
    "severity": {
        "archive": {"C6-05": MAJOR, "C6-06": MAJOR},
        "drop": {},
    },
    # A page the born-digital path cannot process.  `defer` parks it for the
    # scanned pipeline; `escalate` sends it to a human for full annotation.
    "unusable": "defer",
    "risk": {
        # null: the scorecard decides nothing on its own.  Set these only when a
        # calibrated model replaces it and the numbers mean something.
        "escalate": None,
        "reject": None,
    },
    "weights": {
        "severity": {MAJOR: 0.25, ADV: 0.05},
        "unverifiable": 0.15,
        "orphan_rate": 1.0,
        "cross_merge_rate": 0.6,
        "bucket_move_rate": 0.4,
    },
    # Which reviewer task a family raises.  E5 (committee) and E7 (audit) are
    # not raised by checks: one needs a committee, the other is a fixed-rate
    # sample of accepted pages, and both are assigned outside this function.
    # C5 duplication is a boundary fix (delete the repeat).  C7 sanity means
    # the layout is malformed rather than merely wrong, so the reviewer looks at
    # the whole page, which is the E6 task even though nothing here is novel.
    "tasks": {"C1": "E1", "C2": "E2", "C3": "E2", "C4": "E3",
              "C5": "E2", "C6": "E4", "C7": "E6", "C8": "E6"},
    # Precedence when several fire.  One page is one task, and the task chosen
    # is the largest piece of work among those raised: recovering content the
    # model never emitted (E1) subsumes fixing a boundary (E2), which subsumes
    # reordering (E3), which subsumes picking a bucket (E4).  E6 is a full
    # annotation and is only reached when nothing else applies, since a page
    # with a concrete defect is better reviewed as that defect.
    "task_order": ["E1", "E2", "E3", "E4", "E6"],
}


def load_policy(path=None):
    """Read the `policy:` block, falling back to DEFAULTS field by field."""
    path = path or os.environ.get("DLA_CHECKS_CONFIG")
    if not path:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(here, "config", "checks.yaml")
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    try:
        with open(path, encoding="utf8") as f:
            got = (yaml.safe_load(f) or {}).get("policy") or {}
    except FileNotFoundError:
        got = {}
    for k, v in got.items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k].update(v)
        else:
            p[k] = v
    if p["discard"] not in p["severity"]:
        raise ValueError(f"policy.discard={p['discard']!r} has no severity table")
    return p


def severity(cid, base, policy):
    """The severity of a finding under the current downstream policy.

    A check's registered severity says what the defect is when nothing catches
    it.  The override says what the pipeline actually does about it.
    """
    sev = policy["severity"][policy["discard"]].get(cid, base)
    if sev not in (BLOCK, MAJOR, ADV):
        raise ValueError(f"policy severity for {cid} is {sev!r}")
    return sev


def features(res, cmp_=None, policy=None):
    """The inputs to the score: counts of what fired, and the three error-model
    rates when a reference comparison is available."""
    p = policy or load_policy()
    sev = [severity(f["id"], f["severity"], p) for f in res["findings"]]
    out = {
        "n_block": sum(1 for s in sev if s == BLOCK),
        "n_major": sum(1 for s in sev if s == MAJOR),
        "n_adv": sum(1 for s in sev if s == ADV),
        # A non-blocking check that could not run is weak evidence of a page the
        # pipeline does not understand; a blocking one is a veto, not a feature.
        "unverifiable": sum(1 for s in res["skipped"]
                            if severity(s["id"], s["severity"], p) != BLOCK),
        "families": len(res["families"]),
    }
    for k in ("orphan_rate", "cross_merge_rate", "bucket_move_rate"):
        out[k] = float((cmp_ or {}).get(k) or 0.0)
    return out


class Scorecard:
    """Hand-weighted sum of the features, clipped to [0, 1].

    The weights are invented.  There is no corpus behind them and no calibration
    in front of them, so the number is an ordering, not a probability -- it is
    fit for sorting a review queue and unfit for deciding anything.  It is here
    so the interface exists before the model that fills it.
    """

    def __init__(self, weights):
        self.w = weights

    def __call__(self, f):
        w = self.w
        s = (w["severity"][MAJOR] * f["n_major"]
             + w["severity"][ADV] * f["n_adv"]
             + w["unverifiable"] * f["unverifiable"]
             + w["orphan_rate"] * f["orphan_rate"]
             + w["cross_merge_rate"] * f["cross_merge_rate"]
             + w["bucket_move_rate"] * f["bucket_move_rate"])
        return round(min(max(s, 0.0), 1.0), 4)


def scorer(policy=None):
    """The active score function.  A calibrated model replaces this and nothing
    else: same call signature, same output range, better numbers."""
    p = policy or load_policy()
    return Scorecard(p["weights"])


def _task(findings, policy):
    """One page, one task: the largest piece of work among those raised."""
    want = {policy["tasks"].get(f["id"].split("-")[0]) for f in findings}
    for t in policy["task_order"]:
        if t in want:
            return t
    return None


def decide(res, route, cmp_=None, policy=None, score=None):
    """Accept, escalate, reject or defer one page.

    `res` is `checks.run(...)`, `route` is the router's verdict, `cmp_` is the
    optional reference comparison.  Nothing here reads a PDF or a model: the
    decision is a pure function of evidence already gathered, so it can be
    replayed against a stored result when a threshold moves.
    """
    p = policy or load_policy()
    score = score or scorer(p)
    fs = [dict(f, severity=severity(f["id"], f["severity"], p)) for f in res["findings"]]
    fx = features(res, cmp_, p)
    risk = score(fx)
    out = {"decision": None, "task": None, "reason": None, "risk": risk,
           "page_kind": route["page_kind"], "psr_trust": route["psr_trust"],
           "direction": route["direction"], "features": fx,
           "findings": sorted(fs, key=lambda f: (f["severity"] != BLOCK,
                                                 f["severity"] != MAJOR, f["id"])),
           "availability": {"unverifiable": [s["id"] for s in res["skipped"]],
                            "inapplicable": [s["id"] for s in res["inapplicable"]]}}
    if cmp_:
        out["metrics"] = {k: cmp_.get(k) for k in
                          ("orphan_rate", "cross_merge_rate", "bucket_move_rate",
                           "grouping_recall", "order_tau")}

    # --- veto -------------------------------------------------------------
    if route["psr_trust"] == "unusable":
        why = ("scanned page" if route["page_kind"] == "scanned"
               else "no usable text layer")
        if p["unusable"] == "defer":
            return {**out, "decision": DEFER,
                    "reason": f"{why}: outside the born-digital path"}
        return {**out, "decision": ESCALATE, "task": "E6",
                "reason": f"{why}: the page cannot be verified"}

    blocking = [f for f in fs if f["severity"] == BLOCK]
    if blocking:
        return {**out, "decision": ESCALATE,
                "task": _task(blocking, p) or "E6",
                "reason": blocking[0]["message"]}

    unver = [s for s in res["skipped"] if severity(s["id"], s["severity"], p) == BLOCK]
    if unver:
        ids = ", ".join(sorted(s["id"] for s in unver))
        return {**out, "decision": ESCALATE, "task": "E6",
                "reason": f"incomplete evidence: {ids} could not run "
                          f"({unver[0]['reason']})"}

    # --- score ------------------------------------------------------------
    t_rej, t_esc = p["risk"]["reject"], p["risk"]["escalate"]
    if t_rej is not None and risk >= t_rej:
        return {**out, "decision": REJECT, "reason": f"risk {risk:.2f}"}
    if t_esc is not None and risk >= t_esc:
        return {**out, "decision": ESCALATE, "task": _task(fs, p) or "E6",
                "reason": f"risk {risk:.2f}: " +
                          "; ".join(f["message"] for f in fs[:2])}
    return {**out, "decision": ACCEPT,
            "reason": f"{len(fs)} non-blocking finding(s)" if fs else "clean"}


# --- self test -------------------------------------------------------------
#
# The point of the policy block is that changing one line of configuration
# changes what the pipeline does about a defect, without a check being edited.
# That property is worth a guard, because it silently stops holding the moment
# a severity is hardcoded back into a decision.

def _selftest():
    base = load_policy()
    res = {"findings": [{"id": "C6-05", "severity": BLOCK, "regions": [3],
                         "message": "3 regions inside a text column are marked "
                                    "header or footer"}],
           "skipped": [], "inapplicable": [], "families": ["C6"]}
    route = {"page_kind": "born_digital", "psr_trust": "full", "direction": "rtl"}

    keep = decide(res, route, policy={**base, "discard": "archive"})
    assert keep["decision"] == ACCEPT, keep
    assert keep["findings"][0]["severity"] == MAJOR

    drop = decide(res, route, policy={**base, "discard": "drop"})
    assert drop["decision"] == ESCALATE and drop["task"] == "E4", drop
    assert drop["findings"][0]["severity"] == BLOCK

    # the registered severity is untouched by either run: the policy decides the
    # consequence, the check keeps describing the defect
    assert res["findings"][0]["severity"] == BLOCK

    # a scan is parked, not queued, while the scanned path does not exist
    scan = decide({"findings": [], "skipped": [], "inapplicable": [],
                   "families": []},
                  {"page_kind": "scanned", "psr_trust": "unusable",
                   "direction": "ltr"}, policy=base)
    assert scan["decision"] == DEFER and scan["task"] is None, scan
    esc = decide({"findings": [], "skipped": [], "inapplicable": [],
                  "families": []},
                 {"page_kind": "scanned", "psr_trust": "unusable",
                  "direction": "ltr"}, policy={**base, "unusable": "escalate"})
    assert esc["decision"] == ESCALATE and esc["task"] == "E6", esc

    # an unavailable blocking check escalates.  never accepts.
    unv = decide({"findings": [], "families": [], "inapplicable": [],
                  "skipped": [{"id": "C1-04", "severity": BLOCK,
                               "reason": "no bands"}]},
                 {"page_kind": "born_digital", "psr_trust": "degraded",
                  "direction": "ltr"}, policy=base)
    assert unv["decision"] == ESCALATE and "C1-04" in unv["reason"], unv

    # with both thresholds null the scorecard cannot decide anything
    noisy = {"findings": [{"id": f"C5-0{i}", "severity": MAJOR, "regions": [],
                           "message": "m"} for i in (1, 3)],
             "skipped": [], "inapplicable": [], "families": ["C5"]}
    assert decide(noisy, route, policy=base)["decision"] == ACCEPT
    armed = decide(noisy, route,
                   policy={**base, "risk": {"escalate": 0.4, "reject": None}})
    assert armed["decision"] == ESCALATE and armed["task"] == "E2", armed
    print("decide: ok")


if __name__ == "__main__":
    _selftest()
