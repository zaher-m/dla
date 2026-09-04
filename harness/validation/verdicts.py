#!/usr/bin/env python3
"""What a reviewer sends back, and what it is worth.

The queue goes out and, until this existed, nothing came back.  A verdict is
appended to `<workspace>/validation/verdicts.jsonl`, one JSON object per line,
so a reviewing system can append without reading or locking the file.

The field that matters most is `frame`.  A label from the escalation queue and a
label from the random audit answer different questions, and averaging them
answers neither: escalations measure whether a flagged page was really wrong,
audits measure whether an accepted page was really right.  Only the second gives
a false-accept rate, and only if the two are never mixed.

A confirmation is a label.  A reviewer who opens a flagged page and finds
nothing wrong has produced a negative example, which is exactly what a risk
model trained only on corrections lacks -- such a model learns which pages get
flagged, not which pages are wrong.
"""
import json, math, os

SCHEMA = "dla.validation/1"
FRAMES = ("escalation", "audit")
OUTCOMES = ("confirm", "correct", "unusable")
REQUIRED = ("page_id", "system", "frame", "outcome")


def validate(v):
    """-> the verdict, normalised.  Raises ValueError on anything unusable."""
    missing = [k for k in REQUIRED if not v.get(k)]
    if missing:
        raise ValueError(f"verdict is missing {', '.join(missing)}")
    if v["frame"] not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}, got {v['frame']!r}")
    if v["outcome"] not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {v['outcome']!r}")
    if v["outcome"] == "correct" and not v.get("regions"):
        raise ValueError("outcome 'correct' needs the corrected regions")
    return {"schema": SCHEMA, **v}


def path(ws):
    return os.path.join(ws, "validation", "verdicts.jsonl")


def append(ws, verdict):
    v = validate(verdict)
    p = path(ws)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf8") as f:
        f.write(json.dumps(v, ensure_ascii=False) + "\n")
    return v


def load(ws):
    p = path(ws)
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(validate(json.loads(line)))
            except (ValueError, json.JSONDecodeError) as e:
                raise ValueError(f"{p}:{n}: {e}") from None
    return out


def wilson(k, n, z=1.96):
    """Score interval.  Correct at small n and at k=0, where the normal
    approximation returns the zero-width interval that has ended more than one
    quality programme."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def estimate(verdicts):
    """The false-accept rate, from the audit stratum alone.

    Escalation verdicts are counted separately and never folded in: they are
    drawn from flagged pages, so they say how precise the checks are, not how
    often an accepted page is wrong.
    """
    audit = [v for v in verdicts if v["frame"] == "audit"]
    esc = [v for v in verdicts if v["frame"] == "escalation"]
    n, k = len(audit), sum(1 for v in audit if v["outcome"] != "confirm")
    lo, hi = wilson(k, n)
    out = {"audit_n": n, "audit_wrong": k,
           "false_accept_rate": round(k / n, 4) if n else None,
           "ci95": [round(lo, 4), round(hi, 4)],
           "escalation_n": len(esc),
           "escalation_precision": (round(sum(1 for v in esc
                                              if v["outcome"] != "confirm") / len(esc), 4)
                                    if esc else None)}
    # With no errors seen, the rule of three is the honest headline: the true
    # rate could still be as high as 3/n and nothing observed would contradict it.
    if n and k == 0:
        out["note"] = (f"no errors in {n} audited pages: the rate could still be "
                       f"up to {3.0/n:.1%} (rule of three)")
    elif n < 100:
        out["note"] = f"{n} audited pages is too few for a usable interval"
    return out
