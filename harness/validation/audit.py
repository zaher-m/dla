#!/usr/bin/env python3
"""The random audit: a fixed-rate sample of ACCEPTED pages.

Without this every page a reviewer ever sees is a page that failed a check, so
every label comes from the flagged distribution and the one number worth having
-- how often an accepted page is wrong -- cannot be estimated at all.  It is the
only part of the framework that cannot be added later: pages ingested before the
sampler exists are gone.

Selection is a hash, not a shuffle.  A page's membership depends only on its own
id, the system and the seed, so it does not change when the corpus grows, when
pages are processed in a different order, or when a job is re-run.  That makes
an audit stratum accumulated over months a valid sample rather than a series of
unrelated ones.

An audit never changes a decision.  The page is accepted and written; the task
is raised alongside it.
"""
import hashlib

DEFAULTS = {"rate": 0.015, "seed": 20260904}


def sampled(page_id, system, rate, seed):
    """Is this (page, system) in the audit stratum?"""
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    key = f"{seed}|{system}|{page_id}".encode("utf8")
    h = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64) < rate


def settings(policy):
    a = dict(DEFAULTS)
    a.update((policy or {}).get("audit") or {})
    return float(a["rate"]), a["seed"]


def select(rows, system, policy):
    """Accepted pages of one system that fall in the audit stratum."""
    rate, seed = settings(policy)
    return [r for r in rows
            if r["decision"] == "accept" and sampled(r["page_id"], system, rate, seed)]
