#!/usr/bin/env python3
"""Canonical class -> storage bucket.

Layout class errors only matter where they change where the content ends up.
Downstream, text-like blocks go to a vector index, tables to relational storage,
figures to an object store, and running furniture is dropped.  So `heading` read
as `text` costs nothing and is not scored, while `table` read as `text` puts a
table into the vector index as unusable prose, and `text` read as `header`
deletes content with no trace anywhere.

`other` and `sidebar` map to TEXT deliberately: when a system cannot say what a
block is, keeping its content is the recoverable mistake.
"""
TEXT, TABLE, MEDIA, DISCARD = "TEXT", "TABLE", "MEDIA", "DISCARD"

BUCKET = {
    "text": TEXT, "title": TEXT, "heading": TEXT, "list": TEXT,
    "caption": TEXT, "footnote": TEXT, "formula": TEXT,
    "sidebar": TEXT, "other": TEXT,
    "table": TABLE,
    "figure": MEDIA,
    "header": DISCARD, "footer": DISCARD, "page_number": DISCARD,
    "separator": DISCARD,
}

ALL = (TEXT, TABLE, MEDIA, DISCARD)


def bucket(canonical_class):
    return BUCKET.get(canonical_class, TEXT)
