#!/usr/bin/env python3
"""Driver that runs Eynollah's own CLI with one compatibility shim.

`eynollah.predictor.Predictor.__init__` does:

    self.closable = ctxt.Manager().list()

The `SyncManager` returned by `Manager()` is never stored, so it can be garbage
collected before `Process.start()` pickles the Predictor to its child. When the
manager dies its POSIX semaphores are unlinked, and the child then fails with

    FileNotFoundError  ... SemLock._rebuild

Keeping a strong reference to every manager created in this process fixes it
without touching the repository. Everything else — models, flags, code path —
is Eynollah's own.
"""
import multiprocessing.context as _ctx
import sys

_KEEP = []
_orig = _ctx.BaseContext.Manager


def _manager(self, *a, **kw):
    m = _orig(self, *a, **kw)
    _KEEP.append(m)
    return m


_ctx.BaseContext.Manager = _manager

from eynollah.cli.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
