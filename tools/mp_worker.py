"""Worker function for the multiprocessing benchmark in `benchmark.ipynb`.

This has to live in its own importable module rather than a notebook cell:
`ProcessPoolExecutor`'s default "spawn" start method (macOS/Windows) pickles
a *reference* to the function you hand it and re-imports it in each worker
process. A function defined in a Jupyter cell lives in the kernel's
`__main__`, which a spawned child can't re-import -- you'd get a
`PicklingError`/`AttributeError` at call time. A plain module has no such
problem.
"""

from __future__ import annotations

import usaddress
from usaddress import RepeatedLabelError


def tag_one(address: str) -> dict | None:
    """Same policy as the notebook's `python_tag`: RepeatedLabelError -> None.

    Nothing here explicitly opens a `pycrfsuite.Tagger` -- `usaddress` does
    that itself, once, at import time. Since `ProcessPoolExecutor` workers
    are long-lived (imported once, then reused for every task), that import
    cost is paid once per worker process, not once per address -- the same
    "one per task, not one per row" shape as the Rust side's `map_init`.
    """
    try:
        tagged, _addr_type = usaddress.tag(address)
        return dict(tagged)
    except RepeatedLabelError:
        return None
