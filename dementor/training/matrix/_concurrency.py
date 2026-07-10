"""Retry + thread-pool dispatch scaffolding shared by the launch/upload commands."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def _retry_call(
    fn,
    *,
    attempts: int,
    base_wait: float,
    label: str,
    name: str = "",
    err_trunc: int = 120,
    sleep_verb: str = "sleep",
    max_wait: float = 60,
    indent: str = "  ",
):
    """Call ``fn()`` up to ``attempts`` times, retrying on any exception.

    On each failure prints ``{indent}[retry {label} {attempt}/{attempts}] ...``
    then sleeps ``min(max_wait, base_wait * 2 ** (attempt - 1))`` seconds before
    the next try. Returns ``fn()``'s value on the first success; re-raises the
    last exception if every attempt fails.
    """
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            wait_s = min(max_wait, base_wait * 2 ** (attempt - 1))
            name_part = f"{name} " if name else ""
            print(
                f"{indent}[retry {label} {attempt}/{attempts}] "
                f"{name_part}{type(e).__name__}: {str(e)[:err_trunc]} — {sleep_verb} {wait_s}s",
                flush=True,
            )
            time.sleep(wait_s)
    if last_err is None:  # attempts <= 0: loop never ran — avoid `raise None`
        raise RuntimeError(f"{label} failed after {attempts} attempts")
    raise last_err


def _dispatch(items, worker, *, parallel, label, key, on_success, on_error):
    """Run ``worker(item)`` over ``items``, sequentially or via a thread pool.

    When ``parallel <= 1`` results are handed to ``on_success`` in item order and
    worker exceptions propagate (matching the original inline loops). Otherwise a
    ``ThreadPoolExecutor(max_workers=parallel)`` fans the work out: results reach
    ``on_success`` in completion order, and each worker exception is logged as
    ``[worker exception] {key}`` then routed to ``on_error(key, exc)``. A
    ``[progress] done/total {label} complete`` line prints per completion.
    """
    if parallel <= 1:
        for item in items:
            on_success(worker(item))
        return
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(worker, item): key(item) for item in items}
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            done += 1
            try:
                on_success(fut.result())
            except Exception as e:
                k = futures[fut]
                print(f"  [worker exception] {k}: {e}", flush=True)
                on_error(k, e)
            print(f"[progress] {done}/{total} {label} complete", flush=True)
