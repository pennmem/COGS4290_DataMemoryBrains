"""Run a per-session function over many sessions, with caching and local parallelism.

The replacement for cmldask now that the notebooks run off-cluster. The pattern
is unchanged: write a function that handles ONE session and returns a dict of
arrays, then map it over a cohort. Each session's result is saved as one .npz
and skipped on the next run, and a session that raises is logged and dropped.

    import cml_parallel, SpectralHelpers as sh
    results = cml_parallel.run_sessions(sh.session_power, "FR1", INTRAC_SUBS, name="power",
                                        workers=4, freqs=freqs)

`workers=1` is a plain loop. With `workers>1` every session's EEG is downloaded
first (one approval), then that many sessions are processed at once in separate
processes; budget 3-5 GB of RAM per worker. The function must live in a .py
file (e.g. SpectralHelpers.py), not in the notebook, so worker processes can
import it. Results are cached by `name` only: delete `results/<name>/` after
changing the function's parameters.
"""

import datetime
import os
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cml_data  # noqa: E402

__all__ = ["run_sessions", "load_result", "log_error"]


def log_error(msg, log_file, exc=None):
    """Print a timestamped line (and write it, with traceback, to *log_file*)."""
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    if exc is not None:
        line += f": {type(exc).__name__}: {exc}"
    print(line, flush=True)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(line + "\n" + ("".join(traceback.format_exception(exc)) if exc else ""))


def load_result(path):
    """Read one cached session back into a dict of arrays / scalars."""
    with np.load(path, allow_pickle=True) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def _job(args):
    """One session in a worker process; errors are returned, not raised."""
    func, sub, ses, path, kwargs = args
    try:
        out = dict(func(sub, ses, **kwargs), subject=sub, session=str(ses))
        np.savez(path, **out)
        return sub, ses, None
    except Exception as e:                      # noqa: BLE001 - one bad session must not stop the run
        return sub, ses, e


def run_sessions(func, task, subjects, name, workers=1, sessions=None, cache_dir="results",
                 download_workers=8, **kwargs):
    """Apply ``func(subject, session, **kwargs)`` to every session of *subjects*.

    Returns the list of result dicts in (subject, session) order, each with
    ``subject`` and ``session`` added. Cached under ``<cache_dir>/<name>/``;
    failures are logged to ``<cache_dir>/<name>/errors.log``. *sessions* may
    be a DataFrame with subject/session columns (default: everything OpenNeuro
    lists for *task*). ``kwargs["acquisition"]`` (default bipolar) selects which
    EEG files are prefetched when ``workers > 1``.
    """
    df = cml_data.session_dataframe(task) if sessions is None else sessions
    df = df[df["subject"].isin(list(subjects))]
    pairs = list(zip(df["subject"], df["session"]))
    folder = Path(cache_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    log_file = folder / "errors.log"
    path = lambda sub, ses: folder / f"{sub}_ses-{ses}.npz"          # noqa: E731

    def report(outcome):
        sub, ses, err = outcome
        if err is not None:
            log_error(f"{sub} ses-{ses} skipped", log_file, err)
        else:
            print(f"  {sub} ses-{ses}: done", flush=True)

    todo = [(func, s, e, path(s, e), kwargs) for s, e in pairs if not path(s, e).exists()]
    if todo and workers > 1:
        cml_data.prefetch(task, sorted({s for _, s, *_ in todo}), include_timeseries=True,
                          acquisition=kwargs.get("acquisition", "bipolar"), workers=download_workers)
        print(f"{len(todo)} session(s) to compute with {workers} workers "
              f"({len(pairs) - len(todo)} cached)", flush=True)
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for outcome in pool.map(_job, todo):
                report(outcome)
    else:
        for t in todo:
            report(_job(t))

    results = [load_result(path(s, e)) for s, e in pairs if path(s, e).exists()]
    print(f"{len(results)} of {len(pairs)} sessions ({df['subject'].nunique()} subjects)")
    return results
