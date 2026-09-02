"""Fetch Computational Memory Lab BIDS data from OpenNeuro.

The course notebooks used to read from hard-coded rhino paths such as
``/data/LTP_BIDS/FR1``. Those only exist on the lab cluster. This module downloads
the same data from OpenNeuro, where the lab publishes it under a CC0 licence, and
returns a local BIDS root you can hand straight to ``BIDSReader`` or ``mne_bids``.

    from cml_data import openneuro_root

    root = openneuro_root("FR1", subject="R1111M", session=0)
    reader = BIDSReader(root=root, subject="R1111M", session=0, task="FR1")

Only the files you ask for are downloaded, and they are cached, so re-running a
notebook costs nothing. Behavioural data (``.tsv``) is small — a few hundred KB per
session. The EEG recordings themselves are **large** (300-700 MB per session), so
they are skipped unless you pass ``include_timeseries=True``.

Set the cache location with the ``CML_BIDS_CACHE`` environment variable; it
defaults to ``./bids_data`` next to the notebooks.

Datasets: https://memory.psych.upenn.edu/Electrophysiological_Data
"""

from __future__ import annotations

import os
import re
import sys
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree
from typing import Iterable, List, Optional, Sequence, Union

__all__ = ["openneuro_root", "get_bids_root", "dataset_root", "plan_download",
           "available_subjects", "available_tasks", "session_index", "session_dataframe", "prefetch", "approve_downloads",
           "DATASETS", "dataset_of"]

S3 = "https://s3.amazonaws.com/openneuro.org"

#: Task name -> OpenNeuro accession. Task labels are case-sensitive in BIDS.
DATASETS = {
    # intracranial EEG
    "FR1": "ds004789",       # Delayed Free Recall of Word Lists
    "catFR1": "ds004809",    # Categorised free recall
    "PAL1": "ds005059",      # Paired associates
    "pyFR": "ds004865",
    "RepFR1": "ds005411",
    # scalp EEG
    "ltpFR": "ds004395",     # PEERS experiment 1
    "ltpFR2": "ds004395",    # PEERS experiment 2
    "VFFR": "ds004395",      # PEERS verbal free-form recall
    "PEERS": "ds004395",     # alias: the whole PEERS collection
    "NICLS": "ds004706",
}

#: Files at the top of a dataset that mne-bids expects to find.
_ROOT_FILES = ("dataset_description.json", "participants.tsv",
               "participants.json", "README")

#: Extensions treated as bulk time-series and skipped by default.
_TIMESERIES_EXT = (".edf", ".fif", ".set", ".fdt", ".bdf", ".vhdr", ".eeg", ".vmrk",
                   ".nii", ".nii.gz", ".h5")


# --------------------------------------------------------------------------- #
# task-name handling
# --------------------------------------------------------------------------- #

def _canonical_task(task: str) -> str:
    """Map a user-supplied task name onto the exact BIDS label.

    The notebooks historically used inconsistent capitalisation ('ltpfr2',
    'FR1', 'PEERS'), while BIDS task labels are case-sensitive.
    """
    if task is None:
        raise ValueError("task must be given")
    for known in DATASETS:
        if str(task).lower() == known.lower():
            return known
    raise ValueError(
        f"Unknown task {task!r}. Known tasks: {', '.join(sorted(DATASETS))}"
    )


def dataset_of(task: str) -> str:
    """OpenNeuro accession (e.g. 'ds004789') hosting `task`."""
    return DATASETS[_canonical_task(task)]


# --------------------------------------------------------------------------- #
# S3 listing / download
# --------------------------------------------------------------------------- #

_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _s3_query(prefix: str, delimiter: Optional[str] = None):
    """Yield parsed <ListBucketResult> pages for `prefix`, following tokens.

    Parsed with ElementTree rather than a regex: different objects in the same
    bucket carry different child elements (older uploads have no
    <ChecksumAlgorithm>), and a positional regex silently returns nothing on the
    ones it does not match.
    """
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            query["delimiter"] = delimiter
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(f"{S3}?{urllib.parse.urlencode(query)}",
                                    timeout=120) as fh:
            tree = ElementTree.fromstring(fh.read())
        yield tree
        token_el = tree.find("s3:NextContinuationToken", _NS)
        if token_el is None or not token_el.text:
            return
        token = token_el.text


def _s3_list(prefix: str) -> List[tuple]:
    """List (key, size) under `prefix`, following continuation tokens."""
    out = []
    for tree in _s3_query(prefix):
        for contents in tree.findall("s3:Contents", _NS):
            key = contents.findtext("s3:Key", namespaces=_NS)
            size = contents.findtext("s3:Size", default="0", namespaces=_NS)
            if key:
                out.append((key, int(size)))
    return out


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def _download(key: str, dest: Path, size: int, quiet: bool) -> None:
    """Download one S3 key to `dest`, atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if not quiet:
        print(f"  downloading {dest.name} ({_human(size)})", flush=True)
    url = f"{S3}/{urllib.parse.quote(key)}"
    with urllib.request.urlopen(url, timeout=300) as src, open(part, "wb") as out:
        done = 0
        while chunk := src.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if not quiet and size > 50 << 20:
                pct = 100 * done / size
                print(f"\r    {pct:5.1f}%  {_human(done)} / {_human(size)}",
                      end="", file=sys.stdout, flush=True)
        if not quiet and size > 50 << 20:
            print(flush=True)
    part.replace(dest)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def cache_dir() -> Path:
    return Path(os.environ.get("CML_BIDS_CACHE", "bids_data")).expanduser()


def available_subjects(task: str) -> List[str]:
    """Subject labels present in the dataset hosting `task` (no 'sub-' prefix)."""
    ds = dataset_of(task)
    subs = []
    for tree in _s3_query(f"{ds}/", delimiter="/"):
        for cp in tree.findall("s3:CommonPrefixes", _NS):
            prefix = cp.findtext("s3:Prefix", default="", namespaces=_NS)
            m = re.fullmatch(rf"{re.escape(ds)}/sub-([^/]+)/", prefix)
            if m:
                subs.append(m.group(1))
    return sorted(subs)


def available_tasks(task_or_subject: str, subject: Optional[str] = None) -> List[str]:
    """Task labels available for one subject."""
    ds = dataset_of(task_or_subject)
    keys = [k for k, _ in _s3_list(f"{ds}/sub-{subject}/")]
    return sorted({m.group(1) for k in keys
                   if (m := re.search(r"task-([A-Za-z0-9]+)", k))})


def openneuro_root(
    task: str,
    subject: Optional[Union[str, Sequence[str]]] = None,
    session: Optional[Union[int, str, Sequence]] = None,
    include_timeseries: bool = False,
    acquisition: Optional[str] = None,
    datatype: Optional[str] = None,
    cache: Optional[Union[str, Path]] = None,
    quiet: bool = False,
) -> Path:
    """Materialise part of an OpenNeuro dataset locally and return its BIDS root.

    Parameters
    ----------
    task
        'FR1', 'catFR1', 'ltpFR2', ... (case-insensitive; see ``DATASETS``).
    subject
        One subject label, a list of them, or None for dataset-level files only.
    session
        Session number(s) to fetch. None fetches every session for the subject.
    include_timeseries
        Download the actual recordings (.edf etc). These are 300-700 MB per
        session, so the default is False — you get the events, channels, and
        electrode tables only, which is all most of the notebooks need.
    acquisition
        For intracranial data, 'bipolar' or 'monopolar'. Restricts which
        recordings are fetched; ignored unless ``include_timeseries``.
    datatype
        'ieeg', 'eeg', or 'beh' to restrict to one modality.

    Returns
    -------
    Path to a local directory laid out as valid BIDS.
    """
    root, todo = plan_download(task, subject=subject, session=session,
                               include_timeseries=include_timeseries,
                               acquisition=acquisition, datatype=datatype,
                               cache=cache)
    root.mkdir(parents=True, exist_ok=True)
    if todo and not quiet:
        total = sum(s for _, s in todo)
        print(f"{dataset_of(task)} ({_canonical_task(task)}): fetching "
              f"{len(todo)} file(s), {_human(total)} -> {root}")
    for key, size in todo:
        _download(key, root / key.split("/", 1)[1], size, quiet)
    return root


def plan_download(
    task: str,
    subject: Optional[Union[str, Sequence[str]]] = None,
    session: Optional[Union[int, str, Sequence]] = None,
    include_timeseries: bool = False,
    acquisition: Optional[str] = None,
    datatype: Optional[str] = None,
    cache: Optional[Union[str, Path]] = None,
) -> tuple:
    """Work out what would be downloaded, without downloading anything.

    Returns ``(root, todo)`` where `todo` is a list of ``(key, size)`` for files
    not already cached. Used to show the download size before asking permission.
    """
    task = _canonical_task(task)
    ds = dataset_of(task)
    root = Path(cache) if cache else cache_dir() / ds

    wanted: List[tuple] = []

    # Dataset-level files, needed by mne-bids to recognise the root. Listing the
    # whole dataset prefix to find four filenames costs ~10 s on a big dataset,
    # so only look when at least one is actually missing.
    if any(not (root / n).exists() for n in _ROOT_FILES):
        for key, size in _s3_list(f"{ds}/"):
            name = key.split("/")[-1]
            if key.count("/") == 1 and name in _ROOT_FILES:
                wanted.append((key, size))

    subjects = ([subject] if isinstance(subject, str)
                else list(subject) if subject is not None else [])
    sessions = ([session] if isinstance(session, (int, str))
                else list(session) if session is not None else None)

    for sub in subjects:
        sub = str(sub).replace("sub-", "")
        prefixes = ([f"{ds}/sub-{sub}/ses-{str(s).replace('ses-', '')}/"
                     for s in sessions] if sessions else [f"{ds}/sub-{sub}/"])
        found_any = False
        for prefix in prefixes:
            for key, size in _s3_list(prefix):
                found_any = True
                name = key.split("/")[-1]
                # keep only this task's files (files with no task label, e.g.
                # electrodes.tsv or scans.tsv, are always kept)
                m = re.search(r"task-([A-Za-z0-9]+)", name)
                if m and m.group(1) != task and task != "PEERS":
                    continue
                if datatype and f"/{datatype}/" not in key:
                    continue
                is_bulk = name.endswith(_TIMESERIES_EXT)
                if is_bulk and not include_timeseries:
                    continue
                if is_bulk and acquisition and f"acq-{acquisition}" not in name:
                    continue
                wanted.append((key, size))
        if not found_any:
            raise FileNotFoundError(
                f"No files for subject {sub!r} in {ds} ({task}). "
                f"Use available_subjects({task!r}) to list valid subjects."
            )

    seen, todo = set(), []
    for key, size in wanted:
        if key in seen:
            continue
        seen.add(key)
        if not (root / key.split("/", 1)[1]).exists():
            todo.append((key, size))
    return root, todo


# --------------------------------------------------------------------------- #
# interactive entry point used by the notebooks
# --------------------------------------------------------------------------- #

#: Set once the user approves downloading in this process. A single analysis can
#: touch thousands of sessions; asking per session is unusable.
_SESSION_APPROVED = False


def approve_downloads(yes: bool = True) -> None:
    """Pre-approve (or revoke) downloads for the rest of this process."""
    global _SESSION_APPROVED
    _SESSION_APPROVED = bool(yes)


#: Where each dataset lives on the lab cluster, for people working on rhino.
RHINO_ROOTS = {
    "ds004789": "/data/LTP_BIDS/FR1",
    "ds004809": "/data/LTP_BIDS/catFR1",
    "ds004395": "/data/LTP_BIDS",
    "ds004706": "/data/LTP_BIDS",
}


@lru_cache(maxsize=None)
def session_index(task: str) -> tuple:
    """Every (subject, session) that exists for `task`, straight from OpenNeuro.

    Answers "what sessions were run?" without downloading anything. A local
    directory scan cannot answer that on a partial cache -- it only sees what
    has already been fetched.
    """
    task = _canonical_task(task)
    ds = dataset_of(task)
    pat = re.compile(rf"sub-([^/_]+)/ses-([^/_]+)/.*task-{re.escape(task)}",
                     re.IGNORECASE)
    pairs = set()
    for key, _ in _s3_list(f"{ds}/"):
        m = pat.search(key)
        if m:
            pairs.add((m.group(1), m.group(2)))
    # tuple + lru_cache: listing a whole dataset takes ~10 s and several cells
    # ask for the same index.
    return tuple(sorted(pairs))


def prefetch(task: str, subjects=None, sessions=None, workers: int = 12,
             include_timeseries: bool = False, acquisition=None) -> Path:
    """Download many sessions concurrently.

    The notebooks fetch one session at a time, which is fine for a handful but
    slow for a multi-subject analysis (~1.2 s each, sequentially). This grabs
    the whole set in parallel instead. Safe to re-run: cached files are skipped.

        python cml_data.py ltpFR2 --prefetch
    """
    task = _canonical_task(task)
    ds = dataset_of(task)
    root = cache_dir() / ds
    root.mkdir(parents=True, exist_ok=True)

    pairs = session_index(task)
    if subjects is not None:
        want = {str(s).replace("sub-", "") for s in subjects}
        pairs = [p for p in pairs if p[0] in want]
    if sessions is not None:
        want_s = {str(s).replace("ses-", "") for s in sessions}
        pairs = [p for p in pairs if p[1] in want_s]

    todo = []
    seen = set()
    for key, size in _s3_list(f"{ds}/"):
        name = key.split("/")[-1]
        if key.count("/") == 1 and name in _ROOT_FILES:
            if not (root / name).exists():
                todo.append((key, size))
            continue
        m = re.match(rf"{re.escape(ds)}/sub-([^/]+)/ses-([^/]+)/", key)
        if not m or (m.group(1), m.group(2)) not in set(pairs):
            continue
        tm = re.search(r"task-([A-Za-z0-9]+)", name)
        if tm and tm.group(1) != task:
            continue
        if name.endswith(_TIMESERIES_EXT):
            if not include_timeseries:
                continue
            if acquisition and f"acq-{acquisition}" not in name:
                continue
        rel = key.split("/", 1)[1]
        if key not in seen and not (root / rel).exists():
            seen.add(key)
            todo.append((key, size))

    if not todo:
        print(f"{ds} ({task}): {len(pairs)} session(s) already cached")
        return root
    total = sum(sz for _, sz in todo)
    print(f"{ds} ({task}): downloading {len(todo)} files ({_human(total)}) "
          f"for {len(pairs)} sessions, {workers} at a time")

    from concurrent.futures import ThreadPoolExecutor
    done = [0]
    def grab(item):
        key, size = item
        _download(key, root / key.split("/", 1)[1], size, quiet=True)
        done[0] += 1
        if done[0] % 100 == 0:
            print(f"  {done[0]}/{len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(grab, todo))
    print(f"  done: {len(todo)} files")
    return root


def session_dataframe(task: str):
    """Every (subject, session) for `task`, as a DataFrame.

    The BIDS-standard way to ask "what sessions exist?" is to scan the study
    folder. That only works when you have the *whole* study on disk, as on the
    lab cluster. If you are working from downloaded data your folder holds only
    what you have fetched so far, so a scan under-reports. This asks OpenNeuro
    instead and always gives the full picture.

        df = session_dataframe("VFFR")
        len(df)                      # sessions run
        df["subject"].nunique()      # subjects who ran at least one
    """
    import pandas as pd
    task = _canonical_task(task)
    pairs = session_index(task)
    return pd.DataFrame({"subject": [s for s, _ in pairs],
                         "task": [task] * len(pairs),
                         "session": [e for _, e in pairs]}).sort_values(
                             ["subject", "session"]).reset_index(drop=True)


def dataset_root(task: str) -> Path:
    """Where `task` lives, without touching the network.

    Returns the lab-cluster path when it exists, otherwise the local OpenNeuro
    cache directory for that dataset. Use this for module-level constants; use
    ``get_bids_root`` when you actually need the files present.
    """
    ds = dataset_of(task)
    rhino = Path(RHINO_ROOTS.get(ds, ""))
    if str(rhino) != "." and rhino.exists():
        return rhino
    return cache_dir() / ds


def _have_locally(root: Path, task: str, subject, session,
                  include_timeseries: bool, acquisition: Optional[str]) -> bool:
    """True if every requested subject/session already has files on disk.

    Purely local; makes no network call. Conservative -- returns False whenever
    it cannot prove the data is present, so the worst case is a needless listing.
    """
    if subject is None or not root.is_dir():
        return False
    if any(not (root / n).exists() for n in ("dataset_description.json",)):
        return False
    subjects = [subject] if isinstance(subject, str) else list(subject)
    sessions = ([session] if isinstance(session, (int, str))
                else list(session) if session is not None else None)
    if sessions is None:
        return False          # "all sessions" -- cannot verify without listing
    for sub in subjects:
        sub = str(sub).replace("sub-", "")
        for ses in sessions:
            ses = str(ses).replace("ses-", "")
            d = root / f"sub-{sub}" / f"ses-{ses}"
            if not d.is_dir():
                return False
            found = list(d.rglob(f"*task-{task}*"))
            if not found:
                return False
            if include_timeseries:
                bulk = [f for f in found if f.name.endswith(_TIMESERIES_EXT)]
                if acquisition:
                    bulk = [f for f in bulk if f"acq-{acquisition}" in f.name]
                if not bulk:
                    return False
    return True


def _prompt(question: str, default: str) -> Optional[str]:
    """Ask the user; return None if there is no one to ask."""
    try:
        answer = input(question).strip()
    except Exception:          # nbconvert, cron, piped stdin, ...
        return None
    return answer or default


def get_bids_root(
    task: str,
    subject: Optional[Union[str, Sequence[str]]] = None,
    session: Optional[Union[int, str, Sequence]] = None,
    include_timeseries: bool = False,
    acquisition: Optional[str] = None,
    datatype: Optional[str] = None,
    rhino_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Return a BIDS root, asking where the data should come from.

    Offers the rhino copy when it is present, otherwise downloads from OpenNeuro
    — showing how much it is about to fetch and asking before it starts. Files
    already cached are never re-downloaded, and you are only asked when there is
    actually something to fetch.

    Non-interactive runs (nbconvert, scripts) must choose in advance:

        CML_DATA_SOURCE=rhino|openneuro   skip the "where from?" question
        CML_AUTO_APPROVE=1                skip the download confirmation
    """
    global _SESSION_APPROVED
    task = _canonical_task(task)
    ds = dataset_of(task)
    rhino = Path(rhino_root) if rhino_root else Path(RHINO_ROOTS.get(ds, ""))
    have_rhino = str(rhino) != "." and rhino.exists()

    source = os.environ.get("CML_DATA_SOURCE", "").strip().lower()
    if source not in ("rhino", "openneuro"):
        if have_rhino:
            print(f"Where should this notebook read {task} data from?")
            print(f"  [1] the lab cluster   {rhino}   (detected)")
            print(f"  [2] OpenNeuro {ds}    (downloads to {cache_dir() / ds})")
            choice = _prompt("Choice [1]: ", "1")
            if choice is None:
                raise RuntimeError(
                    "Cannot ask which data source to use in a non-interactive "
                    "run. Set CML_DATA_SOURCE=rhino or =openneuro."
                )
            source = "rhino" if choice.startswith("1") else "openneuro"
        else:
            source = "openneuro"

    if source == "rhino":
        if not have_rhino:
            raise FileNotFoundError(
                f"{rhino} not found — you are probably not on the cluster. "
                f"Set CML_DATA_SOURCE=openneuro to download instead."
            )
        return rhino

    # Fast path: if the requested sessions are already on disk, do not touch the
    # network at all. Without this, a loop over N sessions pays a full S3
    # listing per session even when everything is cached.
    if _have_locally(dataset_root(task), task, subject, session,
                     include_timeseries, acquisition):
        return dataset_root(task)

    root, todo = plan_download(task, subject=subject, session=session,
                               include_timeseries=include_timeseries,
                               acquisition=acquisition, datatype=datatype)
    if not todo:
        return root

    total = sum(s for _, s in todo)
    first_time = not (_SESSION_APPROVED or os.environ.get("CML_AUTO_APPROVE"))
    if first_time or total > 100 << 20:
        print(f"\nOpenNeuro {ds} ({task}) — need {len(todo)} file(s), "
              f"{_human(total)} to download.")
        print(f"  destination: {root}")
        if total > 100 << 20:
            print("  (this is the actual EEG recording; it is cached "
                  "afterwards, so you only pay this once)")

    if not (_SESSION_APPROVED or os.environ.get("CML_AUTO_APPROVE")):
        print("  Approving once covers every download for the rest of this "
              "session (until the kernel restarts).")
        answer = _prompt(f"Download {_human(total)}? [y/N]: ", "n")
        if answer is None:
            raise RuntimeError(
                f"Need to download {_human(total)} but cannot ask for approval "
                f"in a non-interactive run. Set CML_AUTO_APPROVE=1 to allow it, "
                f"or pre-download with:\n"
                f"    python cml_data.py {task} --prefetch"
            )
        if not answer.lower().startswith("y"):
            raise RuntimeError(
                "Download declined. Re-run this cell and answer 'y', or fetch "
                "the data yourself with cml_data.py (see the README)."
            )
        _SESSION_APPROVED = True   # ask once per run, not once per session-file

    root.mkdir(parents=True, exist_ok=True)
    for key, size in todo:
        _download(key, root / key.split("/", 1)[1], size, quiet=True)
    return root


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("task")
    p.add_argument("--subject", "-s", nargs="*", default=None)
    p.add_argument("--session", nargs="*", default=None)
    p.add_argument("--eeg", action="store_true", help="also download recordings")
    p.add_argument("--acq", default=None, choices=[None, "bipolar", "monopolar"])
    p.add_argument("--list-subjects", action="store_true")
    p.add_argument("--prefetch", action="store_true",
                   help="download all sessions for this task, in parallel")
    a = p.parse_args()
    if a.prefetch:
        prefetch(a.task, subjects=a.subject, sessions=a.session,
                 include_timeseries=a.eeg, acquisition=a.acq)
    elif a.list_subjects:
        subs = available_subjects(a.task)
        print(f"{len(subs)} subjects in {dataset_of(a.task)}:")
        print(", ".join(subs))
    else:
        print(openneuro_root(a.task, subject=a.subject, session=a.session,
                             include_timeseries=a.eeg, acquisition=a.acq))
