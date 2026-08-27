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
from pathlib import Path
from xml.etree import ElementTree
from typing import Iterable, List, Optional, Sequence, Union

__all__ = ["openneuro_root", "available_subjects", "available_tasks",
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
    task = _canonical_task(task)
    ds = dataset_of(task)
    root = Path(cache) if cache else cache_dir() / ds
    root.mkdir(parents=True, exist_ok=True)

    wanted: List[tuple] = []

    # dataset-level files, needed by mne-bids to recognise the root
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

    todo = [(k, s) for k, s in wanted
            if not (root / k.split("/", 1)[1]).exists()]
    if todo and not quiet:
        total = sum(s for _, s in todo)
        print(f"{ds} ({task}): fetching {len(todo)} file(s), {_human(total)} "
              f"-> {root}")
    for key, size in todo:
        _download(key, root / key.split("/", 1)[1], size, quiet)

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
    a = p.parse_args()
    if a.list_subjects:
        subs = available_subjects(a.task)
        print(f"{len(subs)} subjects in {dataset_of(a.task)}:")
        print(", ".join(subs))
    else:
        print(openneuro_root(a.task, subject=a.subject, session=a.session,
                             include_timeseries=a.eeg, acquisition=a.acq))
