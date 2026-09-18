"""Fetch Computational Memory Lab BIDS data from OpenNeuro (or use the rhino copy).

    from cml_data import get_bids_root
    root = get_bids_root("FR1", subject="R1111M", session=0)          # events/channels only
    root = get_bids_root("FR1", subject="R1111M", session=0,          # + the EEG recording
                         include_timeseries=True, acquisition="bipolar")
    reader = BIDSReader(root=root, subject="R1111M", session="0", task="FR1")

Files are cached under ./bids_data (or $CML_BIDS_CACHE) and never re-downloaded.
EEG recordings are 300-700 MB per session and are only fetched on request.
You are asked once per run before anything is downloaded; non-interactive runs
set CML_AUTO_APPROVE=1 (and CML_DATA_SOURCE=rhino|openneuro if both exist).
CML_DATA_SOURCE=local treats the cache as a complete dataset (no network) --
used with the simulated sessions of cml_sim.py. CML_INTRAC_SUBS=A,B,C overrides
the cohort for the same purpose.

    python cml_data.py FR1 --list-subjects
    python cml_data.py FR1 --subject R1111M --session 0 --eeg --acq bipolar
"""

import os
import re
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree

__all__ = ["get_bids_root", "dataset_root", "plan_download", "prefetch", "session_dataframe",
           "available_subjects", "available_tasks", "dataset_of",
           "DATASETS", "INTRAC_SUBS", "INTRAC_SUBS_SMALL"]

S3 = "https://s3.amazonaws.com/openneuro.org"

#: task -> OpenNeuro accession (task labels are case-sensitive in BIDS)
DATASETS = {
    "FR1": "ds004789", "catFR1": "ds004809", "PAL1": "ds005059", "pyFR": "ds004865",   # intracranial
    "RepFR1": "ds005411",
    "ltpFR": "ds004395", "ltpFR2": "ds004395", "VFFR": "ds004395", "PEERS": "ds004395",  # scalp (PEERS)
    "NICLS": "ds004706",
}

#: where each dataset lives on the lab cluster
RHINO_ROOTS = {"ds004789": "/data/LTP_BIDS/FR1", "ds004809": "/data/LTP_BIDS/catFR1",
               "ds004395": "/data/LTP_BIDS", "ds004706": "/data/LTP_BIDS"}

#: The intracranial cohort used from Module 10 onward: >= 3 FR1 sessions with bipolar
#: and monopolar EEG on OpenNeuro, recall >= 28%, bipolar pairs in both temporal and
#: frontal cortex. Sessions are numbered from 0.
INTRAC_SUBS = ['R1060M', 'R1061T', 'R1065J', 'R1066P', 'R1077T',
               'R1113T', 'R1145J', 'R1151E', 'R1154D', 'R1158T',
               'R1168T', 'R1195E', 'R1217T', 'R1308T', 'R1309M',
               'R1316T', 'R1337E', 'R1341T', 'R1395M', 'R1441T']
if os.environ.get("CML_INTRAC_SUBS"):
    INTRAC_SUBS = os.environ["CML_INTRAC_SUBS"].split(",")
INTRAC_SUBS_SMALL = INTRAC_SUBS[:3]     # develop on these before running the whole cohort

_ROOT_FILES = ("dataset_description.json", "participants.tsv", "participants.json", "README")
_TIMESERIES_EXT = (".edf", ".bdf", ".fif", ".set", ".fdt", ".vhdr", ".eeg", ".vmrk", ".nii", ".nii.gz", ".h5")
_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_APPROVED = False      # set once the user says yes to downloading in this process


# ----- names ------------------------------------------------------------------

def _canonical_task(task):
    """'ltpfr2' -> 'ltpFR2' (BIDS task labels are case-sensitive)."""
    for known in DATASETS:
        if str(task).lower() == known.lower():
            return known
    raise ValueError(f"Unknown task {task!r}. Known: {', '.join(DATASETS)}")


def dataset_of(task):
    """OpenNeuro accession hosting `task`, e.g. 'ds004789'."""
    return DATASETS[_canonical_task(task)]


def cache_dir():
    return Path(os.environ.get("CML_BIDS_CACHE", "bids_data")).expanduser()


def dataset_root(task):
    """Where `task` lives: the rhino copy if present, else the local cache (no network)."""
    rhino = Path(RHINO_ROOTS.get(dataset_of(task), "/nonexistent"))
    return rhino if rhino.exists() else cache_dir() / dataset_of(task)


# ----- S3 listing / download --------------------------------------------------

def _s3_query(prefix, delimiter=None):
    """Yield parsed listing pages for `prefix`, following continuation tokens."""
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            query["delimiter"] = delimiter
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(f"{S3}?{urllib.parse.urlencode(query)}", timeout=120) as fh:
            tree = ElementTree.fromstring(fh.read())
        yield tree
        token = tree.findtext("s3:NextContinuationToken", namespaces=_NS)
        if not token:
            return


def _s3_list(prefix):
    """[(key, size), ...] under `prefix`."""
    return [(c.findtext("s3:Key", namespaces=_NS), int(c.findtext("s3:Size", default="0", namespaces=_NS)))
            for tree in _s3_query(prefix) for c in tree.findall("s3:Contents", _NS)]


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def _download(key, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(f"{S3}/{urllib.parse.quote(key)}", timeout=300) as src, open(part, "wb") as out:
        while chunk := src.read(1 << 20):
            out.write(chunk)
    part.replace(dest)


def _prompt(question, default):
    """Ask the user; None when there is nobody to ask (nbconvert, scripts)."""
    try:
        return input(question).strip() or default
    except Exception:
        return None


def _approve(todo, task):
    """Ask once per process before downloading; honours CML_AUTO_APPROVE."""
    global _APPROVED
    if _APPROVED or os.environ.get("CML_AUTO_APPROVE") or not todo:
        return
    total = _human(sum(s for _, s in todo))
    print(f"OpenNeuro {dataset_of(task)} ({task}): {len(todo)} file(s), {total} to download -> {cache_dir() / dataset_of(task)}")
    answer = _prompt(f"Download {total}? (approving once covers this whole run) [y/N]: ", "n")
    if answer is None:
        raise RuntimeError("cannot ask for download approval in a non-interactive run; set CML_AUTO_APPROVE=1")
    if not answer.lower().startswith("y"):
        raise RuntimeError("download declined")
    _APPROVED = True


# ----- what exists ------------------------------------------------------------

def available_subjects(task):
    """Subject labels in the dataset hosting `task`."""
    ds = dataset_of(task)
    subs = [m.group(1) for tree in _s3_query(f"{ds}/", delimiter="/")
            for cp in tree.findall("s3:CommonPrefixes", _NS)
            if (m := re.fullmatch(rf"{ds}/sub-([^/]+)/", cp.findtext("s3:Prefix", default="", namespaces=_NS)))]
    return sorted(subs)


def available_tasks(task, subject):
    """Task labels one subject has in the dataset hosting `task`."""
    keys = [k for k, _ in _s3_list(f"{dataset_of(task)}/sub-{subject}/")]
    return sorted({m.group(1) for k in keys if (m := re.search(r"task-([A-Za-z0-9]+)", k))})


@lru_cache(maxsize=None)
def session_dataframe(task):
    """Every (subject, session) of `task` on OpenNeuro as a DataFrame (a local scan
    would only see what has been downloaded so far)."""
    import pandas as pd
    task = _canonical_task(task)
    pat = re.compile(rf"sub-([^/_]+)/ses-([^/_]+)/.*task-{task}", re.IGNORECASE)
    if os.environ.get("CML_DATA_SOURCE", "").lower() == "local":         # scan the (complete) local copy
        keys = [str(f.relative_to(cache_dir() / dataset_of(task))) for f in (cache_dir() / dataset_of(task)).rglob("*_beh.tsv")]
    else:
        keys = [k for k, _ in _s3_list(f"{dataset_of(task)}/")]
    pairs = sorted({(m.group(1), m.group(2)) for k in keys if (m := pat.search(k))})
    return pd.DataFrame(pairs, columns=["subject", "session"]).assign(task=task)


# ----- fetching ---------------------------------------------------------------

def plan_download(task, subject=None, session=None, include_timeseries=False, acquisition=None, datatype=None):
    """(root, [(key, size), ...]) of the files not yet cached for these subjects/sessions."""
    task = _canonical_task(task)
    ds = dataset_of(task)
    root = cache_dir() / ds
    subjects = [subject] if isinstance(subject, str) else list(subject or [])
    sessions = [session] if isinstance(session, (int, str)) else session

    wanted = []
    if any(not (root / n).exists() for n in _ROOT_FILES):        # dataset-level files mne-bids needs
        wanted += [(k, s) for k, s in _s3_list(f"{ds}/") if k.count("/") == 1 and k.split("/")[-1] in _ROOT_FILES]
    for sub in subjects:
        sub = str(sub).replace("sub-", "")
        prefixes = [f"{ds}/sub-{sub}/ses-{str(s).replace('ses-', '')}/" for s in sessions] if sessions else [f"{ds}/sub-{sub}/"]
        keys = [kv for p in prefixes for kv in _s3_list(p)]
        if not keys:
            raise FileNotFoundError(f"no files for subject {sub!r} in {ds} ({task}); see available_subjects({task!r})")
        for key, size in keys:
            name = key.split("/")[-1]
            m = re.search(r"task-([A-Za-z0-9]+)", name)
            if m and m.group(1) != task and task != "PEERS":      # files with no task label (electrodes.tsv) are kept
                continue
            if datatype and f"/{datatype}/" not in key:
                continue
            if name.endswith(_TIMESERIES_EXT) and (not include_timeseries or (acquisition and f"acq-{acquisition}" not in name)):
                continue
            wanted.append((key, size))
    todo = [(k, s) for k, s in dict(wanted).items() if not (root / k.split("/", 1)[1]).exists()]
    return root, todo


def prefetch(task, subjects=None, sessions=None, include_timeseries=False, acquisition=None, workers=8):
    """Download everything these subjects/sessions need, `workers` files at a time, after one approval."""
    from concurrent.futures import ThreadPoolExecutor
    if os.environ.get("CML_DATA_SOURCE", "").lower() == "local":
        return cache_dir() / dataset_of(task)
    root, todo = plan_download(task, subjects, sessions, include_timeseries, acquisition)
    _approve(todo, task)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda kv: _download(kv[0], root / kv[0].split("/", 1)[1]), todo))
    if todo:
        print(f"  downloaded {len(todo)} file(s)")
    return root


def _have_locally(root, task, subject, session, include_timeseries, acquisition):
    """True if the requested session files are already on disk (no network)."""
    if subject is None or session is None or not (root / "dataset_description.json").exists():
        return False
    subjects = [subject] if isinstance(subject, str) else subject
    sessions = [session] if isinstance(session, (int, str)) else session
    for sub in subjects:
        for ses in sessions:
            found = list((root / f"sub-{str(sub).replace('sub-', '')}" / f"ses-{str(ses).replace('ses-', '')}").rglob(f"*task-{task}*"))
            if include_timeseries:
                found = [f for f in found if f.name.endswith(_TIMESERIES_EXT) and (not acquisition or f"acq-{acquisition}" in f.name)]
            if not found:
                return False
    return True


def get_bids_root(task, subject=None, session=None, include_timeseries=False, acquisition=None, datatype=None):
    """BIDS root for `task`: the rhino copy if present (and chosen), else the local
    OpenNeuro cache, downloading whatever is missing after one approval."""
    task = _canonical_task(task)
    rhino = Path(RHINO_ROOTS.get(dataset_of(task), "/nonexistent"))
    source = os.environ.get("CML_DATA_SOURCE", "").lower()
    if source == "local":
        return cache_dir() / dataset_of(task)
    if rhino.exists() and source != "openneuro":
        if source == "rhino":
            return rhino
        choice = _prompt(f"Read {task} from [1] the lab cluster ({rhino}) or [2] OpenNeuro? [1]: ", "1")
        if choice is None:
            raise RuntimeError("set CML_DATA_SOURCE=rhino or =openneuro for non-interactive runs")
        if choice.startswith("1"):
            return rhino
    root = cache_dir() / dataset_of(task)
    if _have_locally(root, task, subject, session, include_timeseries, acquisition):
        return root                                                # skip the S3 listing when cached
    root, todo = plan_download(task, subject, session, include_timeseries, acquisition, datatype)
    _approve(todo, task)
    for key, _ in todo:
        _download(key, root / key.split("/", 1)[1])
    return root


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("task")
    p.add_argument("--subject", "-s", nargs="*")
    p.add_argument("--session", nargs="*")
    p.add_argument("--eeg", action="store_true", help="also download the recordings")
    p.add_argument("--acq", choices=["bipolar", "monopolar"])
    p.add_argument("--list-subjects", action="store_true")
    a = p.parse_args()
    if a.list_subjects:
        print(", ".join(available_subjects(a.task)))
    else:
        print(prefetch(a.task, a.subject, a.session, include_timeseries=a.eeg, acquisition=a.acq))
