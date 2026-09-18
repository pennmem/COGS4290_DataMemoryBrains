# Big Data, Memory, and the Human Brain (COGS 4290/PSYC 4290)

These are the assignments and supporting material for Michael Kahana's course
on "Big Data, Memory, and the Human Brain" (COGS 4290/PSYC 4290) at the
University of Pennsylvania. They form a solid introduction to performing EEG
analyses and include many resources for learning these tools and methods beyond
the homework assignments. These assignments additionally provide a resource for
getting up to speed on the research done in Mike's lab.

# Course Structure
Our goal is to familiarize you with fundamental concepts in human memory and
electrophysiology as well as with programming tools needed for the large-scale
computing in these fields. The psychology and neuroscience at play in these
analyses will be primarily covered in the course lectures.

The material is a sequence of numbered modules. Most come in pairs: an
*introduction* notebook that teaches a tool or method on one subject, followed
by an *assignment* notebook that applies it and is auto-graded (see
[`grader/README.md`](grader/README.md)). Work through them in order.

| Module | Notebook | Type | What it covers |
|---|---|---|---|
| 00 | Python, Jupyter, Numpy, Pandas | assignment 0 | Python warm-up and plotting |
| 01 | OpenBIDS | intro | how the lab's data are laid out (BIDS), `BIDSReader` |
| 02 | Behavioral Analysis with BIDSReader | intro | events tables, `BehavioralHelpers.py` |
| 03 | Behavioral Analysis of Memory | assignment 1 | recall probability, serial position curves |
| 04 | Exceptions, IRT, Lag-CRP, PLI | intro | robust code; inter-response times, lag-CRP, prior-list intrusions |
| 05 | IRT, Lag-CRP, PLI | assignment 2 | the analyses of Module 04 across a cohort |
| 06 | EEG and ERPs | intro | loading intracranial EEG, event-related potentials |
| 07 | EEG and ERPs | assignment 3 | electrode localisation, subsequent-memory ERPs, per-timepoint tests |
| 08 | Univariate Statistics | intro (required) | t-tests, multiple comparisons, FDR |
| 09 | Spectral Analysis | intro | Welch, Morlet wavelets, line noise, the 1/f spectrum |
| 10 | Signal / Spectral Analysis | assignment 4 | power spectra for recalled vs non-recalled words; `SpectralHelpers.py` |
| 11 | Simulation and Unit Testing | intro | simulate EEG with a known answer, write tests that fail on purpose |
| 12 | Parallel Computing | intro | running one-session functions over a cohort in parallel (`cml_parallel`) |
| 13 | RAM and File I/O | intro | predicting memory use, saving intermediate results |
| 14 | Spectral Inferences and the SME | assignment 5 | 20-subject spectra with confidence bands, normalisation, regional SME, referencing |
| 15 | Machine Learning | intro | scikit-learn, cross-validation |
| 16 | Machine Learning | assignment 6 | logistic-regression classifiers of memory state, ROC/AUC |
| 17 | Classifier Validation | intro | permutation tests, leakage |
| 18 | Hyperparameters and Nested CV | assignment 7 | nested cross-validation, penalisation schemes |
| 19 | Representational Similarity Analysis | intro | RSA theory and computation |
| 20 | RSA | assignment 8 | encoding-retrieval similarity |
| X, Y | Connectivity, Oscillation Detection | optional | further methods (Rhino only: need `pycircstat` / `irasa`, which no longer install locally) |

Behavioral modules (01-05) use the scalp-EEG PEERS studies (`ltpFR`, `ltpFR2`,
`VFFR`); every EEG module from 06 onward uses intracranial FR1 data. Module 10
onward analyse the 20-subject cohort `cml_data.INTRAC_SUBS`.
`cml_parallel.run_sessions` runs your per-session function over that cohort
(caching one result per session, several sessions at a time) — the local
replacement for the lab's cluster job launcher.
`cml_sim.make_session` writes a fake session with a known effect, so you can check
your pipeline against a known answer before running it on real data (Module 11).

By the end of this course, you should be able to carry out EEG/iEEG/ECoG
analyses, like computing spectral power and phase, and to compute statistics or
to apply machine learning models to those data.

These notebooks prepare you for doing in-depth multi-subject analyses with
electrophysiological data. Though this course assumes a basic knowledge of
Python and command line tools, we have linked additional recommended resources
at the top of **Module 00** for getting started with Python and common
data analysis tools. Though the externally linked supplemental material isn't
strictly part of the course, we recommend reviewing it before proceeding to the
materials included here unless you are confident in your experience with numpy,
pandas, scipy, and basic python syntax. If those words don't mean anything to
you (or you want to brush up), please read through these resources! 

# Initial Setup

To start working with any materials contained or linked here, you'll need to
set up tools for writing and running Python code on your own computer. Members
of the Computational Memory Lab can alternatively work on Rhino, the lab's
computing cluster, when it is available; see *Working on Rhino* at the end of
this section. Everything in these notebooks runs in either place.

## Command line access

All subsequent stages of these instructions will assume familiarity with and
access to a Linux (or other \*NIX) command line. If this is unfamiliar to you,
please use the resources below to get yourself oriented.

If you are using an apple computer running macOS or a Linux computer, you
already have access to a command line. On Windows, we recommend using the
Ubuntu subsystem
<https://docs.microsoft.com/en-us/windows/wsl/install-win10> or Cygwin
<https://www.cygwin.com/>.

General Introduction:
https://ubuntu.com/tutorials/command-line-for-beginners#1-overview

## Getting the course GitHub repository

In a terminal in the location where you would like to download these course
assignment materials, enter the following:

    git clone --recurse-submodules https://github.com/pennmem/COGS4290_DataMemoryBrains.git

If git is not installed, you can find instructions
[here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

This repository will be downloaded to a folder named COGS4290_DataMemoryBrains in
the same location where you ran the git clone command. `--recurse-submodules`
also fetches the `bidsreader` package the notebooks import from
`dependencies/bidsreader`; if you cloned without it, run
`git submodule update --init --recursive` inside the folder.

## Setting up your environment

We use conda to manage the various libraries needed to perform analyses using
Python. Conda is a tool that allows Python libraries to be installed into
'environments.' This is a folder that lets you manage the needs of different
projects independently; using some sort of virtual environment system is a
standard practice and isolates issues when they come up. Conda is available
from the [Anaconda project
home](https://docs.conda.io/projects/conda/en/latest/index.html). We recommend
installing miniconda, though you can read the installation instructions and
decide for yourself which distribution is best for you.

Once you have conda set up, create a new environment and activate it. The
activation step is necessary any time you open a new terminal:

    conda create -y -n <environmentname> python=3.11
    conda activate <environmentname>
    NOTE: 'environmentname' is a placeholder, please replace it with a more descriptive name!

Next, install the suite of tools for EEG analysis. First MNE (this may take a
while, because MNE has a lot of dependencies):

    conda install -c conda-forge mne mne-bids

If this does not work at first, try `pip install mne mne-bids`.

Next, install PTSA, which is a set of EEG tools developed by former lab
members:

    conda install "traitlets<5"
    conda install -c pennmem ptsa

Install a few extra packages in use for these notes:

    conda install scikit-learn statsmodels seaborn psutil dask distributed pytest

Finally, install Jupyter and link it with this environment:

    conda install jupyterlab ipykernel
    python -m ipykernel install --user --name environmentname --display-name "environmentname"

Start JupyterLab with `jupyter lab` from inside the repository folder and pick
"environmentname" as the kernel when you open a notebook.

## Working on Rhino

Lab members with a Rhino account can do everything above on the cluster
instead, where the data are already on disk and a kernel with all dependencies
exists. Log in with any ssh client, replacing "username" with your username,
and set your password:

    ssh username@rhino2.psych.upenn.edu
    passwd

JupyterLab runs at
[https://rhino2.psych.upenn.edu:9500](https://rhino2.psych.upenn.edu:9500) on
campus. Off campus, either use the Penn VPN GlobalProtect
([https://www.isc.upenn.edu/how-to/university-vpn-getting-started-guide](https://www.isc.upenn.edu/how-to/university-vpn-getting-started-guide))
or open an ssh tunnel and browse to
[https://127.0.0.1:8000](https://127.0.0.1:8000) (the "s" in https matters;
override the certificate warning, the connection is protected by ssh):

    ssh -L8000:rhino2.psych.upenn.edu:9500 username@rhino2.psych.upenn.edu

In JupyterLab, open any notebook, go to Kernel -> Change Kernel... and select
"workshop" (or `workshop_311`) from the dropdown. Module 12 (parallel
computing) additionally uses the cluster's job scheduler.

## Getting the data

The data lives in two places, and the notebooks work with either.

**On Rhino**, it is already on disk under `/data/LTP_BIDS/` and you need to do
nothing at all.

**Anywhere else**, download it from [OpenNeuro](https://openneuro.org), where the
lab publishes its BIDS datasets under a CC0 licence.

### Downloading from OpenNeuro

Use the [`cml_data.py`](cml_data.py) helper included in this repository. To see
what a dataset contains before downloading anything:

    python cml_data.py FR1 --list-subjects

To download one subject's behavioural events, channel tables, and electrode
coordinates (a few hundred KB):

    python cml_data.py FR1 --subject R1111M --session 0

To also download the EEG recording itself — this is the big one, 300–700 MB per
session, so be deliberate about it:

    python cml_data.py FR1 --subject R1111M --session 0 --eeg --acq monopolar

Everything lands in `./bids_data/` (gitignored) and is never re-downloaded. Set
`CML_BIDS_CACHE` to keep the cache somewhere else — a shared drive, say, so a
whole class does not each fetch their own copy.

The subjects each notebook needs are named at the top of its data-loading cell.

### What happens when you run a notebook

The first data cell calls `get_bids_root(...)`, which **asks you where to read
from** rather than deciding for you:

```
Where should this notebook read FR1 data from?
  [1] the lab cluster   /data/LTP_BIDS/FR1   (detected)
  [2] OpenNeuro ds004789    (downloads to bids_data/ds004789)
Choice [1]:
```

Option [1] only appears when the cluster copy is actually there. If you choose
OpenNeuro, it works out exactly which files are missing and asks before fetching:

```
OpenNeuro ds004789 (FR1) — need 2 file(s), 600.1 MB to download.
  destination: bids_data/ds004789
Download 600.1 MB? [y/N]:
```

Nothing is ever downloaded without you saying yes, and if you already have the
files it does not ask at all. If you pre-downloaded with the commands above, the
notebook finds them and moves on.

For non-interactive runs (scripted execution, `nbconvert`, CI) answer in advance
with environment variables — without them, a run that would need to download
stops with an error rather than quietly pulling gigabytes:

    export CML_DATA_SOURCE=openneuro   # or: rhino
    export CML_AUTO_APPROVE=1

### Exploring a dataset from Python

```python
from cml_data import available_subjects, available_tasks
available_subjects("ltpFR2")          # 364 PEERS subjects
available_tasks("ltpFR2", "LTP093")   # ['ltpFR', 'ltpFR2']
```

| Task | OpenNeuro | Type |
|---|---|---|
| `FR1` | [ds004789](https://openneuro.org/datasets/ds004789) | intracranial |
| `catFR1` | [ds004809](https://openneuro.org/datasets/ds004809) | intracranial |
| `PAL1` | [ds005059](https://openneuro.org/datasets/ds005059) | intracranial |
| `pyFR` | [ds004865](https://openneuro.org/datasets/ds004865) | intracranial |
| `RepFR1` | [ds005411](https://openneuro.org/datasets/ds005411) | intracranial |
| `ltpFR`, `ltpFR2`, `VFFR` (PEERS) | [ds004395](https://openneuro.org/datasets/ds004395) | scalp EEG |
| `NICLS` | [ds004706](https://openneuro.org/datasets/ds004706) | scalp EEG |

Note that BIDS task labels are **case-sensitive** (`ltpFR2`, not `ltpfr2`) when you
pass them to `BIDSReader`; `get_bids_root` accepts either.

The RSA modules additionally need precomputed derivative files that are *not* part
of the OpenNeuro release — see the note at the top of those notebooks.

### The bidsreader submodule

Most notebooks import `bidsreader` from `dependencies/bidsreader`. Fetch it with:

    git submodule update --init --recursive

If you still have Rhino access and prefer to read from the cluster directly, pass
the cluster path as `root=` instead of calling `get_bids_root`. For questions about
data access, contact kahana-sysadmin@sas.upenn.edu.

