"""Write a fake but BIDS-valid FR1 session with a planted, known effect.

Use it to test a pipeline where the right answer is known (Module 11):

    with simulated("sim_bids"):                                   # loaders read the fake data, not OpenNeuro
        make_session("sim_bids/ds004789", "R9001S", 0)
        res = session_power("R9001S", 0, freqs=[5, 10, 20])      # rec/nrec power ratio at 10 Hz >> 1, ~1 elsewhere

During each 1600 ms word an `osc_hz` sine of amplitude `rec_amp` (later-recalled
words) or `nrec_amp` (others; default 0) is added on contacts in `effect_regions`,
plus an optional evoked bump (`erp_amp`, recalled words). Every contact also
carries a little 1/f noise (`noise_amp`, default 0.5 uV, so variances stay finite);
raise it (e.g. 30) for realistic data. Bipolar pairs are differences of
neighbouring contacts, as in the real data.
"""

import contextlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

STRIPS = {"LT": ("temporal", 7), "LF": ("frontal", 7), "LH": ("hippocampus", 7)}   # name: (region, contacts)
LABELS = {"temporal": ("superiortemporal", "Left STG superior temporal gyrus"),
          "frontal": ("superiorfrontal", "Left SFG superior frontal gyrus"),
          "hippocampus": ("n/a", "Left Hippocampus")}
WORDS = [f"WORD{i:03d}" for i in range(400)]


@contextlib.contextmanager
def simulated(cache="sim_bids"):
    """Inside the block cml_data reads `cache` as a complete local dataset (no downloads)."""
    import cml_data
    old = {k: os.environ.get(k) for k in ("CML_DATA_SOURCE", "CML_BIDS_CACHE")}
    os.environ.update(CML_DATA_SOURCE="local", CML_BIDS_CACHE=str(cache))
    cml_data.session_dataframe.cache_clear()
    try:
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.update({k: v})
        cml_data.session_dataframe.cache_clear()


def pink_noise(n, amp, rng):
    """1/f noise of length n with std `amp`."""
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    spec[1:] /= np.sqrt(np.arange(1, len(spec)))
    x = np.fft.irfft(spec, n)
    return amp * x / x.std()


def make_session(root, subject, session, task="FR1", sfreq=500, n_lists=6, list_len=12, p_recall=0.4,
                 osc_hz=10.0, rec_amp=20.0, nrec_amp=0.0, erp_amp=0.0, effect_regions=tuple(LABELS),
                 noise_amp=0.5, strips=STRIPS, seed=0):
    """Write sub-<subject>/ses-<session> under `root` (a dataset folder). Returns root."""
    import mne
    rng = np.random.default_rng(seed + 1000 * int(session))
    root = Path(root)
    sub, ses = f"sub-{subject}", f"ses-{session}"

    # ----- timeline: lists of words, then a recall period
    rows, t = [], 5.0
    for li in range(1, n_lists + 1):
        items = rng.choice(WORDS, list_len, replace=False)
        recalled = rng.random(list_len) < p_recall
        for sp, item in enumerate(items, 1):
            rows.append((t, 1.6, "WORD", item, sp, li)); t += 2.5
        t += 5.0
        rows.append((t, 0.0, "REC_START", "n/a", -999, li))
        for item in items[recalled]:
            t += 1.5; rows.append((t, 0.0, "REC_WORD", item, -999, li))
        t += 3.0
        rows.append((t, 0.0, "REC_END", "n/a", -999, li)); t += 5.0
    n = int((t + 10) * sfreq)
    ev = pd.DataFrame(rows, columns=["onset", "duration", "trial_type", "item_name", "serialpos", "list"])
    ev.insert(2, "sample", (ev["onset"] * sfreq).round().astype(int))
    words = ev[ev.trial_type == "WORD"]
    rec_items = {(r.list, r.item_name) for r in ev[ev.trial_type == "REC_WORD"].itertuples()}
    word_recalled = np.array([(r.list, r.item_name) in rec_items for r in words.itertuples()])

    # ----- signals per contact
    names, regions, data = [], [], []
    win = np.arange(int(1.6 * sfreq))
    bump = erp_amp * np.exp(-0.5 * ((win / sfreq - 0.4) / 0.1) ** 2)
    taper = np.minimum(1, np.minimum(win, win[::-1]) / (0.1 * sfreq))     # 100 ms ramps: no on/off splatter
    for strip, (region, n_c) in strips.items():
        for c in range(1, n_c + 1):
            x = pink_noise(n, noise_amp, rng)
            for s0, rec in zip(words["sample"], word_recalled):
                amp = rec_amp if (rec and region in effect_regions) else nrec_amp
                x[s0:s0 + len(win)] += amp * taper * np.sin(2 * np.pi * osc_hz * win / sfreq + rng.uniform(0, 2 * np.pi))
                if rec and region in effect_regions:
                    x[s0:s0 + len(win)] += bump * (-1) ** c          # polarity alternates along the strip, so pairs keep it
            names.append(f"{strip}{c}"); regions.append(region); data.append(x)
    mono = np.array(data) * 1e-6                                    # volts
    pairs = [(f"{s}{c}", f"{s}{c + 1}") for s, (_, n_c) in strips.items() for c in range(1, n_c)]
    bip = np.array([mono[names.index(a)] - mono[names.index(b)] for a, b in pairs])

    # ----- files
    d = root / sub / ses
    (d / "beh").mkdir(parents=True, exist_ok=True); (d / "ieeg").mkdir(exist_ok=True)
    for name, content in [("dataset_description.json", {"Name": "Simulated FR1", "BIDSVersion": "1.7.0", "DatasetType": "raw"}),
                          ("participants.tsv", "participant_id\n"), ("participants.json", {}), ("README", "simulated data\n")]:
        p = root / name
        if not p.exists():
            p.write_text(json.dumps(content, indent=1) if isinstance(content, dict) else content)
    beh = ev.assign(response_time="n/a", stim_file="n/a", test="n/a", answer="n/a", experiment=task, session=session, subject=subject)
    beh = beh[["onset", "duration", "sample", "trial_type", "response_time", "stim_file", "item_name", "serialpos", "list",
               "test", "answer", "experiment", "session", "subject"]]
    beh.to_csv(d / "beh" / f"{sub}_{ses}_task-{task}_beh.tsv", sep="\t", index=False)
    stem = d / "ieeg" / f"{sub}_{ses}_task-{task}"
    beh.to_csv(f"{stem}_events.tsv", sep="\t", index=False)
    for acq, arr, chs in [("monopolar", mono, names), ("bipolar", bip, [f"{a}-{b}" for a, b in pairs])]:
        raw = mne.io.RawArray(arr, mne.create_info(chs, sfreq, "seeg"), verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")               # EDF is 16-bit; the precision warning is expected
            mne.export.export_raw(f"{stem}_acq-{acq}_ieeg.edf", raw, fmt="edf", overwrite=True, verbose=False)
        json.dump({"TaskName": task, "SamplingFrequency": sfreq, "PowerLineFrequency": 60.0, "SoftwareFilters": "n/a",
                   "iEEGReference": acq, "SEEGChannelCount": len(chs)}, open(f"{stem}_acq-{acq}_ieeg.json", "w"), indent=1)
        pd.DataFrame({"name": chs, "type": "SEEG", "units": "V", "low_cutoff": "n/a", "high_cutoff": "n/a",
                      "group": [c.split("-")[0].rstrip("0123456789") for c in chs], "sampling_frequency": sfreq,
                      "description": "depth", "notch": "n/a"}).to_csv(f"{stem}_acq-{acq}_channels.tsv", sep="\t", index=False)
    pd.DataFrame({"name": names, "x": np.arange(len(names)) * 5.0, "y": 0.0, "z": 0.0, "size": -999,
                  "group": [nm.rstrip("0123456789") for nm in names], "hemisphere": "L", "type": "depth",
                  "wb.region": [LABELS[r][1] for r in regions], "ind.region": [LABELS[r][0] for r in regions],
                  "das.region": "n/a", "stein.region": "n/a"}).to_csv(f"{stem}_space-Talairach_electrodes.tsv", sep="\t", index=False)
    json.dump({"iEEGCoordinateSystem": "Talairach", "iEEGCoordinateUnits": "mm"}, open(f"{stem}_space-Talairach_coordsystem.json", "w"))
    return root
