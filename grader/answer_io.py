"""answer_io — save/load per-question answers for the COGS 4290 autograder.

Both the solution notebooks (ground truth) and the student notebooks import this
to persist the exact arrays/values they plot for each question into
``answers/Module_<NN>/``.  A ``manifest.json`` in each module directory records how
every answer was serialized (type, shape, dtype, file) so the grader can load and
compare student vs. truth answers **without re-running any notebook**.

This file is the canonical copy (it lives in the ``grading-assignments`` skill and
is vendored identically into both course repos as ``grader/answer_io.py``).

Typical use inside a notebook (run from the repo root)::

    from grader.answer_io import save_answer
    save_answer("Q3.3_ave_spc", ave_spc, module=4, question="3.3")

Design goals: tiny, dependency-light (only numpy + optional pandas), never raises
out of a student's notebook for a recoverable problem (it warns instead), and
idempotent per key (re-saving a key overwrites cleanly).
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np

try:  # pandas is present in the course env but keep the import soft
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

MANIFEST_NAME = "manifest.json"

# Holds a strong reference to the most recently shown matplotlib figure so a
# *separate* grader cell can save it (the inline backend closes figures at cell
# end, so plt.gcf() in a later cell is blank — see enable_figure_capture).
_STATE = {"last_fig": None}


def enable_figure_capture():
    """Patch ``matplotlib.pyplot.show`` so each shown figure is stashed, letting a
    later grader cell save it via ``save_answer(..., fig="last")``.

    Idempotent and safe to call outside IPython/matplotlib. Relies on plot cells
    calling ``plt.show()`` (the convention in these notebooks); cells that only
    rely on auto-display won't be captured.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if getattr(plt.show, "_grader_wrapped", False):
        return
    _orig_show = plt.show

    def _show(*args, **kwargs):
        try:
            fig = plt.gcf()
            if fig.axes:
                _STATE["last_fig"] = fig
        except Exception:
            pass
        return _orig_show(*args, **kwargs)

    _show._grader_wrapped = True
    _show._grader_orig = _orig_show
    plt.show = _show


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def module_dir(module, answers_root="answers"):
    """Return (and create) the ``answers/Module_<NN>`` directory for *module*."""
    d = Path(answers_root) / f"Module_{int(module):02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(module, answers_root="answers"):
    return module_dir(module, answers_root) / MANIFEST_NAME


def _read_manifest(module, answers_root="answers"):
    p = _manifest_path(module, answers_root)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            warnings.warn(f"answer_io: could not parse {p}; starting a fresh manifest")
    return {"module": int(module), "generated_by": "answer_io", "answers": {}}


def _write_manifest(manifest, module, answers_root="answers"):
    _manifest_path(module, answers_root).write_text(json.dumps(manifest, indent=2))


# --------------------------------------------------------------------------- #
# type helpers
# --------------------------------------------------------------------------- #
def _is_scalar(value):
    return (
        value is None
        or isinstance(value, (bool, int, float, str))
        or isinstance(value, np.generic)
    )


def _jsonify(value):
    """Best-effort conversion of a scalar/short-container to a JSON-safe value."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    return value


def _json_safe(value):
    try:
        json.dumps(_jsonify(value))
        return True
    except Exception:
        return False


def _numeric_array(value):
    """Return an ndarray if *value* is a list/tuple of numbers, else None."""
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.dtype != object and np.issubdtype(arr.dtype, np.number):
        return arr
    return None


# --------------------------------------------------------------------------- #
# save
# --------------------------------------------------------------------------- #
def save_answer(
    key,
    value,
    *,
    module,
    answers_root="answers",
    question=None,
    role="plot",
    fig=None,
    desc=None,
):
    """Persist *value* under *key* into ``answers/Module_<NN>/`` and index it.

    Serialization is chosen by type:

    * ``numpy.ndarray``                      -> ``<key>.npy``
    * ``pandas.DataFrame`` / ``Series``      -> ``<key>.pkl``
    * scalar (int/float/bool/str/None)       -> inline ``value`` in the manifest
    * list/tuple of numbers                  -> ``<key>.npy`` (as an array)
    * JSON-safe list/dict                    -> inline ``value`` in the manifest
    * anything else                          -> ``<key>.pkl`` (pickle)

    Pass ``fig=<matplotlib Figure>`` or ``fig="current"`` to also snapshot the
    figure to ``<key>.png`` (used only for the report's visual context).

    Returns the manifest entry dict.
    """
    d = module_dir(module, answers_root)
    entry = {
        "key": key,
        "question": question,
        "role": role,
        "desc": desc,
        "type": None,
        "file": None,
        "value": None,
        "shape": None,
        "dtype": None,
        "columns": None,
        "fig": None,
    }

    try:
        if isinstance(value, np.ndarray):
            entry["type"] = "array"
            entry["file"] = f"{key}.npy"
            entry["shape"] = list(value.shape)
            entry["dtype"] = str(value.dtype)
            np.save(d / entry["file"], value)

        elif pd is not None and isinstance(value, (pd.DataFrame, pd.Series)):
            entry["type"] = "series" if isinstance(value, pd.Series) else "dataframe"
            entry["file"] = f"{key}.pkl"
            entry["shape"] = list(np.shape(value))
            if isinstance(value, pd.DataFrame):
                entry["columns"] = [str(c) for c in value.columns]
            value.to_pickle(d / entry["file"])

        elif _is_scalar(value):
            entry["type"] = "scalar"
            entry["value"] = _jsonify(value)

        elif isinstance(value, (list, tuple)) and _numeric_array(value) is not None:
            arr = _numeric_array(value)
            entry["type"] = "array"
            entry["file"] = f"{key}.npy"
            entry["shape"] = list(arr.shape)
            entry["dtype"] = str(arr.dtype)
            np.save(d / entry["file"], arr)

        elif isinstance(value, (list, tuple, dict)) and _json_safe(value):
            entry["type"] = "json"
            entry["value"] = _jsonify(value)

        else:
            entry["type"] = "pickle"
            entry["file"] = f"{key}.pkl"
            with open(d / entry["file"], "wb") as fh:
                pickle.dump(value, fh)
    except Exception as exc:  # never blow up a notebook run over a save
        warnings.warn(f"answer_io: failed to save '{key}' (module {module}): {exc}")
        return entry

    # optional figure snapshot: fig="last" (stashed by enable_figure_capture),
    # fig="current" (plt.gcf()), or an explicit Figure/Axes object
    if fig is not None:
        try:
            import matplotlib.pyplot as plt

            if fig == "last":
                figure = _STATE.get("last_fig")
            elif fig == "current":
                figure = plt.gcf()
            else:
                figure = getattr(fig, "figure", fig)  # accept an Axes too
            if figure is not None and getattr(figure, "axes", None):
                entry["fig"] = f"{key}.png"
                figure.savefig(d / entry["fig"], dpi=80, bbox_inches="tight")
        except Exception as exc:
            warnings.warn(f"answer_io: could not snapshot figure for '{key}': {exc}")

    manifest = _read_manifest(module, answers_root)
    manifest["answers"][key] = entry
    _write_manifest(manifest, module, answers_root)
    return entry


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def load_manifest(module, answers_root="answers"):
    """Return the runtime manifest dict for *module* (``answers`` key -> entry)."""
    return _read_manifest(module, answers_root)


def load_answer(key, module, answers_root="answers"):
    """Load a previously-saved answer *key* for *module*.

    Returns the reconstructed object, or raises ``KeyError`` if the key was never
    saved / ``FileNotFoundError`` if its backing file is missing.
    """
    manifest = _read_manifest(module, answers_root)
    if key not in manifest["answers"]:
        raise KeyError(f"answer '{key}' not found in module {module} manifest")
    entry = manifest["answers"][key]
    d = module_dir(module, answers_root)
    t = entry["type"]

    if t in ("scalar", "json"):
        return entry["value"]
    if t == "array":
        return np.load(d / entry["file"], allow_pickle=True)
    if t in ("dataframe", "series"):
        return pd.read_pickle(d / entry["file"])
    if t == "pickle":
        with open(d / entry["file"], "rb") as fh:
            return pickle.load(fh)
    raise ValueError(f"answer '{key}' has unknown stored type {t!r}")


def list_answers(module, answers_root="answers"):
    """Return the list of answer keys saved for *module*."""
    return list(_read_manifest(module, answers_root)["answers"].keys())
