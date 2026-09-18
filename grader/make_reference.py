#!/usr/bin/env python3
"""make_reference — export self-check fingerprints from the solutions' answers.

Writes ``grader/reference/Module_<NN>.json`` for the student repo. Each key gets
its shape/columns and either a SHA-256 of the reference value (``kind: hash`` —
integers, labels, and any manifest entry marked ``"exact": true``) or ``kind:
form`` (float answers graded with a tolerance; only shape/columns are checked
locally). Hashes cannot be reversed, so nothing here reveals an answer.

    python grader/make_reference.py --solutions ../COGS4290_DataMemoryBrainsSolutions [--modules 3 5]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from answer_io import fingerprint, load_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", required=True, help="solutions repo root (has answers/Module_NN)")
    ap.add_argument("--modules", nargs="*", type=int, help="default: every manifest with generated answers")
    a = ap.parse_args()
    answers_root = str(Path(a.solutions) / "answers")
    (HERE / "reference").mkdir(exist_ok=True)
    for mpath in sorted((HERE / "manifests").glob("Module_*.json")):
        spec = json.loads(mpath.read_text())
        nn = spec["module"]
        if a.modules and nn not in a.modules:
            continue
        if not (Path(answers_root) / f"Module_{nn:02d}" / "manifest.json").exists():
            print(f"Module {nn:02d}: no answers generated yet, skipped")
            continue
        ref, missing = {}, []
        for entry in spec["answers"]:
            try:
                value = load_answer(entry["key"], nn, answers_root)
            except Exception:
                missing.append(entry["key"])
                continue
            fp = fingerprint(value)
            kind = "hash" if (not fp["float"] or entry.get("exact")) else "form"
            ref[entry["key"]] = {"kind": kind, "shape": fp["shape"], "columns": fp["columns"],
                                 **({"sha256": fp["sha256"]} if kind == "hash" else {}),
                                 **({"hint": entry["hint"]} if entry.get("hint") else {})}
        (HERE / "reference" / f"Module_{nn:02d}.json").write_text(json.dumps(ref, indent=1) + "\n")
        kinds = [r["kind"] for r in ref.values()]
        print(f"Module {nn:02d}: {kinds.count('hash')} hashed, {kinds.count('form')} form-only"
              + (f", missing {missing}" if missing else ""))


if __name__ == "__main__":
    main()
