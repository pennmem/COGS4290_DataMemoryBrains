#!/usr/bin/env python3
"""check_solutions — audit that questions, grader cells, solutions and manifests agree.

For every ``grader/manifests/Module_<NN>.json`` this script cross-checks four things
that drift apart when notebooks are edited independently:

1. **Student notebook** — every manifest key is saved by exactly one grader cell,
   every graded question has a ``# Question X.Y`` stub, and every ``## Question N``
   section either has manifest entries or is flagged as ungraded.
2. **Solution notebook** — every manifest key is saved by a ``save_answer`` call and
   the ``solution_expr`` it saves is actually defined somewhere in the notebook.
3. **Manifest hygiene** — hints that mention "solution-only" / "review notes" leak
   grading internals to students; duplicate keys; keys whose question prefix does
   not match their ``question`` field.
4. **Reference answers** (optional, ``--answers``) — if the solution repo has an
   ``answers/Module_<NN>/manifest.json`` (produced by executing the solution), every
   manifest key must be present in it.

Nothing here executes a notebook; it is a static audit that runs in a second, so it
can be run after every edit.

Usage
-----
    python grader/check_solutions.py --repo . --solutions ../COGS4290_DataMemoryBrainsSolutions
    python grader/check_solutions.py --repo . --solutions ../Solutions --modules 10 14
    python grader/check_solutions.py --repo . --solutions ../Solutions --answers   # also check answers/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SAVE_RE = re.compile(r"save_answer\(\s*[\"']([^\"']+)[\"']\s*,\s*(.+?)\s*,\s*module\s*=", re.S)
Q_CODE_RE = re.compile(r"^\s*#\s*Q(?:uestion|roblem)?\s*([0-9]+[a-z]?(?:\.[0-9]+)?)", re.I | re.M)
Q_HEAD_RE = re.compile(r"^\s*#{1,6}\s*(?:Question|Problem)\s+([0-9]+)", re.I | re.M)
LEAK_RE = re.compile(r"solution[- ]only|review[_ ]notes|binding note", re.I)


def _cells(path):
    nb = json.loads(Path(path).read_text())
    return [("".join(c["source"]), c["cell_type"], c.get("metadata", {})) for c in nb["cells"]]


def _saves(cells):
    """{key: [expr, ...]} for every save_answer call in code cells."""
    out = {}
    for src, kind, _ in cells:
        if kind != "code":
            continue
        for key, expr in SAVE_RE.findall(src):
            out.setdefault(key, []).append(expr.strip())
    return out


import builtins
import keyword

_BUILTINS = set(dir(builtins)) | set(keyword.kwlist) | {"np", "pd", "plt", "scipy", "stats", "mne", "ptsa"}


def _undefined_names(expr, cells):
    """Identifiers used in `expr` that nothing in the notebook's code cells binds.

    `int(np.sum(FDR_sig))` -> checks `FDR_sig` only. Attribute accesses (`.Player`)
    and keyword names (`s` in a comprehension) are skipped, so this is a heuristic:
    it finds forgotten renames, not every possible NameError.
    """
    code = "\n".join(src for src, kind, _ in cells if kind == "code")
    # strip string literals and attribute tails from the expression
    stripped = re.sub(r"(['\"]).*?\1", "", expr)
    stripped = re.sub(r"\.[A-Za-z_][A-Za-z0-9_]*", "", stripped)
    names = set(re.findall(r"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_]*)", stripped))
    # names bound inside the expression itself (comprehension variables, lambdas)
    local = set(re.findall(r"\bfor\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\b", stripped))
    local |= set(re.findall(r"\blambda\s+([A-Za-z_][A-Za-z0-9_, ]*):", stripped))
    missing = []
    for n in sorted(names - _BUILTINS - local):
        pat = re.compile(rf"(^|[\s,(\[])({re.escape(n)})\s*(=[^=]|,|\)|\bin\b)|"
                         rf"\bdef\s+{re.escape(n)}\b|\bclass\s+{re.escape(n)}\b|"
                         rf"\bimport\s+.*\b{re.escape(n)}\b|\bas\s+{re.escape(n)}\b|"
                         rf"\bfor\s+.*\b{re.escape(n)}\b", re.M)
        if not pat.search(code):
            missing.append(n)
    return missing


def check_module(manifest_path, repo, solutions, check_answers):
    m = json.loads(manifest_path.read_text())
    nn = f"{int(m['module']):02d}"
    problems = []
    notes = []
    answers = m.get("answers", [])
    keys = [a["key"] for a in answers]

    # ---- manifest hygiene -------------------------------------------------
    dupes = {k for k in keys if keys.count(k) > 1}
    for k in sorted(dupes):
        problems.append(f"manifest: duplicate key {k}")
    for a in answers:
        q = str(a.get("question", ""))
        if not a["key"].lower().startswith(f"q{q}_".lower()):
            problems.append(f"manifest: key {a['key']} does not match question {q}")
        hint = a.get("student_hint", "")
        if LEAK_RE.search(hint):
            problems.append(f"manifest: student_hint for {a['key']} leaks grading notes: {hint!r}")
        if not a.get("solution_expr"):
            problems.append(f"manifest: {a['key']} has no solution_expr")

    # ---- student notebook -------------------------------------------------
    stud = repo / m["student_notebook"]
    if not stud.exists():
        problems.append(f"student notebook missing: {m['student_notebook']}")
    else:
        cells = _cells(stud)
        saves = _saves(cells)
        for k in keys:
            n = len(saves.get(k, []))
            if n == 0:
                problems.append(f"student: key {k} has no grader cell (re-run prep_assignments?)")
            elif n > 1:
                problems.append(f"student: key {k} saved {n} times")
        for k in saves:
            if k not in keys:
                problems.append(f"student: grader cell saves {k}, not in manifest")
        alltext = "\n".join(src for src, _, _ in cells)
        stubs = {s.lower() for s in Q_CODE_RE.findall(alltext)}
        for q in sorted({str(a["question"]).lower() for a in answers}):
            if q in stubs:
                continue
            sec = str(re.match(r"[0-9]+", q).group(0))
            if sec in stubs and len([s for s in stubs if s.startswith(sec)]) == 1:
                # single `# Question N` stub, manifest says N.1 -> prep_assignments places
                # the grader cell at the end of the section; harmless
                notes.append(f"student: manifest question {q} placed via '# Question {sec}' section fallback")
            else:
                problems.append(f"student: no '# Question {q}' code stub for graded question {q} "
                                f"(stubs in section {sec}: {sorted(s for s in stubs if s.startswith(sec))})")
        heads = {int(h) for h in Q_HEAD_RE.findall(alltext)}
        graded_sections = {int(re.match(r"[0-9]+", str(a["question"])).group(0)) for a in answers}
        for h in sorted(heads - graded_sections):
            notes.append(f"student: '## Question {h}' has no graded answers (text-only question?)")
        for s in sorted(graded_sections - heads):
            problems.append(f"student: manifest grades question {s} but no '## Question {s}' header")

    # ---- solution notebook ------------------------------------------------
    sol = solutions / m["solution_notebook"] if solutions else None
    if sol is None:
        pass
    elif not sol.exists():
        problems.append(f"solution notebook missing: {m['solution_notebook']}")
    else:
        cells = _cells(sol)
        saves = _saves(cells)
        for a in answers:
            k = a["key"]
            if k not in saves:
                problems.append(f"solution: key {k} never saved")
                continue
            for expr in saves[k]:
                missing = _undefined_names(expr, cells)
                if missing:
                    problems.append(f"solution: {k} saves `{expr}` but nothing defines "
                                    + ", ".join(f"`{n}`" for n in missing))
            if a.get("solution_expr") and a["solution_expr"] not in saves[k]:
                notes.append(f"solution: {k} saves `{saves[k][0]}`, manifest says `{a['solution_expr']}`")
        for k in saves:
            if k not in keys:
                problems.append(f"solution: saves {k}, not in manifest")

    # ---- reference answers ------------------------------------------------
    if check_answers and solutions:
        ref = solutions / "answers" / f"Module_{nn}" / "manifest.json"
        if not ref.exists():
            notes.append(f"answers: no reference answers generated yet ({ref.relative_to(solutions)})")
        else:
            got = json.loads(ref.read_text())
            # answer_io writes {"module": .., "answers": {key: entry, ..}}
            if isinstance(got, dict) and isinstance(got.get("answers"), dict):
                got_keys = set(got["answers"])
            else:
                got_keys = set(got.keys() if isinstance(got, dict) else [x["key"] for x in got])
            for k in keys:
                if k not in got_keys:
                    problems.append(f"answers: reference answers lack {k}")

    return nn, m.get("title", ""), problems, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".", help="student repo root (contains grader/manifests)")
    ap.add_argument("--solutions", default=None, help="solutions repo root")
    ap.add_argument("--modules", nargs="*", type=int, help="restrict to these module numbers")
    ap.add_argument("--answers", action="store_true", help="also check answers/Module_NN in the solutions repo")
    ap.add_argument("-q", "--quiet", action="store_true", help="only print problems, not notes")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    solutions = Path(args.solutions).resolve() if args.solutions else None
    manifests = sorted((repo / "grader" / "manifests").glob("Module_*.json"))
    if args.modules:
        want = {f"{n:02d}" for n in args.modules}
        manifests = [p for p in manifests if p.stem.split("_")[1] in want]

    total = 0
    for mp in manifests:
        nn, title, problems, notes = check_module(mp, repo, solutions, args.answers)
        total += len(problems)
        status = "OK " if not problems else f"{len(problems):3d}"
        print(f"[{status}] Module {nn}  {title}")
        for p in problems:
            print(f"      ✗ {p}")
        if not args.quiet:
            for n in notes:
                print(f"      · {n}")
    print(f"\n{total} problem(s) across {len(manifests)} module(s)")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
