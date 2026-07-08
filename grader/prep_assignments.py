#!/usr/bin/env python3
"""prep_assignments — inject answer-saving scaffolding into course notebooks.

Driven entirely by the per-module answer manifest (``grader/manifests/Module_<NN>.json``).
Two modes:

* ``--mode student`` (run on ``COGS4290_DataMemoryBrains-BIDS``): adds a top
  "how to save your answers" instructions cell, a pre-filled ``save_answer(...)``
  **grader cell** after each ``# Question X.Y`` stub, and a one-line "save for
  grading" note appended to each ``## Question N`` prompt.

* ``--mode solution`` (run on ``DataSolutions-BIDS``): inserts the same grader
  cells after each answer, referencing the solution's real variable
  (``solution_var`` / ``solution_expr`` from the manifest) so executing the
  notebook populates the ground-truth ``answers/Module_<NN>/``.

The script is **idempotent**: it strips every cell/marker it previously added
before re-inserting, so re-running (or editing the manifest and re-running) always
yields the same notebook. Grader cells are tagged with ``metadata.grader`` and
prompt notes are wrapped in ``<!-- grader-note -->`` markers.

Usage
-----
    python prep_assignments.py --mode student  --repo /path/to/COGS4290_DataMemoryBrains-BIDS --modules 4
    python prep_assignments.py --mode solution --repo /path/to/DataSolutions-BIDS       --modules 4 6 15
    python prep_assignments.py --mode student  --repo <repo> --all      # every manifest in the repo
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nbformat

# Matches a code cell's sub-question tag. Handles integer (`# Question 3.3`,
# `# Question 1.1`) AND letter (`# Question 1c`, `# Q2c`, `# Q3`) sub-parts.
Q_CODE_RE = re.compile(r"^\s*#\s*Q(?:uestion|roblem)?\s*([0-9]+[a-z]?(?:\.[0-9]+)?)",
                       re.IGNORECASE)
# Matches a section header (`## Question N`, `## Problem N`).
Q_HEAD_RE = re.compile(r"^\s*#{1,6}\s*(?:Question|Problem)\s+([0-9]+)", re.IGNORECASE)
NOTE_RE = re.compile(r"\n*<!-- grader-note -->.*?<!-- /grader-note -->\n*", re.S)


def _norm_q(tok):
    """Normalize a sub-question token, e.g. '1C ' -> '1c', '3.3' -> '3.3'."""
    return str(tok).strip().lower()


def _section_of(q):
    """Leading integer (section number) of a question token, or None."""
    m = re.match(r"\s*([0-9]+)", str(q))
    return int(m.group(1)) if m else None


def _qkey(q):
    """Natural-sort key tolerant of letter sub-parts: 1a < 1b < 2 < 3.3."""
    parts = []
    for p in str(q).split("."):
        m = re.match(r"\s*([0-9]*)([a-z]*)", p.strip().lower())
        parts.append((int(m.group(1)) if m.group(1) else 0, m.group(2)))
    return parts

INSTRUCTIONS = """\
## 📥 Saving your answers for grading

This assignment is **auto-graded**. After each question there is a **grader cell**
that saves the data you plotted or computed into an `answers/Module_{nn}/` folder so
it can be compared against the reference answers.

**For each question:**
1. The question tells you which result(s) to produce and the expected data
   structure/format. Do your analysis and bind each result to a variable.
2. In the grader cell, replace the placeholder variable with **your** variable name.
3. Run the grader cell — it calls `save_answer(...)` and writes your answer.

Your **plots are saved too** — make sure each plotting cell calls `plt.show()` so the
figure can be captured for the grade report.

Make sure every grader cell runs without error before you submit. You don't need to
change anything else.
"""

SETUP = """\
# grader setup — enables saving the figures you plot (run once, early)
try:
    from grader.answer_io import enable_figure_capture
    enable_figure_capture()
except Exception as _e:
    print("grader: figure capture not enabled:", _e)
"""


def _setup_cell():
    cell = nbformat.v4.new_code_cell(SETUP)
    cell.metadata["grader"] = "setup"
    return cell


def _first_line(src):
    for line in src.splitlines():
        if line.strip():
            return line
    return ""


def _placeholder(key):
    """Suggested student variable name derived from the answer key."""
    return re.sub(r"^Q[0-9a-z.]+?_", "", key, flags=re.I) or "your_result"


def _hint(a):
    bits = []
    if a.get("type"):
        t = a["type"]
        if a.get("columns"):
            t += f": {', '.join(a['columns'])}"
        elif a.get("shape_hint"):
            t += f" {a['shape_hint']}"
        bits.append(t)
    if a.get("student_hint"):
        bits.append(a["student_hint"])
    return " — ".join(bits)


def _save_cell(module, question, answers, mode):
    """Build one tagged grader code cell holding all save_answer calls for a question."""
    lines = [f"# ── grader cell (Question {question}) ── saves your answer(s); "
             f"edit the variable name ──",
             "from grader.answer_io import save_answer"]
    fig_added = False
    for a in answers:
        key = a["key"]
        # attach the plotted figure to the first plot-role answer of this question
        figarg = ""
        if a.get("role") == "plot" and not fig_added:
            figarg = ', fig="last"'
            fig_added = True
        if mode == "solution":
            expr = a.get("solution_expr") or a.get("solution_var") or _placeholder(key)
            lines.append(
                f'save_answer("{key}", {expr}, module={module}, question="{question}"{figarg})'
            )
        else:
            var = _placeholder(key)
            lines.append(
                f'save_answer("{key}", {var}, module={module}, question="{question}"{figarg})'
                f'   # ← replace `{var}` with your variable'
            )
    cell = nbformat.v4.new_code_cell("\n".join(lines))
    cell.metadata["grader"] = "save"
    cell.metadata["grader_question"] = question
    return cell


def _instructions_cell(module):
    cell = nbformat.v4.new_markdown_cell(INSTRUCTIONS.format(nn=f"{module:02d}"))
    cell.metadata["grader"] = "instructions"
    return cell


def _note_for_question(qnum, answers):
    items = []
    for a in sorted(answers, key=lambda x: _qkey(x.get("question", ""))):
        items.append(f"`{_placeholder(a['key'])}` ({_hint(a) or 'see grader cell'}) "
                     f"→ Q{a['question']}")
    body = "; ".join(items)
    return (f"\n\n<!-- grader-note -->\n> **📥 For grading, produce and save:** {body}. "
            f"Bind each to a variable, then run the grader cell(s) below.\n"
            f"<!-- /grader-note -->")


def _strip_grader(nb):
    """Remove previously-inserted grader cells and note markers (idempotency)."""
    nb.cells = [c for c in nb.cells if c.get("metadata", {}).get("grader") not in
                ("save", "instructions", "setup")]
    for c in nb.cells:
        if c.cell_type == "markdown":
            c.source = NOTE_RE.sub("", c.source)


def prep_notebook(nb_path, spec, mode):
    """Rewrite the notebook at *nb_path* in place. Returns a per-question status dict."""
    nb = nbformat.read(str(nb_path), as_version=4)
    _strip_grader(nb)

    module = spec["module"]
    by_q = {}
    for a in spec.get("answers", []):
        # in solution mode, drop answers the solution doesn't (yet) produce
        if mode == "solution" and a.get("skip_solution"):
            continue
        by_q.setdefault(_norm_q(a["question"]), []).append(a)

    placed = set()
    status = {q: "unplaced" for q in by_q}

    # pass 1: anchor grader cells on `# Question X.Y` / `# Question 1c` code cells
    i = 0
    while i < len(nb.cells):
        c = nb.cells[i]
        if c.cell_type == "code":
            m = Q_CODE_RE.match(_first_line(c.source))
            if m:
                q = _norm_q(m.group(1))
                if q in by_q and q not in placed:
                    nb.cells.insert(i + 1, _save_cell(module, q, by_q[q], mode))
                    placed.add(q)
                    status[q] = "anchored"
                    i += 1
        i += 1

    # pass 2: fallback for unplaced questions -> end of their `## Question N` section
    unplaced = [q for q in by_q if q not in placed]
    if unplaced:
        # map each cell index to the section (leading integer) it belongs to
        section_end = {}  # section_int -> index of last cell before next section header
        cur = None
        for idx, c in enumerate(nb.cells):
            if c.cell_type == "markdown":
                hm = Q_HEAD_RE.match(_first_line(c.source))
                if hm:
                    cur = int(hm.group(1))
            if cur is not None:
                section_end[cur] = idx
        # insert from the bottom up so indices stay valid
        for q in sorted(unplaced, key=_qkey, reverse=True):
            sec = _section_of(q)
            if sec in section_end:
                nb.cells.insert(section_end[sec] + 1, _save_cell(module, q, by_q[q], mode))
                placed.add(q)
                status[q] = "section-fallback"

    # figure-capture setup cell, right after the first (title) cell — both modes
    nb.cells.insert(1 if nb.cells else 0, _setup_cell())

    if mode == "student":
        # top instructions cell (after the first H1/title cell)
        insert_at = 1 if nb.cells else 0
        nb.cells.insert(insert_at, _instructions_cell(module))
        # append save-note to each `## Question N` prompt
        for c in nb.cells:
            if c.cell_type == "markdown":
                hm = Q_HEAD_RE.match(_first_line(c.source))
                if hm:
                    qnum = int(hm.group(1))
                    ans = [a for a in spec.get("answers", [])
                           if _section_of(a["question"]) == qnum]
                    if ans:
                        c.source = c.source.rstrip() + _note_for_question(qnum, ans)

    nbformat.validate(nb)
    nbformat.write(nb, str(nb_path))
    return status


def _manifests(repo, modules, do_all):
    mdir = Path(repo) / "grader" / "manifests"
    if do_all:
        return sorted(mdir.glob("Module_*.json"))
    out = []
    for m in modules:
        p = mdir / f"Module_{int(m):02d}.json"
        if p.exists():
            out.append(p)
        else:
            print(f"  ! no manifest {p}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["student", "solution"], required=True)
    ap.add_argument("--repo", required=True, help="course repo root")
    ap.add_argument("--modules", "-m", nargs="*", type=int, default=[])
    ap.add_argument("--all", action="store_true", help="every manifest in the repo")
    args = ap.parse_args(argv)

    repo = Path(args.repo)
    for mpath in _manifests(repo, args.modules, args.all):
        spec = json.loads(mpath.read_text())
        nb_key = "student_notebook" if args.mode == "student" else "solution_notebook"
        nb_name = spec.get(nb_key)
        if not nb_name:
            print(f"Module {spec['module']:02d}: manifest has no {nb_key}; skipping")
            continue
        nb_path = repo / nb_name
        if not nb_path.exists():
            print(f"Module {spec['module']:02d}: notebook {nb_path} not found; skipping")
            continue
        status = prep_notebook(nb_path, spec, args.mode)
        fb = [q for q, s in status.items() if s == "section-fallback"]
        up = [q for q, s in status.items() if s == "unplaced"]
        msg = f"Module {spec['module']:02d} [{args.mode}]: {nb_path.name} — " \
              f"{sum(1 for s in status.values() if s != 'unplaced')}/{len(status)} placed"
        if fb:
            msg += f"; section-fallback: {fb}"
        if up:
            msg += f"; UNPLACED (review): {up}"
        print(msg)


if __name__ == "__main__":
    main()
