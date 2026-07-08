# Saving your answers for grading

Your assignments are auto-graded by comparing the data you plot/compute to the
reference answers. This `grader/` package handles saving.

## What you do
Each assignment notebook has, after every question, a **grader cell** like:

```python
from grader.answer_io import save_answer
# Q3.3_ave_spc: dataframe: serialpos, recalled — between-subject averaged ltpFR2 SPC
save_answer("Q3.3_ave_spc", ave_spc, module=4, question="3.3")   # ← replace `ave_spc` with your variable
```

1. Do the analysis however you like and put your result in a variable.
2. In the grader cell, swap the placeholder for **your** variable name. The comment
   tells you the expected type/format (scalar, array, DataFrame + columns, …).
3. Run the grader cell — it writes your answer into `answers/Module_<NN>/`.

Run every grader cell without error before you submit. Your answers land in
`answers/Module_<NN>/` (a `manifest.json` + one file per non-scalar answer). Submit
that `answers/` folder (or the whole notebook + folder) as instructed.

**Figures are saved too.** A one-line "grader setup" cell near the top enables figure
capture; for each **plotted** question the grader cell also stores your figure as
`answers/Module_<NN>/<key>.png`. For this to work, **make sure each plotting cell calls
`plt.show()`** — that's when the figure is captured. Don't delete the setup cell.

## Notes
- The grader cells are added/updated by the instructor with
  `python grader/prep_assignments.py --mode student --repo . --all`. Don't delete them.
- `save_answer` never breaks your run — if a save fails it prints a warning and keeps going.
- You can re-save a question as many times as you want; the latest value wins.

See `grader/manifests/Module_<NN>.json` for the full list of answers expected per module.
