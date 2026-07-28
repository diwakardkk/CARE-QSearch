# CARE-QSearch Run Commands

Optional but recommended for real small-N circuit metrics:

```bash
python3 -m pip install -r requirements.txt
```

If Qiskit/Aer segfaults on your Python build, run without Qiskit metrics.
This is the default. Qiskit metrics are opt-in with `CARE_USE_QISKIT=1`.

Run a quick validation:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 run_all.py --mode smoke --force
```

Run the full controlled experiment and replace old outputs.
This includes `CARE-Fuse`, the cost/selectivity-aware fixed-budget method, and
`CARE-Fuse-95`, the success-constrained variant:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 run_all.py --mode full --force
```

Opt-in Qiskit small-N circuit metrics only if your Qiskit installation is stable:

```bash
CARE_USE_QISKIT=1 MPLCONFIGDIR=/private/tmp/mplconfig python3 run_all.py --mode full --force
```

Recommended stronger-paper sanity check: run a separate small-N Qiskit validation
instead of mixing Qiskit into the full run:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 run_all.py --mode qiskit-validation
```

Regenerate analysis/tables/summaries from existing raw results:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 run_all.py --mode analysis-only --force
```

Regenerate figures from existing raw results:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 run_all.py --mode plots-only --force
```

`--force` removes the old `output/` directory before smoke/full runs, so manual reruns replace previous results.
For analysis-only and plots-only, `--force` refreshes derived outputs from the existing raw results.

Primary outputs after the run:

- PNG figures: `output/figures/png/`
- Excel workbook: `output/tables/excel/all_tables.xlsx`
- Individual Excel tables: `output/tables/excel/table*.xlsx`
- Small-N Qiskit validation workbook: `output/qiskit_validation/qiskit_smallN_validation.xlsx`
