# Glass Quality Prediction — Engineering-maintained baseline

This repository contains a reproducible Extra Trees classification baseline for the MachineHack glass quality challenge. The challenge data is **not included**. The workflow refuses to run when the required files are absent or invalid; it does not download, synthesize, or claim results for missing data.

## Repository contents

- `Glass Quality Prediction  Weekend Hackathon 6 #Rank 3.ipynb` — cleaned notebook using the reusable validation/model code.
- `glass_quality.py` — data-contract validation, row-local feature engineering, deterministic feature selection, stratified log-loss validation, and prediction export.
- `requirements.txt` — runtime and development dependencies.
- `tests/test_repository_static.py` — static checks for notebook hygiene and unsafe path/artifact patterns.
- `ENGINEERING_REVIEW.md` — audit findings, changes, and validation limits.

## Data contract

Obtain the authorized challenge files separately and place them here:

```text
data/
├── Train.csv
└── Test.csv
```

`Train.csv` must contain a numeric `class` target and numeric feature columns. The geometry workflow additionally requires `xmax`, `xmin`, `ymax`, `ymin`, `thickness`, `log_area`, and `pixel_area`. `Test.csv` must contain exactly the same feature columns (without `class`). Both files must be non-empty and free of missing or non-finite feature values. The loader reports a clear error for missing files, mismatched schemas, duplicate columns, non-numeric values, missing/infinite values, missing geometry inputs, or insufficient class variety. No sample-submission spreadsheet is required.

## Setup and run

Use Python 3.10–3.12 from the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open the notebook and run cells top-to-bottom. It writes class-probability columns in estimator class order to `extratrees_addv1.xlsx`. The notebook uses `StratifiedKFold(shuffle=True, random_state=11)` and a fresh seeded Extra Trees estimator in every fold.

For static checks:

```bash
ruff check .
pytest -q
```

## Leakage and limitations

Feature engineering is row-local and does not use the target. Model validation fits each fold separately. Identifier-like columns, the target, and explicitly redundant derived columns from the original notebook are excluded. This remains a challenge baseline rather than a production API; there is no web/API surface in the repository. No accuracy, log-loss result, or competition rank is asserted by this maintenance review because input data is unavailable.
