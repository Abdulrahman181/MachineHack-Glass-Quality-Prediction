# Glass Quality Prediction — Weekend Hackathon 6

This repository contains a Jupyter notebook for an Extra Trees classification baseline for the Glass Quality Prediction challenge. The data files are **not included**; the notebook cannot be run end to end until the original challenge data is obtained separately.

## Repository contents

- `Glass Quality Prediction  Weekend Hackathon 6 #Rank 3.ipynb` — exploratory analysis, feature engineering, stratified cross-validation, model training, and probability export.
- `requirements.txt` — pinned Python dependencies for the notebook.

## Data expected by the notebook

Place the challenge CSVs in a `data/` directory at the repository root:

```text
data/
├── Train.csv
└── Test.csv
```

The training CSV must contain the `class` target and the feature columns used in the notebook; the test CSV must have matching feature columns. Obtain the original files from the challenge provider or an authorized copy. No dataset, sample-submission spreadsheet, or download link is provided in this repository, and none is fabricated here.

## Setup and run

Use Python 3.10–3.12 in a virtual environment. From the repository root:

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install -r requirements.txt
jupyter lab
```

Open the notebook and run its cells from top to bottom. It reads `data/Train.csv` and `data/Test.csv` relative to the repository root and writes the predicted class probabilities to `extratrees_addv1.xlsx` in the current working directory. The output columns follow the estimator's class order. The notebook does not consume a sample-submission file.

## Validation and limitations

Cross-validation uses stratified folds with a fixed random seed and reports mean log loss. The feature selection and preprocessing are intended as a challenge baseline, not a production pipeline. No accuracy or competition score is claimed: this repository does not include its input data, and the model workflow has not been run as part of this maintenance change.
