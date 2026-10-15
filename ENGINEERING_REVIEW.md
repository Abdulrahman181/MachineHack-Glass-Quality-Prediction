# Engineering Review — Glass Quality Prediction

## Scope

Audited the complete tracked repository: the single Jupyter notebook and README, plus the repaired Python module, dependency file, ignore rules, and static tests added during this review. No other repository was accessed or modified. No challenge data or generated prediction artifact was present.

## Findings and remediation

### High-impact issues fixed

1. **Missing data and artifact assumptions:** the notebook previously read `data/Train.csv`, `data/Test.csv`, and an unavailable `data/Sample_Submission.xlsx` without a guard. It now validates the two required CSVs and raises an actionable `FileNotFoundError` when either is absent. The sample-submission dependency was removed; predictions are written as `.xlsx` through `openpyxl`.
2. **Validation reproducibility:** the original workflow used plain `KFold`, ignored its `n_splits` argument, and did not shuffle or stratify. Validation now uses shuffled `StratifiedKFold` with `random_state=11`, validates the smallest class size, and fits a fresh cloned estimator per fold.
3. **Feature-order instability:** the original `list(set(...))` produced non-deterministic feature order. Selection now preserves input column order and explicitly excludes the target, identifier-like columns, and the original notebook's redundant/derived exclusions.
4. **Target-derived heuristic risk:** the original notebook forcibly changed test probabilities based on an observed `x_component`/class relationship. That shortcut was removed; predictions come from the fitted estimator only. Geometry features are row-local and never read the target.
5. **Pandas/index and file-format fragility:** the original used chained/positional assignments and attempted `.xls` output without declaring the legacy writer dependency. The repaired flow avoids the heuristic mutation and exports class-probability columns in estimator class order to `.xlsx`.
6. **Stale notebook state:** all old execution counts and outputs were removed. Notebook code is now AST-parseable and does not contain hardcoded machine paths or the removed sample-submission filename.

### Other issues addressed

- Added `glass_quality.py` so data validation, feature engineering, cross-validation, and prediction behavior are testable outside a notebook.
- Added `requirements.txt` with bounded compatible versions for the actual imports; no unused XGBoost dependency was introduced.
- Added `.gitignore` rules for challenge data, generated spreadsheets, caches, and virtual environments.
- Replaced README claims about untracked files and multiple algorithms with the actual repository contract and explicit limitations.
- Added static pytest checks for JSON validity, Python AST validity, stale output removal, unsafe local paths, and required reproducibility guards.

## Leakage, split, and metric review

- The model is evaluated with stratified folds; each fold fits a fresh Extra Trees estimator only on its training partition.
- Log loss is computed with `labels=fitted.classes_`, avoiding class-column ordering assumptions.
- Feature derivation uses only same-row input columns. No target, fold-wide statistics, test labels, or external artifacts are read during feature creation.
- Identifier-like names (`id`, `index`, `unnamed: 0`) are excluded. Other domain-specific leakage cannot be established without the unavailable challenge schema and provenance, so this remains a baseline rather than a production claim.
- No accuracy, log-loss value, competition rank, or end-to-end result is reported by this review because the data is absent.

## Security and API review

There is no web service, public API, credential handling, network download, shell execution, deserialization, or user-input endpoint in the repository. The repaired loader only reads explicitly named local CSV paths under the selected data directory and does not invent fallback data. Generated prediction files are ignored by Git.

## Validation performed

- `ruff check .` — passed with no remaining diagnostics.
- `pytest -q` — **4 passed** static tests.
- Python AST parse and `py_compile` for `glass_quality.py` and tests — passed.
- Notebook JSON parse and AST parse for every code cell — passed; all code-cell outputs are empty and execution counts are `null`.
- Missing-data guard invocation — correctly stopped at `data/Train.csv` with a clear acquisition message.
- Full model training and score generation — **not run** because the repository does not contain `Train.csv`/`Test.csv`; no synthetic data was created.

## Remaining limitations

The challenge data is not bundled, so schema-dependent model behavior, class balance, runtime, probability calibration, and competition submission compatibility remain unverified. Users must obtain authorized data separately and run the notebook from the repository root after installing `requirements.txt`.

## Second Senior Review

### Scope and audit approach

The second review re-inspected every tracked file, the complete notebook, Git history and current diff, Python/config/test/README files, dependency bounds, path and secret patterns, notebook execution state, model contracts, split/evaluation logic, and artifact behavior. The repository is still limited to one notebook, one reusable Python module, one dependency file, static tests, and documentation; no web/API surface or model artifact loader exists.

### Findings fixed in this pass

1. **Documented numeric-target contract was not enforced.** `load_data` now rejects non-numeric `class` labels, missing labels, and non-finite numeric labels before model fitting.
2. **Non-finite feature values could pass the numeric dtype check.** Train and test features now reject `NaN`, positive/negative infinity, and other non-finite values with actionable errors.
3. **The loader accepted schemas that the notebook could not engineer.** It now explicitly requires the seven geometry inputs used by `create_features`, so invalid challenge files fail at loading rather than midway through the notebook.
4. **Public helpers had weak preconditions.** `create_features`, cross-validation, and final fitting now validate empty frames, missing/duplicate features, numeric finite matrices, target cardinality, fold feasibility, integer `n_splits`, and the presence of `predict_proba`.
5. **There were no functional tests for the reusable module.** Added `tests/test_glass_quality.py` covering CSV contracts, finite-value rejection, row-local feature generation, exclusion rules, fresh-estimator cross-validation, class-order probability output, and early failure on invalid inputs.

### Second-pass verification

- `python -m pytest -q` — 9 tests passed after the second-pass changes.
- `ruff check .` — passed.
- `python -m compileall -q glass_quality.py tests` — passed.
- Notebook JSON, code-cell AST parsing, empty outputs, and null execution counts — passed.
- Git/status and repository-wide scans found no tracked credentials, hardcoded machine paths, unsafe deserialization, network download, or untracked source/config files required by the workflow.
- No challenge data was created, downloaded, or used; therefore no model metric, submission artifact, or end-to-end run is claimed.

### Remaining risks after the second review

- The authorized challenge data is absent, so real schema values, class balance, runtime, calibration, and competition submission compatibility remain unverified.
- Feature exclusion is name-based and cannot prove domain-specific leakage absent provenance; the baseline still requires domain review of the challenge columns.
- The notebook writes an `.xlsx` probability table, but the competition's exact submission schema is not available for confirmation.
- Dependency ranges are bounded major-version ranges rather than a lockfile; exact transitive environments may differ.
